#!/usr/bin/env python

"""Measure the VLM/action task-effective dimension gap in PI0.5.

This is a lightweight, paper-motivation diagnostic rather than a training script.
It loads one PI0.5 checkpoint and one small LIBERO batch, computes a task-conditioned
empirical tangent-kernel proxy for the VLM and action pathway separately, and reports
spectral effective dimensions.

Why a proxy instead of the exact NTK?
-------------------------------------
The exact vector-output NTK of a multi-billion-parameter VLA is prohibitively large.
For a scalar per-sample VLA loss l_i, we use the gradient feature

    g_i^m = d l_i / d theta_m,

for module m in {VLM, Action}.  The Gram matrix

    K_m[i,j] = <g_i^m, g_j^m>

is a task-conditioned empirical tangent/Fisher kernel.  We preserve its pairwise
inner products approximately with a deterministic CountSketch before forming K_m.
The resulting spectrum is suitable for comparing *relative downstream adaptation
complexity* between the two modules on the same data and objective.

Recommended paper use
---------------------
Run this at the same initialization/checkpoint for both modules and report:
  * spectral effective rank (entropy rank),
  * participation ratio,
  * d90: number of eigenmodes explaining 90% spectral mass,
  * Action/VLM ratios for the above quantities.

A gap > 1 means that, under the same downstream VLA objective, the action pathway
uses a higher-dimensional task-conditioned tangent space than the VLM.

Example
-------
python -m lerobot.scripts.lerobot_ntk_gap \
  --policy-path /data/models/lerobot/pi05_libero_base \
  --dataset-root /data/datasets/libero \
  --dataset-repo-id libero \
  --batch-size 4 \
  --sketch-dim 256 \
  --video-backend pyav \
  --output /data/wyn/ntk_gap/pi05_base.json

For a clean motivation experiment, use a single GPU and disable torch.compile in
any surrounding training process.  This script does not modify model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05TrainingStage
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION


def _unique_parameters(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            pid = id(parameter)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(parameter)
    return params


def _countsketch_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    *,
    sketch_dim: int,
    chunk_size: int,
) -> torch.Tensor:
    """Deterministically CountSketch one flattened module gradient.

    CountSketch gives an unbiased inner-product estimator while avoiding a dense
    [num_parameters, sketch_dim] random projection.  The implementation hashes
    each flattened gradient element into one bucket with one deterministic sign.
    """
    if sketch_dim <= 0:
        raise ValueError(f"sketch_dim must be > 0, got {sketch_dim}")

    device = next((g.device for g in gradients if g is not None), parameters[0].device)
    sketch = torch.zeros(sketch_dim, device=device, dtype=torch.float32)
    global_offset = 0

    # Fixed integer hash constants.  We use element index + global parameter offset
    # so the same virtual random projection is reused for every sample.
    hash_a = 1_103_515_245
    hash_b = 12_345
    sign_a = 2_654_435_761
    mask63 = (1 << 63) - 1

    for parameter, gradient in zip(parameters, gradients, strict=True):
        numel = parameter.numel()
        if gradient is not None:
            flat = gradient.detach().reshape(-1).to(dtype=torch.float32)
            for start in range(0, numel, chunk_size):
                end = min(start + chunk_size, numel)
                local = torch.arange(start, end, device=device, dtype=torch.int64)
                absolute = local + global_offset
                # Keep arithmetic in signed int64 range.
                mixed = (absolute * hash_a + hash_b) & mask63
                buckets = torch.remainder(mixed, sketch_dim)
                sign_bits = ((absolute * sign_a + hash_b) & mask63) & 1
                signs = sign_bits.to(torch.float32).mul_(2.0).sub_(1.0)
                sketch.scatter_add_(0, buckets, flat[start:end] * signs)
        global_offset += numel

    return sketch


def _kernel_metrics(sketches: torch.Tensor, energy_threshold: float) -> dict[str, object]:
    # [N, R] -> [N, N].  CPU float64 makes the tiny eigendecomposition stable.
    kernel = (sketches @ sketches.T).detach().cpu().to(torch.float64)
    kernel = 0.5 * (kernel + kernel.T)
    eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0).flip(0)
    trace = eigenvalues.sum()

    if trace <= 0:
        return {
            "trace": 0.0,
            "eigenvalues": eigenvalues.tolist(),
            "normalized_eigenvalues": [0.0 for _ in eigenvalues],
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "d90": 0,
        }

    p = eigenvalues / trace
    nonzero = p[p > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    effective_rank = entropy.exp()
    participation_ratio = trace.square() / eigenvalues.square().sum().clamp_min(torch.finfo(torch.float64).eps)
    cumulative = torch.cumsum(p, dim=0)
    d90 = int(torch.searchsorted(cumulative, torch.tensor(energy_threshold, dtype=cumulative.dtype)).item()) + 1

    return {
        "trace": float(trace.item()),
        "eigenvalues": [float(x) for x in eigenvalues.tolist()],
        "normalized_eigenvalues": [float(x) for x in p.tolist()],
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "d90": d90,
    }


def _module_sketches(
    per_sample_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    sketch_dim: int,
    chunk_size: int,
    retain_after_last: bool,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    n = per_sample_loss.numel()
    for index in range(n):
        retain_graph = retain_after_last or index < n - 1
        gradients = torch.autograd.grad(
            per_sample_loss[index],
            parameters,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
            materialize_grads=False,
        )
        rows.append(
            _countsketch_gradients(
                gradients,
                parameters,
                sketch_dim=sketch_dim,
                chunk_size=chunk_size,
            )
        )
    return torch.stack(rows, dim=0)


def analyze_task_effective_dimension(
    policy: PI05Policy,
    batch: dict[str, torch.Tensor],
    *,
    num_samples: int,
    sketch_dim: int,
    chunk_size: int,
    energy_threshold: float,
    seed: int,
) -> dict[str, object]:
    """Compare VLM and action task-conditioned tangent-space dimensions."""
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1]")

    # Keep a small, identical sample set for the two modules.
    actual_batch = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor))
    n = min(num_samples, actual_batch)
    batch = {
        key: (value[:n] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == actual_batch else value)
        for key, value in batch.items()
    }

    # Probe the formal observation-conditioned VLA objective regardless of how a
    # staged-training checkpoint was produced.  We restore all flags afterwards.
    original_stage = policy.config.training_stage
    original_mode = policy.training
    vlm_parameters = _unique_parameters(policy._vlm_modules())
    action_parameters = _unique_parameters(policy._action_modules())
    all_parameters = vlm_parameters + action_parameters
    original_requires_grad = [parameter.requires_grad for parameter in all_parameters]

    try:
        policy.config.training_stage = PI05TrainingStage.FLOW
        policy.eval()
        for parameter in all_parameters:
            parameter.requires_grad_(True)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        per_sample_loss, loss_dict = policy.forward(batch, reduction="none")
        if per_sample_loss.ndim != 1:
            raise RuntimeError(f"Expected per-sample PI0.5 loss, got shape {tuple(per_sample_loss.shape)}")

        # Two passes avoid materializing VLM and action gradients simultaneously.
        # The VLM pass retains the graph; the action pass releases it on its final sample.
        vlm_sketches = _module_sketches(
            per_sample_loss,
            vlm_parameters,
            sketch_dim=sketch_dim,
            chunk_size=chunk_size,
            retain_after_last=True,
        )
        action_sketches = _module_sketches(
            per_sample_loss,
            action_parameters,
            sketch_dim=sketch_dim,
            chunk_size=chunk_size,
            retain_after_last=False,
        )

        vlm = _kernel_metrics(vlm_sketches, energy_threshold)
        action = _kernel_metrics(action_sketches, energy_threshold)

        eps = 1e-12
        gap = {
            "effective_rank_ratio_action_over_vlm": float(action["effective_rank"] / (vlm["effective_rank"] + eps)),
            "participation_ratio_action_over_vlm": float(
                action["participation_ratio"] / (vlm["participation_ratio"] + eps)
            ),
            "d90_ratio_action_over_vlm": float(action["d90"] / max(vlm["d90"], 1)),
            # Trace is scale-sensitive; report it as an optimization-strength diagnostic,
            # not as the primary task-effective dimension metric.
            "kernel_trace_ratio_action_over_vlm": float(action["trace"] / (vlm["trace"] + eps)),
        }

        return {
            "method": "task-conditioned empirical tangent kernel with deterministic CountSketch",
            "note": (
                "This is a practical loss-gradient tangent/Fisher-kernel proxy, not the exact vector-output NTK. "
                "Use spectral-shape metrics (effective rank, participation ratio, d90) as the primary complexity comparison."
            ),
            "num_samples": n,
            "sketch_dim": sketch_dim,
            "energy_threshold": energy_threshold,
            "seed": seed,
            "mean_vla_loss": float(per_sample_loss.detach().mean().item()),
            "vlm_num_parameters": int(sum(p.numel() for p in vlm_parameters)),
            "action_num_parameters": int(sum(p.numel() for p in action_parameters)),
            "vlm": vlm,
            "action": action,
            "gap": gap,
            "loss_metadata": loss_dict,
        }
    finally:
        policy.config.training_stage = original_stage
        policy.train(original_mode)
        for parameter, requires_grad in zip(all_parameters, original_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", default="libero")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--sketch-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--energy-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("ntk_gap.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < args.num_samples:
        raise ValueError("batch-size must be >= num-samples")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    # Load the checkpoint config, but probe both modules as trainable under the
    # formal observation-conditioned objective.  This does not alter saved weights.
    config = PI05Config.from_pretrained(args.policy_path, local_files_only=True)
    config.device = args.device
    config.training_stage = PI05TrainingStage.FLOW
    config.train_expert_only = False
    config.freeze_vision_encoder = False
    config.compile_model = False
    config.gradient_checkpointing = False

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )

    policy = PI05Policy.from_pretrained(
        args.policy_path,
        config=config,
        local_files_only=True,
        strict=False,
    )

    preprocessor_overrides = {
        "device_processor": {"device": torch.device(args.device).type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
    )

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    batch = next(iter(dataloader))
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(batch)

    result = analyze_task_effective_dimension(
        policy,
        batch,
        num_samples=args.num_samples,
        sketch_dim=args.sketch_dim,
        chunk_size=args.sketch_chunk_size,
        energy_threshold=args.energy_threshold,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== PI0.5 Cross-Module Task Complexity Gap ===")
    print(f"VLM    effective_rank={result['vlm']['effective_rank']:.3f}  d90={result['vlm']['d90']}")
    print(f"Action effective_rank={result['action']['effective_rank']:.3f}  d90={result['action']['d90']}")
    print(
        "Gap    effective_rank(Action/VLM)="
        f"{result['gap']['effective_rank_ratio_action_over_vlm']:.3f}, "
        f"d90(Action/VLM)={result['gap']['d90_ratio_action_over_vlm']:.3f}"
    )
    print(f"Saved full spectrum to: {args.output}")


if __name__ == "__main__":
    main()
