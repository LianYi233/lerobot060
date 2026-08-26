#!/usr/bin/env python

"""Memory-safe VLM/action task-effective-dimension diagnostic for PI0.5.

The script estimates a task-conditioned empirical tangent/Fisher kernel for the
VLM and action pathway under the same observation-conditioned VLA objective.
To keep memory bounded on multi-billion-parameter PI0.5 models, samples are
processed one at a time. For each sample, the VLM and action gradients are
immediately CountSketched and then released before the next sample is loaded.

This is a practical scalar-loss tangent-kernel proxy, not the exact vector-output
NTK. The main paper-facing statistics are spectral effective rank, participation
ratio, d90, and Action/VLM ratios of those quantities.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05TrainingStage
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.collate import lerobot_collate_fn


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
    """CountSketch one flattened module gradient without a dense projection."""
    if sketch_dim <= 0:
        raise ValueError(f"sketch_dim must be > 0, got {sketch_dim}")

    device = next((g.device for g in gradients if g is not None), parameters[0].device)
    sketch = torch.zeros(sketch_dim, device=device, dtype=torch.float32)
    global_offset = 0
    hash_a = 1_103_515_245
    hash_b = 12_345
    sign_a = 2_654_435_761
    mask63 = (1 << 63) - 1

    for parameter, gradient in zip(parameters, gradients, strict=True):
        numel = parameter.numel()
        if gradient is not None:
            flat = gradient.detach().reshape(-1)
            for start in range(0, numel, chunk_size):
                end = min(start + chunk_size, numel)
                local = torch.arange(start, end, device=device, dtype=torch.int64)
                absolute = local + global_offset
                mixed = (absolute * hash_a + hash_b) & mask63
                buckets = torch.remainder(mixed, sketch_dim)
                sign_bits = ((absolute * sign_a + hash_b) & mask63) & 1
                signs = sign_bits.to(torch.float32).mul_(2.0).sub_(1.0)
                values = flat[start:end].to(dtype=torch.float32)
                sketch.scatter_add_(0, buckets, values * signs)
                del local, absolute, mixed, buckets, sign_bits, signs, values
        global_offset += numel

    return sketch.detach().cpu()


def _kernel_metrics(sketches: torch.Tensor, energy_threshold: float) -> dict[str, object]:
    kernel = (sketches @ sketches.T).to(torch.float64)
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
    nz = p[p > 0]
    effective_rank = (-(nz * nz.log()).sum()).exp()
    participation_ratio = trace.square() / eigenvalues.square().sum().clamp_min(
        torch.finfo(torch.float64).eps
    )
    cumulative = torch.cumsum(p, dim=0)
    d90 = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(energy_threshold, dtype=cumulative.dtype),
        ).item()
    ) + 1
    return {
        "trace": float(trace.item()),
        "eigenvalues": [float(x) for x in eigenvalues.tolist()],
        "normalized_eigenvalues": [float(x) for x in p.tolist()],
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "d90": d90,
    }


def _prepare_one_batch(batch, dataset, preprocessor):
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    return preprocessor(batch)


def _gradient_sketch_for_group(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    sketch_dim: int,
    chunk_size: int,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
        materialize_grads=False,
    )
    sketch = _countsketch_gradients(
        gradients,
        parameters,
        sketch_dim=sketch_dim,
        chunk_size=chunk_size,
    )
    del gradients
    return sketch


def analyze_streaming(
    policy: PI05Policy,
    dataloader,
    dataset,
    preprocessor,
    *,
    num_samples: int,
    sketch_dim: int,
    chunk_size: int,
    energy_threshold: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    if num_samples < 2:
        raise ValueError("num-samples must be >= 2 for a meaningful spectrum")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy-threshold must be in (0, 1]")

    original_stage = policy.config.training_stage
    original_mode = policy.training
    vlm_parameters = _unique_parameters(policy._vlm_modules())
    action_parameters = _unique_parameters(policy._action_modules())
    all_parameters = vlm_parameters + action_parameters
    original_requires_grad = [p.requires_grad for p in all_parameters]

    vlm_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    losses: list[float] = []
    loss_metadata: dict[str, object] = {}

    try:
        policy.config.training_stage = PI05TrainingStage.FLOW
        policy.eval()
        for p in all_parameters:
            p.requires_grad_(True)

        iterator = iter(dataloader)
        for sample_index in range(num_samples):
            # Keep flow noise/time reproducible per sample and identical for both module groups.
            sample_seed = seed + sample_index
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)

            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Dataset exhausted after {sample_index} samples; requested {num_samples}."
                ) from exc

            batch = _prepare_one_batch(batch, dataset, preprocessor)
            per_sample_loss, current_metadata = policy.forward(batch, reduction="none")
            if per_sample_loss.numel() != 1:
                raise RuntimeError(
                    "Memory-safe NTK diagnostic expects DataLoader batch_size=1; "
                    f"got per-sample loss shape {tuple(per_sample_loss.shape)}"
                )
            loss = per_sample_loss.reshape(())
            losses.append(float(loss.detach().item()))
            loss_metadata = current_metadata

            # VLM gradient first, retaining only this single-sample graph for the action pass.
            vlm_rows.append(
                _gradient_sketch_for_group(
                    loss,
                    vlm_parameters,
                    sketch_dim=sketch_dim,
                    chunk_size=chunk_size,
                    retain_graph=True,
                )
            )
            action_rows.append(
                _gradient_sketch_for_group(
                    loss,
                    action_parameters,
                    sketch_dim=sketch_dim,
                    chunk_size=chunk_size,
                    retain_graph=False,
                )
            )

            del loss, per_sample_loss, batch, current_metadata
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"[NTK gap] processed sample {sample_index + 1}/{num_samples}", flush=True)

        vlm_sketches = torch.stack(vlm_rows, dim=0)
        action_sketches = torch.stack(action_rows, dim=0)
        vlm = _kernel_metrics(vlm_sketches, energy_threshold)
        action = _kernel_metrics(action_sketches, energy_threshold)
        eps = 1e-12
        gap = {
            "effective_rank_ratio_action_over_vlm": float(
                action["effective_rank"] / (vlm["effective_rank"] + eps)
            ),
            "participation_ratio_action_over_vlm": float(
                action["participation_ratio"] / (vlm["participation_ratio"] + eps)
            ),
            "d90_ratio_action_over_vlm": float(action["d90"] / max(vlm["d90"], 1)),
            "kernel_trace_ratio_action_over_vlm": float(action["trace"] / (vlm["trace"] + eps)),
        }
        return {
            "method": "streaming task-conditioned empirical tangent kernel with deterministic CountSketch",
            "note": (
                "Scalar-loss tangent/Fisher-kernel proxy; not the exact vector-output NTK. "
                "Use spectral-shape metrics as the primary complexity comparison."
            ),
            "num_samples": num_samples,
            "sketch_dim": sketch_dim,
            "energy_threshold": energy_threshold,
            "seed": seed,
            "mean_vla_loss": float(sum(losses) / len(losses)),
            "vlm_num_parameters": int(sum(p.numel() for p in vlm_parameters)),
            "action_num_parameters": int(sum(p.numel() for p in action_parameters)),
            "vlm": vlm,
            "action": action,
            "gap": gap,
            "loss_metadata": loss_metadata,
        }
    finally:
        policy.config.training_stage = original_stage
        policy.train(original_mode)
        for p, requires_grad in zip(all_parameters, original_requires_grad, strict=True):
            p.requires_grad_(requires_grad)
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", default="libero")
    parser.add_argument("--video-backend", default="pyav")
    # Kept for CLI compatibility; the diagnostic intentionally forces microbatch=1.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--sketch-chunk-size", type=int, default=250_000)
    parser.add_argument("--energy-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("ntk_gap.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.batch_size != 1:
        print(
            f"[NTK gap] ignoring --batch-size={args.batch_size}; using batch_size=1 to bound memory.",
            flush=True,
        )

    config = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config at {args.policy_path}, got {type(config).__name__}")
    config.device = args.device
    config.training_stage = PI05TrainingStage.FLOW
    config.train_expert_only = False
    config.freeze_vision_encoder = False
    config.compile_model = False
    # Keep checkpointing off here because the diagnostic uses autograd.grad over explicit
    # parameter lists; streaming batch_size=1 is the primary memory-control mechanism.
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    result = analyze_streaming(
        policy,
        dataloader,
        dataset,
        preprocessor,
        num_samples=args.num_samples,
        sketch_dim=args.sketch_dim,
        chunk_size=args.sketch_chunk_size,
        energy_threshold=args.energy_threshold,
        seed=args.seed,
        device=args.device,
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
