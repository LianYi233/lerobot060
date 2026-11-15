#!/usr/bin/env python3
"""Two-action-chunk SAP probe for PrimingVLA.

The train command uses the actual PI0.5 action-inpainting forward and trains only
``model.prompt_tokens.weight``. No visual/language observations are loaded.

The analyze command evaluates the same two chunks with *new* masks and noise,
and tests whether the reconstruction depends on correctly ordered visible actions.
It is an in-sample mechanism diagnostic, not a generalization benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def parse_episode_offsets(spec: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse two ``episode:offset`` selectors, including robust validation."""
    selectors = spec.split(",")
    if len(selectors) != 2:
        raise ValueError("Exactly two episode:offset selectors are required, e.g. 0:10,1:10")
    pairs = []
    for value in selectors:
        values = value.strip().split(":")
        if len(values) != 2 or any(not part.isdecimal() for part in values):
            raise ValueError("Selectors must be nonnegative episode:offset pairs")
        pairs.append((int(values[0]), int(values[1])))
    if len(set(pairs)) != 2:
        raise ValueError("The two action chunks must have different selectors")
    return pairs[0], pairs[1]


def select_chunk_starts(
    episode_records: list[dict],
    selectors: tuple[tuple[int, int], tuple[int, int]],
    horizon: int,
) -> list[int]:
    """Select non-overlapping, unpadded full chunks by absolute dataset index."""
    selected = []
    ranges = []
    for episode, offset in selectors:
        if episode >= len(episode_records):
            raise ValueError(f"Episode {episode} is outside [0, {len(episode_records)})")
        record = episode_records[episode]
        lo, hi = int(record["dataset_from_index"]), int(record["dataset_to_index"])
        start = lo + offset
        if start + horizon > hi:
            raise ValueError(
                f"Episode {episode} has {hi - lo} frames; offset {offset} cannot fit "
                f"an unpadded {horizon}-step chunk"
            )
        selected.append(start)
        ranges.append((start, start + horizon))
    (a0, a1), (b0, b1) = ranges
    if max(a0, b0) < min(a1, b1):
        raise ValueError("The two chunks overlap; choose two different episodes or distant offsets")
    return selected


def quantile_normalize(raw: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Match the repository's PI0.5 QUANTILES normalization in float32."""
    raw = np.asarray(raw, dtype=np.float32)
    low = np.asarray(q01, dtype=np.float32)
    denom = np.asarray(q99, dtype=np.float32) - low
    denom = np.where(denom == 0, np.float32(1e-8), denom)
    return (np.float32(2.0) * (raw - low) / denom - np.float32(1.0)).astype(np.float32)


def interpolate_visible(actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-joint linear interpolation of observed timesteps, with endpoint holding."""
    if actions.ndim != 3 or mask.shape != actions.shape[:2]:
        raise ValueError("Expected actions [batch, horizon, dim] and mask [batch, horizon]")
    horizon = actions.shape[1]
    result = np.empty_like(actions)
    for batch in range(actions.shape[0]):
        visible = np.flatnonzero(~mask[batch])
        if len(visible) == 0:
            result[batch] = np.mean(actions[batch], axis=0, keepdims=True)
            continue
        for dim in range(actions.shape[2]):
            result[batch, :, dim] = np.interp(
                np.arange(horizon), visible, actions[batch, visible, dim]
            )
    return result


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if prediction.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("Prediction, target, and mask shapes are incompatible")
    if not mask.any():
        raise ValueError("At least one timestep must be masked")
    return float(np.mean((prediction[mask] - target[mask]) ** 2))


def _get_torch():
    # Keep numerical helpers importable for CPU unit tests without the ML stack.
    import torch

    return torch


def _configure_pi05(base: Path, device: str, *, gradient_checkpointing: bool = True):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05TrainingStage

    cfg = PreTrainedConfig.from_pretrained(base)
    if not isinstance(cfg, PI05Config):
        raise TypeError(f"Expected a PI05 checkpoint config, found {type(cfg).__name__}")
    cfg.training_stage = PI05TrainingStage.NEXT_ACTION
    cfg.next_action_pretrain_steps = 0
    cfg.next_action_bridge_steps = 0
    cfg.next_action_masked_steps = 40
    cfg.next_action_full_mask_probability = 0.0
    cfg.num_prompt_tokens = 16
    cfg.num_vlm_prompt_tokens = 16
    cfg.cabo_enabled = False  # CPC is off in Stage I in the original PrimingVLA.
    cfg.dtype = "float32"
    cfg.device = device
    cfg.compile_model = False
    cfg.gradient_checkpointing = gradient_checkpointing
    # The original Stage-I recipe, also specified in lerobot_train.py.
    cfg.optimizer_lr = 2.5e-5
    cfg.optimizer_betas = (0.9, 0.95)
    cfg.optimizer_eps = 1e-8
    cfg.optimizer_weight_decay = 0.01
    cfg.scheduler_warmup_steps = 1_000
    cfg.scheduler_decay_steps = 30_000
    cfg.scheduler_decay_lr = 1e-5
    return cfg


def _load_real_action_chunks(args, cfg):
    """Use the original LeRobot index and dataset quantiles, but never decode images."""
    import torch

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend="pyav")
    horizon = cfg.chunk_size
    records = [meta.episodes[i] for i in range(meta.total_episodes)]
    selectors = parse_episode_offsets(args.chunks)
    starts = select_chunk_starts(records, selectors, horizon)

    # The original training pipeline takes full action chunks via action_delta_indices.
    # Reading HF action columns directly avoids decoding unused camera videos.
    chunks = []
    for start in starts:
        values = dataset.hf_dataset[list(range(start, start + horizon))][ACTION]
        chunks.append(torch.stack(values).to(dtype=torch.float32).cpu().numpy())
    raw = np.stack(chunks, axis=0)
    if raw.ndim != 3 or raw.shape[1] != horizon:
        raise ValueError(f"Expected two action chunks of shape [2, {horizon}, D], got {raw.shape}")
    if raw.shape[2] > cfg.max_action_dim:
        raise ValueError("Dataset action dimension exceeds PI0.5 max_action_dim")
    stats = meta.stats[ACTION]
    normalized = quantile_normalize(raw, stats["q01"].numpy(), stats["q99"].numpy())
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized actions; check dataset action quantile statistics")
    return normalized, selectors, starts


def _set_seed(seed: int):
    torch = _get_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _checkpoint_prompt(policy):
    return policy.model.prompt_tokens.weight.detach().float().cpu().clone()


def train(args):
    torch = _get_torch()
    from lerobot.policies.pi05.configuration_pi05 import PI05TrainingObjective
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, pad_vector

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.masked_steps < 1 or args.masked_steps >= 50:
        raise ValueError("--masked-steps must be in [1, 49]")

    _set_seed(args.seed)
    cfg = _configure_pi05(args.base_checkpoint, args.device)
    cfg.next_action_masked_steps = args.masked_steps
    normalized, selectors, starts = _load_real_action_chunks(args, cfg)
    data_hash = hashlib.sha256(normalized.tobytes()).hexdigest()

    # Only create output after validating all input paths and action windows.
    policy = PI05Policy.from_pretrained(args.base_checkpoint, config=cfg, local_files_only=True)
    policy.train()
    for name, param in policy.named_parameters():
        param.requires_grad_(name == "model.prompt_tokens.weight")
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if len(trainable) != 1 or trainable[0][0] != "model.prompt_tokens.weight":
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in trainable]}")

    optimizer = cfg.get_optimizer_preset().build([trainable[0][1]])
    scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
    prompt_initial = _checkpoint_prompt(policy)
    action_dim = normalized.shape[-1]
    actions = torch.as_tensor(normalized, dtype=torch.float32, device=args.device)
    actions = pad_vector(actions, cfg.max_action_dim)
    action_is_pad = torch.zeros((2, cfg.chunk_size), dtype=torch.bool, device=args.device)

    # Independent, per-update permutation removes any consistent temporal
    # association while preserving all 50 vectors and their joint marginals.
    shuffle_rng = np.random.default_rng(args.seed + 2026)
    if args.variant == "full_mask":
        cfg.next_action_full_mask_probability = 1.0

    output.mkdir(parents=True)
    torch.save(
        {
            "actions": torch.from_numpy(normalized),
            "selectors": selectors,
            "absolute_starts": starts,
            "sha256": data_hash,
        },
        output / "two_chunks.pt",
    )
    trace = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        inpainting_mask = policy.model.sample_inpainting_mask(action_is_pad)
        step_actions = actions
        if args.variant == "shuffled":
            step_actions = actions[:, shuffle_rng.permutation(cfg.chunk_size), :]
        losses = policy.model.forward(
            actions=step_actions,
            action_is_pad=action_is_pad,
            inpainting_mask=inpainting_mask,
            objective=PI05TrainingObjective.ACTION_INPAINTING,
        )
        # Exact Stage-I reduction: supervise only masked real action dimensions.
        loss = losses[:, :, :action_dim].sum() / (inpainting_mask.sum() * action_dim)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            value = float(loss.detach().item())
            trace.append({"step": step, "loss": value, "lr": optimizer.param_groups[0]["lr"]})
            print(f"[{args.variant}] step={step:04d}/{args.steps} loss={value:.6f}", flush=True)

    prompt_trained = _checkpoint_prompt(policy)
    torch.save(
        {
            "prompt_initial": prompt_initial,
            "prompt_trained": prompt_trained,
        },
        output / "action_prompt.pt",
    )
    _save_json(
        output / "train_metadata.json",
        {
            "variant": args.variant,
            "steps": args.steps,
            "seed": args.seed,
            "dataset_root": str(args.dataset_root),
            "dataset_repo_id": args.dataset_repo_id,
            "selectors": selectors,
            "absolute_starts": starts,
            "action_sha256": data_hash,
            "base_checkpoint": str(args.base_checkpoint),
            "masked_steps": args.masked_steps,
            "shuffle_protocol": "independent per-update temporal permutation (seed+2026)"
            if args.variant == "shuffled" else None,
            "trainable_parameters": int(prompt_trained.numel()),
            "objective": "original pi05 action-only masked flow matching, CPC disabled",
            "loss_trace": trace,
        },
    )
    print(f"Saved: {output}")


def _build_probes(actions, repeat: int, seed: int, masked_steps: int):
    """Sample identical masks and Gaussian noise for every evaluated model."""
    torch = _get_torch()
    generator = torch.Generator(device="cpu").manual_seed(seed + repeat * 7919)
    batch, horizon, _ = actions.shape
    masked = torch.zeros((batch, horizon), dtype=torch.bool)
    for b in range(batch):
        masked[b, torch.randperm(horizon, generator=generator)[:masked_steps]] = True
    noise = torch.randn(actions.shape, generator=generator)
    shuffled_context = actions.clone()
    for b in range(batch):
        visible = torch.nonzero(~masked[b]).flatten()
        order = torch.randperm(len(visible), generator=generator)
        shuffled_context[b, visible] = actions[b, visible[order]]
    return masked, noise, shuffled_context


def _predict_clean(core, clean, noise, mask, t: float, context_override=None):
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import pad_vector

    batch, horizon, dims = clean.shape
    x = torch.where(mask.unsqueeze(-1), (1 - t) * clean + t * noise, clean)
    if context_override is not None:
        x = torch.where(mask.unsqueeze(-1), x, context_override)
    x_pad = pad_vector(x, core.config.max_action_dim)
    is_pad = torch.zeros((batch, horizon), device=x.device, dtype=torch.bool)
    # Single-step clean-action estimate, justified by x_t=a+t(z-a).
    with torch.no_grad():
        velocity = core.predict_inpainting_velocity(
            x_pad,
            torch.full((batch,), t, device=x.device, dtype=torch.float32),
            is_pad,
            mask,
        )
    result = x_pad - t * velocity
    return result[:, :, :dims]


def _directional_influence(core, clean, noise, mask, t: float, eps: float, dim: int):
    """Finite-difference column sensitivities: a directional Jacobian probe.

    Perturb ONLY visible context; keep the masked noisy target fixed. Non-visible
    columns are intentionally NaN, since they provide no contextual information.
    """
    torch = _get_torch()
    horizon = clean.shape[1]
    influence = np.full((horizon, horizon), np.nan, dtype=np.float32)
    for index in torch.nonzero(~mask[0]).flatten().tolist():
        plus = clean.clone()
        minus = clean.clone()
        plus[0, index, dim] += eps
        minus[0, index, dim] -= eps
        upper = _predict_clean(core, clean, noise, mask, t, context_override=plus)
        lower = _predict_clean(core, clean, noise, mask, t, context_override=minus)
        sensitivity = torch.linalg.vector_norm((upper - lower)[0], dim=-1) / (2 * eps)
        influence[mask[0].cpu().numpy(), index] = sensitivity[mask[0]].float().cpu().numpy()
    return influence


def analyze(args):
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    runs = {name: Path(path).resolve() for name, path in (
        ("sap", args.sap), ("shuffled", args.shuffled), ("full_mask", args.full_mask)
    )}
    samples = []
    metadata = {}
    prompts = {}
    for label, directory in runs.items():
        metadata[label] = json.loads((directory / "train_metadata.json").read_text())
        if metadata[label]["variant"] != label:
            raise ValueError(f"{directory} is {metadata[label]['variant']}, expected {label}")
        sample = torch.load(directory / "two_chunks.pt", map_location="cpu", weights_only=True)
        samples.append(sample)
        prompts[label] = torch.load(directory / "action_prompt.pt", map_location="cpu", weights_only=True)
    if not 1 <= args.masked_steps < int(samples[0]["actions"].shape[1]):
        raise ValueError("--masked-steps must leave at least one visible action timestep")
    if len({s["sha256"] for s in samples}) != 1:
        raise ValueError("Controls did not train on the exact same two normalized chunks")
    if len({(m["seed"], m["steps"], m["base_checkpoint"]) for m in metadata.values()}) != 1:
        raise ValueError("Controls must share the same seed, updates, and pretrained checkpoint")
    for label in ("shuffled", "full_mask"):
        if not torch.equal(prompts["sap"]["prompt_initial"], prompts[label]["prompt_initial"]):
            raise ValueError("Controls must share identical action-prompt initialization")

    _set_seed(args.eval_seed)
    cfg = _configure_pi05(Path(metadata["sap"]["base_checkpoint"]), args.device, gradient_checkpointing=False)
    cfg.next_action_masked_steps = args.masked_steps
    policy = PI05Policy.from_pretrained(metadata["sap"]["base_checkpoint"], config=cfg, local_files_only=True)
    policy.eval()
    ground_truth = samples[0]["actions"].to(args.device, dtype=torch.float32)
    action_dim = ground_truth.shape[-1]
    if not 0 <= args.influence_dim < action_dim:
        raise ValueError(f"--influence-dim must be in [0, {action_dim})")

    evaluations = {name: {"mse": [], "context_shuffled_mse": []} for name in ("init", *runs)}
    interpolations = []
    saved_curve = {}
    fixed_probe = None
    for rep in range(args.repeats):
        mask_cpu, noise_cpu, corrupted_cpu = _build_probes(
            samples[0]["actions"], rep, args.eval_seed, args.masked_steps
        )
        mask = mask_cpu.to(args.device)
        noise = noise_cpu.to(args.device)
        shuffled_context = corrupted_cpu.to(args.device)
        baseline = interpolate_visible(samples[0]["actions"].numpy(), mask_cpu.numpy())
        interpolations.append(masked_mse(baseline, samples[0]["actions"].numpy(), mask_cpu.numpy()))
        if rep == 0:
            fixed_probe = (mask, noise)

        for label in evaluations:
            prompt = prompts["sap"]["prompt_initial"] if label == "init" else prompts[label]["prompt_trained"]
            with torch.no_grad():
                policy.model.prompt_tokens.weight.copy_(prompt.to(args.device))
            pred = _predict_clean(policy.model, ground_truth, noise, mask, args.t)
            corrupted = _predict_clean(
                policy.model, ground_truth, noise, mask, args.t, context_override=shuffled_context
            )
            eval_mse = masked_mse(pred.cpu().numpy(), ground_truth.cpu().numpy(), mask_cpu.numpy())
            shuffled_mse = masked_mse(
                corrupted.cpu().numpy(), ground_truth.cpu().numpy(), mask_cpu.numpy()
            )
            evaluations[label]["mse"].append(eval_mse)
            evaluations[label]["context_shuffled_mse"].append(shuffled_mse)
            if rep == 0:
                saved_curve[label] = pred[0].detach().cpu().numpy()

    summary = {}
    for label, record in evaluations.items():
        correct = float(np.mean(record["mse"]))
        corrupt = float(np.mean(record["context_shuffled_mse"]))
        summary[label] = {
            "masked_mse": correct,
            "shuffled_visible_context_mse": corrupt,
            "context_order_gain": corrupt - correct,
            "context_order_gain_relative": (corrupt - correct) / max(correct, 1e-12),
        }
    summary["linear_interpolation"] = {"masked_mse": float(np.mean(interpolations))}
    summary["protocol"] = {
        "train_examples": 2,
        "action_horizon": int(ground_truth.shape[1]),
        "action_dim": action_dim,
        "masked_steps": args.masked_steps,
        "evaluation_repeats": args.repeats,
        "time_t": args.t,
        "shared_sample_sha256": samples[0]["sha256"],
        "interpretation": "Held-out masks/noise on the SAME two chunks; not independent test trajectories",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    _save_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # A visual readout uses the dimension with most temporal variation in chunk 1.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})
    mask_first = fixed_probe[0][0].cpu().numpy()
    original = ground_truth[0].cpu().numpy()
    dim = int(np.argmax(original.std(axis=0)))
    x = np.arange(len(original))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9), layout="constrained")
    axes[0].plot(x, original[:, dim], color="#1f2937", lw=2, label="Ground truth")
    axes[0].scatter(x[~mask_first], original[~mask_first, dim], color="#111827", s=28, zorder=5, label="10 visible")
    for label, color in (("sap", "#006d77"), ("shuffled", "#c57b39"), ("full_mask", "#a35076")):
        axes[0].plot(x, saved_curve[label][:, dim], lw=1.45, label=label, color=color, alpha=0.85)
    axes[0].set(xlabel="Action timestep", ylabel=f"Normalized action, dim {dim}", title="(a) Masked-action reconstruction")
    axes[0].legend(frameon=False, fontsize=8)
    names = ["init", "sap", "shuffled", "full_mask", "linear_interpolation"]
    values = [summary[n]["masked_mse"] for n in names]
    axes[1].bar(np.arange(len(names)), values, color=["#929aa4", "#006d77", "#c57b39", "#a35076", "#84a59d"])
    axes[1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[1].set(ylabel="Masked MSE (lower better)", title="(b) Same two chunks, new random masks")
    for i, y in enumerate(values):
        axes[1].text(i, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    width = 0.35
    names = ["init", "sap", "shuffled", "full_mask"]
    locations = np.arange(len(names))
    axes[2].bar(locations - width / 2, [summary[n]["masked_mse"] for n in names], width, color="#006d77", label="Correct visible actions")
    axes[2].bar(locations + width / 2, [summary[n]["shuffled_visible_context_mse"] for n in names], width, color="#c57b39", label="Permuted visible actions")
    axes[2].set_xticks(locations, names, rotation=25, ha="right")
    axes[2].set(ylabel="Masked MSE (lower better)", title="(c) Context-order intervention")
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(args.output / "two_sample_structure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    influence_grids = {}
    for label in ("init", "sap"):
        prompt = prompts["sap"]["prompt_initial"] if label == "init" else prompts[label]["prompt_trained"]
        with torch.no_grad():
            policy.model.prompt_tokens.weight.copy_(prompt.to(args.device))
        influence_grids[label] = _directional_influence(
            policy.model, ground_truth, fixed_probe[1], fixed_probe[0],
            args.t, args.influence_epsilon, args.influence_dim
        )
    vmax = max(np.nanmax(a) for a in influence_grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, label in zip(axes, ("init", "sap"), strict=True):
        im = ax.imshow(influence_grids[label], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(vmax, 1e-8))
        ax.set(xlabel="Visible context timestep", ylabel="Masked predicted timestep", title=f"{label}: directional influence")
    fig.colorbar(im, ax=axes, label="Output change / context perturbation")
    fig.savefig(args.output / "context_influence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(args.output / "influence.npz", **influence_grids)
    print(f"Figures and metrics: {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="Train on two fixed, fully valid action chunks")
    train_parser.add_argument("--variant", choices=("sap", "shuffled", "full_mask"), required=True)
    train_parser.add_argument("--dataset-root", type=Path, required=True)
    train_parser.add_argument("--dataset-repo-id", default="libero")
    train_parser.add_argument("--base-checkpoint", type=Path, required=True)
    train_parser.add_argument("--chunks", default="0:10,1:10", help="Two distinct episode:frame-offset pairs")
    train_parser.add_argument("--steps", type=int, default=750)
    train_parser.add_argument("--masked-steps", type=int, default=40)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze", help="Evaluate matched controls and plot a compact figure")
    analyze_parser.add_argument("--sap", type=Path, required=True)
    analyze_parser.add_argument("--shuffled", type=Path, required=True)
    analyze_parser.add_argument("--full-mask", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--device", default="cuda")
    analyze_parser.add_argument("--repeats", type=int, default=8)
    analyze_parser.add_argument("--masked-steps", type=int, default=40)
    analyze_parser.add_argument("--eval-seed", type=int, default=117)
    analyze_parser.add_argument("--t", type=float, default=0.5, help="Flow noise level strictly between 0 and 1")
    analyze_parser.add_argument("--influence-dim", type=int, default=0)
    analyze_parser.add_argument("--influence-epsilon", type=float, default=0.03)
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.log_every <= 0:
            parser.error("--log-every must be positive")
        train(args)
    else:
        if not (0.0 < args.t < 1.0) or args.repeats < 1 or args.influence_epsilon <= 0:
            parser.error("Requires 0 < --t < 1, --repeats >= 1, --influence-epsilon > 0")
        analyze(args)


if __name__ == "__main__":
    main()
