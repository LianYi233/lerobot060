#!/usr/bin/env python3
"""Full-ODE action-structure evaluation for the existing two-chunk SAP prompts.

This is an *evaluation-only* companion to ``two_sample_sap_structure.py``.
No action prompts, frozen weights, or training artifacts are changed.  By
default it tests new masks/noise on the original two chunks.  ``--eval-chunks``
instead loads independent episodes with the same low-memory Parquet reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from two_sample_sap_structure import (
    _configure_pi05,
    _get_torch,
    _load_action_chunks_from_selectors,
    _set_seed,
    interpolate_visible,
    masked_mse,
    parse_episode_offsets_many,
)


LABELS = ("init", "sap", "shuffled", "full_mask", "linear_interpolation")
MASK_MODES = ("random", "long_gap")


def select_eval_mask(
    n: int, horizon: int, masked_steps: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Fix mask cardinality across modes, independently of the true event time.

    ``long_gap`` hides one central contiguous block, leaving clean endpoints.
    Crucially, its location does not reveal the ground-truth transition time.
    """
    if n < 1 or not 1 <= masked_steps <= horizon - 2:
        raise ValueError("Masks must leave at least two visible timesteps")
    mask = np.zeros((n, horizon), dtype=bool)
    if mode == "random":
        for b in range(n):
            mask[b, rng.choice(horizon, size=masked_steps, replace=False)] = True
    elif mode == "long_gap":
        start = (horizon - masked_steps) // 2
        mask[:, start : start + masked_steps] = True
    else:
        raise ValueError(f"Unknown mask mode {mode!r}")
    return mask


def permute_visible_context(actions: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Use a per-example permutation; preserve action vectors and visible slots."""
    corrupted = np.array(actions, copy=True)
    for b in range(len(actions)):
        visible = np.flatnonzero(~mask[b])
        corrupted[b, visible] = actions[b, rng.permutation(visible)]
    return corrupted


def second_order_levy_area(path: np.ndarray, *, include_time: bool = True) -> np.ndarray:
    """Antisymmetric level-2 path signature for a piecewise-linear path.

    Unlike a covariance estimate, this is a deterministic descriptor of one
    path and encodes the order of changes across channels.  Adding normalized
    time distinguishes the timing of an otherwise identical path geometry.
    """
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("Expected one path [timesteps >= 2, channels]")
    samples = np.asarray(path, dtype=np.float64)
    if include_time:
        time = np.linspace(0.0, 1.0, len(samples), dtype=np.float64)[:, None]
        samples = np.concatenate([samples, time], axis=-1)
    dimension = samples.shape[1]
    area = np.zeros((dimension, dimension), dtype=np.float64)
    prefix = np.zeros(dimension, dtype=np.float64)
    for delta in np.diff(samples, axis=0):
        area += 0.5 * (np.outer(prefix, delta) - np.outer(delta, prefix))
        prefix += delta
    return area


def action_transitions(path: np.ndarray, channel: int, *, threshold: float = 0.0):
    """Detect discrete positive/negative state changes by a fixed sign cut."""
    states = np.asarray(path[:, channel]) >= threshold
    event_indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    directions = np.where(states[event_indices], 1, -1)
    return [(int(index), int(direction)) for index, direction in zip(event_indices, directions)]


def evaluate_transition_recovery(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    channel: int,
    tolerance: int = 2,
    threshold: float = 0.0,
) -> dict:
    """Count only transitions whose *two flanking states* were hidden.

    A transition visible in the input is not evidence of reconstruction.
    Missing events reduce recall; timing MAE is reported only for recovered
    events (with its own count), avoiding silently excluding failed cases.
    """
    if tolerance < 0 or len(mask) != len(target):
        raise ValueError("Invalid event tolerance or mask shape")
    gt = [
        (index, direction)
        for index, direction in action_transitions(target, channel, threshold=threshold)
        if mask[index - 1] and mask[index]
    ]
    guesses = [
        (index, direction)
        for index, direction in action_transitions(prediction, channel, threshold=threshold)
        if mask[index - 1] and mask[index]
    ]
    used = set()
    errors = []
    for position, direction in gt:
        matches = [
            (abs(other_pos - position), i)
            for i, (other_pos, other_direction) in enumerate(guesses)
            if i not in used and other_direction == direction
        ]
        if matches:
            distance, i = min(matches)
            if distance <= tolerance:
                used.add(i)
                errors.append(int(distance))
    return {
        "events": len(gt),
        "correct_events": len(errors),
        "transition_timing_mae_matched": float(np.mean(errors)) if errors else None,
        "timing_error_sum": int(sum(errors)),
        "false_transitions": len(guesses) - len(used),
    }


def reconstruction_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    event_channel: int,
    tolerance: int,
) -> dict:
    """Metrics computed on one generated, hard-clamped action trajectory."""
    mse = masked_mse(predicted[None], target[None], mask[None])
    adjacent = mask[:-1] | mask[1:]
    # Continuous motion channels are scored separately from the discrete gripper.
    motion_channels = [d for d in range(target.shape[1]) if d != event_channel]
    motion_difference = np.diff(predicted[:, motion_channels], axis=0) - np.diff(
        target[:, motion_channels], axis=0
    )
    derivative_mse = float(np.mean(motion_difference[adjacent] ** 2)) if adjacent.any() else None
    target_area = second_order_levy_area(target)
    predicted_area = second_order_levy_area(predicted)
    difference = float(np.linalg.norm(predicted_area - target_area))
    denominator = float(np.linalg.norm(target_area))
    event = evaluate_transition_recovery(
        predicted, target, mask, channel=event_channel, tolerance=tolerance
    )
    return {
        "masked_mse": mse,
        "motion_derivative_mse": derivative_mse,
        "signature_area_l2": difference,
        "signature_area_relative_error": difference / denominator if denominator >= 1e-6 else None,
        **event,
    }


def _aggregate(rows: list[dict]) -> dict:
    """Aggregate repetitions *within a chunk* before averaging across chunks."""
    if not rows:
        return {}
    per_chunk = defaultdict(list)
    for row in rows:
        per_chunk[row["chunk"]].append(row)
    fields = (
        "masked_mse", "permuted_context_mse", "context_order_gain",
        "motion_derivative_mse", "signature_area_l2", "signature_area_relative_error",
    )
    result = {"chunks": len(per_chunk), "corruptions_per_chunk": len(rows) / len(per_chunk)}
    for field in fields:
        chunk_means = [
            float(np.mean([r[field] for r in group if r.get(field) is not None]))
            for group in per_chunk.values() if any(r.get(field) is not None for r in group)
        ]
        result[field] = float(np.mean(chunk_means)) if chunk_means else None
        result[field + "_std_across_chunks"] = float(np.std(chunk_means, ddof=1)) if len(chunk_means) > 1 else None
    n_events = sum(r["events"] for r in rows)
    recovered = sum(r["correct_events"] for r in rows)
    result.update({
        "eligible_transition_count": n_events,
        "transition_recall_at_tolerance": recovered / n_events if n_events else None,
        "transition_timing_mae_matched": (
            sum(r["timing_error_sum"] for r in rows) / recovered if recovered else None
        ),
        "false_transitions_per_chunk_per_repeat": (
            sum(r["false_transitions"] for r in rows) / len(rows)
        ),
    })
    return result


def _load_runs(args):
    torch = _get_torch()
    directories = {
        "sap": args.sap.resolve(),
        "shuffled": args.shuffled.resolve(),
        "full_mask": args.full_mask.resolve(),
    }
    metadata, prompts, samples = {}, {}, {}
    for label, directory in directories.items():
        metadata[label] = json.loads((directory / "train_metadata.json").read_text(encoding="utf-8"))
        if metadata[label]["variant"] != label:
            raise ValueError(f"{directory} is not a {label} training run")
        prompts[label] = torch.load(directory / "action_prompt.pt", map_location="cpu", weights_only=True)
        samples[label] = torch.load(directory / "two_chunks.pt", map_location="cpu", weights_only=True)
    if len({s["sha256"] for s in samples.values()}) != 1:
        raise ValueError("The three runs were not trained on identical action chunks")
    if len({(m["seed"], m["steps"], m["base_checkpoint"]) for m in metadata.values()}) != 1:
        raise ValueError("The three runs must share seed, steps and base checkpoint")
    for label in ("shuffled", "full_mask"):
        if not torch.equal(prompts["sap"]["prompt_initial"], prompts[label]["prompt_initial"]):
            raise ValueError("Different action prompt initializations detected")
    return metadata, prompts, samples["sap"]


def _integrate_inpainting(core, clean, noise, mask, num_steps: int, context_override=None):
    """Euler-integrate the actual Stage-I masked flow, with known actions clamped.

    At t=1 masked slots contain *only noise* (never a mixture with ground truth).
    The 7 real action dimensions are padded to PI0.5's internal action dimension;
    the other masked dimensions also receive Gaussian noise, as in training.
    """
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import pad_vector

    batch, horizon, dims = clean.shape
    if num_steps < 1 or mask.shape != clean.shape[:2]:
        raise ValueError("Invalid ODE steps or mask shape")
    if noise.shape != (batch, horizon, core.config.max_action_dim):
        raise ValueError("Noise must include all padded internal action dimensions")
    visible = pad_vector(clean if context_override is None else context_override, core.config.max_action_dim)
    mask3d = mask.unsqueeze(-1)
    action_is_pad = torch.zeros(mask.shape, dtype=torch.bool, device=clean.device)
    x_t = torch.where(mask3d, noise, visible)
    dt = -1.0 / num_steps
    with torch.inference_mode():
        for step in range(num_steps):
            time = torch.full((batch,), 1.0 + step * dt, dtype=torch.float32, device=clean.device)
            velocity = core.predict_inpainting_velocity(x_t, time, action_is_pad, mask)
            x_t = torch.where(mask3d, x_t + dt * velocity, visible)
    return x_t[:, :, :dims]


def _save_figures(output: Path, rows: list[dict], example: dict | None, summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), layout="constrained")
    mode = "long_gap"
    names = list(LABELS)
    x = np.arange(len(names))
    colors = ["#8c929b", "#006d77", "#c57b39", "#a35076", "#84a59d"]
    if example is not None:
        gt = example["ground_truth"]
        event_channel = example["event_channel"]
        time = np.arange(len(gt))
        axes[0].plot(time, gt[:, event_channel], color="black", lw=2.1, label="ground truth")
        for i, label in enumerate(names):
            if label in example["predictions"]:
                axes[0].plot(time, example["predictions"][label][:, event_channel], lw=1.1,
                             alpha=0.8, color=colors[i], label=label)
        axes[0].scatter(time[~example["mask"]], gt[~example["mask"], event_channel],
                        color="black", s=17, zorder=4, label="visible")
        axes[0].set(xlabel="Action timestep", ylabel="Normalized gripper action",
                    title="(a) Full-ODE transition reconstruction")
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for ax, metric, title in (
        (axes[1], "signature_area_relative_error", "(b) Ordered path geometry"),
        (axes[2], "transition_recall_at_tolerance", "(c) Hidden transition recall"),
    ):
        values = [summary[mode][name].get(metric) for name in names]
        plot_values = [value if value is not None else 0.0 for value in values]
        ax.bar(x, plot_values, color=colors)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Relative signature error ↓" if ax is axes[1] else "Recall ↑")
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, 0.01, "N/A", ha="center", va="bottom", fontsize=8)
    fig.savefig(output / "structure_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_v2(args):
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    _set_seed(args.eval_seed)
    metadata, prompts, saved = _load_runs(args)
    cfg = _configure_pi05(Path(metadata["sap"]["base_checkpoint"]), args.device,
                          gradient_checkpointing=False)
    source = "train_chunks_only"
    if args.eval_chunks:
        selectors = parse_episode_offsets_many(args.eval_chunks)
        train_episodes = {int(ep) for ep, _ in metadata["sap"]["selectors"]}
        if train_episodes.intersection(ep for ep, _ in selectors):
            raise ValueError("Held-out evaluation episodes must differ from both training episodes")
        data_root = Path(args.dataset_root or metadata["sap"]["dataset_root"])
        chunks, _, _ = _load_action_chunks_from_selectors(data_root, cfg, selectors)
        source = "held_out_episodes"
    else:
        selectors = tuple(tuple(pair) for pair in saved["selectors"])
        chunks = saved["actions"].cpu().numpy()
    n, horizon, dimensions = chunks.shape
    if not 1 <= args.masked_steps <= horizon - 2 or args.ode_steps < 1 or args.repeats < 1:
        raise ValueError("Invalid mask size, ODE step count or repeat count")
    event_dim = args.event_dim if args.event_dim >= 0 else dimensions + args.event_dim
    if not 0 <= event_dim < dimensions or dimensions < 2:
        raise ValueError("--event-dim must select one dimension with at least one other motion dimension")
    if args.batch_size < 1 or args.event_tolerance < 0:
        raise ValueError("batch size must be positive and transition tolerance non-negative")
    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {args.output}")

    print(f"[DATA] evaluation_source={source}, chunks={n}, horizon={horizon}, dim={dimensions}", flush=True)
    print(f"[ODE] Euler steps={args.ode_steps}, masks={MASK_MODES}, repeats={args.repeats}", flush=True)
    policy = PI05Policy.from_pretrained(metadata["sap"]["base_checkpoint"], config=cfg,
                                       local_files_only=True)
    policy.eval()
    all_rows, example = [], None
    # Generate all mask/noise probes once: every model sees *exactly* the same inputs.
    probes = []
    for mode_index, mode in enumerate(MASK_MODES):
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.eval_seed + 100003 * mode_index + 7919 * repeat)
            mask = select_eval_mask(n, horizon, args.masked_steps, mode, rng)
            noise = rng.standard_normal((n, horizon, cfg.max_action_dim), dtype=np.float32)
            shuffled = permute_visible_context(chunks, mask, rng)
            probes.append((mode, repeat, mask, noise, shuffled))

    for label in LABELS:
        if label != "linear_interpolation":
            prompt = prompts["sap"]["prompt_initial"] if label == "init" else prompts[label]["prompt_trained"]
            with torch.no_grad():
                policy.model.prompt_tokens.weight.copy_(prompt.to(args.device))
        for mode, repeat, mask, noise, corrupted_context in probes:
            if label == "linear_interpolation":
                predictions = interpolate_visible(chunks, mask)
                corrupted_predictions = None
            else:
                predictions = np.empty_like(chunks)
                corrupted_predictions = np.empty_like(chunks)
                for start in range(0, n, args.batch_size):
                    end = min(start + args.batch_size, n)
                    clean_t = torch.as_tensor(chunks[start:end], device=args.device, dtype=torch.float32)
                    mask_t = torch.as_tensor(mask[start:end], device=args.device, dtype=torch.bool)
                    noise_t = torch.as_tensor(noise[start:end], device=args.device, dtype=torch.float32)
                    corrupt_t = torch.as_tensor(corrupted_context[start:end], device=args.device, dtype=torch.float32)
                    predictions[start:end] = _integrate_inpainting(
                        policy.model, clean_t, noise_t, mask_t, args.ode_steps
                    ).cpu().numpy()
                    corrupted_predictions[start:end] = _integrate_inpainting(
                        policy.model, clean_t, noise_t, mask_t, args.ode_steps,
                        context_override=corrupt_t
                    ).cpu().numpy()
            for i, selector in enumerate(selectors):
                record = reconstruction_metrics(predictions[i], chunks[i], mask[i],
                                                event_channel=event_dim, tolerance=args.event_tolerance)
                corrupt_mse = (
                    masked_mse(corrupted_predictions[i:i+1], chunks[i:i+1], mask[i:i+1])
                    if corrupted_predictions is not None else None
                )
                record.update({
                    "variant": label, "mask_mode": mode, "repeat": repeat,
                    "chunk": f"{selector[0]}:{selector[1]}",
                    "permuted_context_mse": corrupt_mse,
                    "context_order_gain": corrupt_mse - record["masked_mse"] if corrupt_mse is not None else None,
                })
                all_rows.append(record)
                if mode == "long_gap" and repeat == 0 and i == 0:
                    if example is None:
                        example = {"ground_truth": chunks[i], "mask": mask[i],
                                   "event_channel": event_dim, "predictions": {}}
                    example["predictions"][label] = predictions[i].copy()
        print(f"[DONE] {label}: {len(probes)} mask/noise probes", flush=True)

    summary = {}
    for mode in MASK_MODES:
        summary[mode] = {
            label: _aggregate([r for r in all_rows if r["mask_mode"] == mode and r["variant"] == label])
            for label in LABELS
        }
    summary["protocol"] = {
        "training_examples": 2, "evaluation_source": source,
        "evaluation_chunks": [f"{ep}:{offset}" for ep, offset in selectors],
        "evaluation_sample_sha256": hashlib.sha256(chunks.tobytes()).hexdigest(),
        "seed": args.eval_seed, "repeats": args.repeats, "ode_steps": args.ode_steps,
        "masked_steps": args.masked_steps, "event_dim": event_dim,
        "event_tolerance": args.event_tolerance,
        "flow_protocol": "Euler t=1 to t=0, all masked dimensions pure noise; visible actions clamped each step",
        "uncertainty_unit": "distinct test chunks, not random corruptions; same-episode training probes are descriptive only",
        "signature": "L2 error of antisymmetric level-2 signature for action+normalized time",
        "mask_protocol": "random positions and fixed central long gap; masks do not encode true event time",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics_v2.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output / "per_chunk_v2.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n")
    _save_figures(args.output, all_rows, example, summary)

    for mode in MASK_MODES:
        print(f"\n[{mode}] Masked MSE ↓ | Signature rel. ↓ | Transition recall ↑ | Order gain ↑")
        for label in LABELS:
            row = summary[mode][label]
            fmt = lambda key: "N/A" if row.get(key) is None else f"{row[key]:.4f}"
            print(f"  {label:20s} {fmt('masked_mse'):>8s}  {fmt('signature_area_relative_error'):>8s}  "
                  f"{fmt('transition_recall_at_tolerance'):>8s}  {fmt('context_order_gain'):>8s}")
    print(f"\nSaved: {args.output / 'metrics_v2.json'}")
    print(f"Saved: {args.output / 'structure_v2.png'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sap", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--full-mask", dest="full_mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-chunks", default="",
                        help="Optional held-out episode:offset selectors (comma separated); training episodes forbidden")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--masked-steps", type=int, default=40)
    parser.add_argument("--eval-seed", type=int, default=117)
    parser.add_argument("--event-dim", type=int, default=-1, help="Default: last action dimension (LIBERO gripper)")
    parser.add_argument("--event-tolerance", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    analyze_v2(parser.parse_args(argv))


if __name__ == "__main__":
    main()
