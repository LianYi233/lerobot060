#!/usr/bin/env python3
"""Theory-grounded action-structure probes for the *existing* two-sample SAP runs.

Four evaluation-only analyses can run separately or together:

* temporal_gain: paired ordered/permuted visible-context reconstruction risk;
* hankel: delayed-trajectory dynamical subspace and singular-energy spectrum;
* flow_field: alignment with PI0.5's exact masked Flow Matching target;
* lag_correlation: time-lagged cross-joint velocity covariance error.

All methods use the same pretrained PI0.5, action chunks, masks and noise.
No weights are updated. Hankel/correlation measurements are restricted to the
*hidden continuous-action portion* of a contiguous long-gap mask; they do not
credit the model for copying the known visible actions or infer a physical
system identification from only two training trajectories.
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
from two_sample_sap_structure_v2 import (
    _integrate_inpainting,
    _load_runs,
    permute_visible_context,
    select_eval_mask,
)


MODEL_NAMES = ("init", "sap", "shuffled", "full_mask")
ALL_NAMES = (*MODEL_NAMES, "linear_interpolation")
MASK_MODES = ("random", "long_gap")
ANALYSES = ("temporal_gain", "hankel", "flow_field", "lag_correlation")
COLORS = {
    "init": "#8c929b", "sap": "#006d77", "shuffled": "#c57b39",
    "full_mask": "#a35076", "linear_interpolation": "#84a59d", "ground_truth": "#18212d",
}


def parse_analyses(value: str) -> tuple[str, ...]:
    selected = tuple(x.strip() for x in value.split(",") if x.strip())
    if selected == ("all",):
        return ANALYSES
    if not selected or "all" in selected or len(set(selected)) != len(selected):
        raise ValueError("Use 'all' or a comma-separated list of distinct analysis names")
    unknown = set(selected) - set(ANALYSES)
    if unknown:
        raise ValueError(f"Unknown analysis: {sorted(unknown)}; options: {ANALYSES}")
    return selected


def masked_motion_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                      motion_channels: list[int]) -> float:
    return masked_mse(prediction[..., motion_channels], target[..., motion_channels], mask)


def temporal_gain_scores(prediction: np.ndarray, permuted_prediction: np.ndarray,
                         target: np.ndarray, mask: np.ndarray,
                         motion_channels: list[int]) -> dict:
    """A paired *model-level intervention*, not the population Bayes-risk gap.

    A randomly permuted visible context is not mathematically the same object as
    an unordered set of observations. This intervention therefore tests order
    sensitivity, but its gain can also reflect out-of-distribution corruption.
    """
    correct = masked_mse(prediction, target, mask)
    permuted = masked_mse(permuted_prediction, target, mask)
    motion_correct = masked_motion_mse(prediction, target, mask, motion_channels)
    motion_permuted = masked_motion_mse(permuted_prediction, target, mask, motion_channels)
    return {
        "masked_mse": correct,
        "permuted_mse": permuted,
        "temporal_predictive_gain": permuted - correct,
        "motion_masked_mse": motion_correct,
        "motion_permuted_mse": motion_permuted,
        "motion_temporal_predictive_gain": motion_permuted - motion_correct,
    }


def delay_hankel(sequence: np.ndarray, window: int) -> np.ndarray:
    """Each column is one ordered k-step, d-channel delay vector."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or window < 2 or window > len(values):
        raise ValueError("Hankel requires [timesteps, channels] and 2 <= window <= timesteps")
    return np.column_stack([
        values[start:start + window].reshape(-1)
        for start in range(len(values) - window + 1)
    ])


def hankel_structure_scores(prediction: np.ndarray, target: np.ndarray,
                            *, window: int = 5, rank: int = 3,
                            scale_floor: float = 0.05) -> dict:
    """Compare hidden-continuous-action delay embeddings, never visible context.

    Standardization is determined by the *reference* and shared by every model;
    using a per-prediction scale could artificially hide collapsed amplitudes.
    Projector distance is normalized to [0, 1] for equal-rank projectors.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or truth.ndim != 2 or rank < 1:
        raise ValueError("Hankel paths must have the same [timesteps, channels] shape")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    offset = truth.mean(axis=0, keepdims=True)
    scale = np.maximum(truth.std(axis=0, keepdims=True), scale_floor)
    target_matrix = delay_hankel((truth - offset) / scale, window)
    pred_matrix = delay_hankel((predicted - offset) / scale, window)
    u_true, s_true, _ = np.linalg.svd(target_matrix, full_matrices=False)
    u_pred, s_pred, _ = np.linalg.svd(pred_matrix, full_matrices=False)
    effective = int(np.count_nonzero(s_true > max(1e-10, s_true[0] * 1e-6))) if s_true.size else 0
    use_rank = min(rank, effective, len(s_true), len(s_pred))
    if use_rank == 0:
        return {"hankel_subspace_distance": None, "hankel_spectrum_distance": None,
                "hankel_effective_rank": 0,
                "hankel_reference_singular_values": s_true.tolist(),
                "hankel_prediction_singular_values": s_pred.tolist()}
    p_true = u_true[:, :use_rank] @ u_true[:, :use_rank].T
    p_pred = u_pred[:, :use_rank] @ u_pred[:, :use_rank].T
    distance = float(np.linalg.norm(p_true - p_pred) / np.sqrt(2 * use_rank))
    # SVD energy fractions sum to one, so their total-variation distance is in [0, 1].
    target_energy = s_true ** 2 / max(float(np.sum(s_true ** 2)), 1e-12)
    pred_energy = s_pred ** 2 / max(float(np.sum(s_pred ** 2)), 1e-12)
    return {
        "hankel_subspace_distance": min(distance, 1.0),
        "hankel_spectrum_distance": float(np.abs(target_energy - pred_energy).sum() / 2),
        "hankel_effective_rank": use_rank,
        "hankel_reference_singular_values": s_true.tolist(),
        "hankel_prediction_singular_values": s_pred.tolist(),
    }


def lagged_covariance(sequence: np.ndarray, max_lag: int) -> np.ndarray:
    """Non-symmetrized cross-joint lag covariances of *action increments*."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3 or not 0 <= max_lag < len(values) - 1:
        raise ValueError("Require 3+ timesteps and 0 <= max_lag < timesteps-1")
    velocity = np.diff(values, axis=0)
    velocity = velocity - velocity.mean(axis=0, keepdims=True)
    return np.stack([
        velocity[:len(velocity) - lag].T @ velocity[lag:] / (len(velocity) - lag)
        for lag in range(max_lag + 1)
    ])


def lag_correlation_scores(prediction: np.ndarray, target: np.ndarray,
                           *, max_lag: int = 5, scale_floor: float = 0.05) -> dict:
    """Compare correlation of increments, including within-/cross-joint lags.

    Returns null for an entirely static reference to avoid inferring meaningful
    dynamics from an all-zero denominator.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or truth.ndim != 2 or scale_floor <= 0:
        raise ValueError("Lag paths must match and scale_floor must be positive")
    target_std = np.diff(truth, axis=0).std(axis=0, keepdims=True)
    scale = np.maximum(target_std, scale_floor)
    true_cov = lagged_covariance(truth / scale, max_lag)
    pred_cov = lagged_covariance(predicted / scale, max_lag)
    reference_norm = float(np.linalg.norm(true_cov))
    return {
        "lag_correlation_relative_error": (
            float(np.linalg.norm(pred_cov - true_cov) / reference_norm)
            if reference_norm > 1e-8 else None
        ),
        "lag_reference_norms": np.linalg.norm(true_cov, axis=(1, 2)).tolist(),
        "lag_prediction_norms": np.linalg.norm(pred_cov, axis=(1, 2)).tolist(),
    }


def flow_alignment_scores(predicted_velocity: np.ndarray, target_velocity: np.ndarray) -> dict:
    """Compare actual inpainting velocity to PI0.5's z-a target on masked DOFs."""
    p = np.asarray(predicted_velocity, dtype=np.float64).reshape(-1)
    u = np.asarray(target_velocity, dtype=np.float64).reshape(-1)
    if p.shape != u.shape or not p.size:
        raise ValueError("Flow alignment requires equal-length nonempty vectors")
    norm_p, norm_u = np.linalg.norm(p), np.linalg.norm(u)
    return {
        "flow_cosine": float(p @ u / (norm_p * norm_u)) if norm_p > 1e-10 and norm_u > 1e-10 else None,
        "flow_relative_mse": float(np.sum((p - u) ** 2) / max(np.sum(u ** 2), 1e-12)),
    }


def _aggregate(rows: list[dict], fields: tuple[str, ...]) -> dict:
    """Aggregate corruption/time repeats *within* each distinct test chunk."""
    per_chunk = defaultdict(list)
    for row in rows:
        per_chunk[row["chunk"]].append(row)
    output = {"chunks": len(per_chunk),
              "measurements_per_chunk": len(rows) / len(per_chunk) if per_chunk else 0}
    for field in fields:
        values = [
            float(np.mean([r[field] for r in group if r.get(field) is not None]))
            for group in per_chunk.values() if any(r.get(field) is not None for r in group)
        ]
        output[field] = float(np.mean(values)) if values else None
        output[field + "_std_across_chunks"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else None
        )
    return output


def _evaluate_flow(core, clean, noise, mask, permuted_context, times: tuple[float, ...], dims: int):
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import pad_vector

    padded = pad_vector(clean, core.config.max_action_dim)
    context = pad_vector(permuted_context, core.config.max_action_dim)
    visible = ~mask.unsqueeze(-1)
    action_is_pad = torch.zeros(mask.shape, dtype=torch.bool, device=clean.device)
    target_velocity = noise - padded
    outputs = []
    with torch.inference_mode():
        for value in times:
            t = torch.full((len(clean),), value, dtype=torch.float32, device=clean.device)
            x_t = torch.where(visible, padded, (1 - value) * padded + value * noise)
            x_perm = torch.where(visible, context, (1 - value) * padded + value * noise)
            estimate = core.predict_inpainting_velocity(x_t, t, action_is_pad, mask)
            estimate_perm = core.predict_inpainting_velocity(x_perm, t, action_is_pad, mask)
            per_example = []
            for i in range(len(clean)):
                selected = mask[i]
                ideal = target_velocity[i, selected, :dims].float().cpu().numpy()
                correct = flow_alignment_scores(estimate[i, selected, :dims].float().cpu().numpy(), ideal)
                corrupted = flow_alignment_scores(estimate_perm[i, selected, :dims].float().cpu().numpy(), ideal)
                per_example.append({
                    **correct,
                    "flow_permuted_relative_mse": corrupted["flow_relative_mse"],
                    "flow_order_gain": corrupted["flow_relative_mse"] - correct["flow_relative_mse"],
                    "flow_permuted_cosine": corrupted["flow_cosine"],
                })
            outputs.append(per_example)
    return outputs


def _draw_figures(output: Path, analyses: tuple[str, ...], summary: dict, examples: dict,
                  times: tuple[float, ...]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10})

    if "temporal_gain" in analyses:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
        for ax, mode in zip(axes, MASK_MODES):
            vals = [summary["temporal_gain"][mode][name]["temporal_predictive_gain"] for name in ALL_NAMES]
            ax.bar(range(len(ALL_NAMES)), vals, color=[COLORS[n] for n in ALL_NAMES])
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(range(len(ALL_NAMES)), ALL_NAMES, rotation=26, ha="right")
            ax.set(title=f"{mode}: context-order intervention", ylabel="MSE(permuted) - MSE(ordered) ↑")
        fig.savefig(output / "temporal_gain_v3.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if "hankel" in analyses:
        fig, axes = plt.subplots(1, 3, figsize=(14.3, 3.6), layout="constrained")
        for label, singular in examples.get("hankel_spectra", {}).items():
            energy = np.asarray(singular, dtype=np.float64) ** 2
            if energy.sum() > 1e-12:
                energy /= energy.sum()
                axes[0].plot(np.arange(1, min(len(energy), 12) + 1), energy[:12],
                             marker="o", markersize=2.7, lw=1.2, label=label,
                             color=COLORS[label])
        axes[0].set(title="(a) Hidden motion: Hankel mode energy",
                    xlabel="Mode index", ylabel="Normalized singular energy")
        axes[0].legend(frameon=False, fontsize=7)
        for ax, field, title in (
            (axes[1], "hankel_subspace_distance", "(b) Dynamic subspace distance ↓"),
            (axes[2], "hankel_spectrum_distance", "(c) Hankel energy-spectrum error ↓"),
        ):
            records = summary["hankel"]["long_gap"]
            vals = [records[name][field] for name in ALL_NAMES]
            ax.bar(range(len(ALL_NAMES)), [v if v is not None else 0 for v in vals],
                   color=[COLORS[n] for n in ALL_NAMES])
            ax.set_xticks(range(len(ALL_NAMES)), ALL_NAMES, rotation=26, ha="right")
            ax.set_title(title)
            for i, value in enumerate(vals):
                if value is None:
                    ax.text(i, 0.005, "N/A", ha="center", fontsize=7)
        fig.savefig(output / "hankel_v3.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if "flow_field" in analyses:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
        for ax, mode in zip(axes, MASK_MODES):
            records = summary["flow_field"][mode]
            for name in MODEL_NAMES:
                values = [records[name]["by_t"][str(t)]["flow_cosine"] for t in times]
                ax.plot(times, values, marker="o", markersize=3.5, lw=1.4,
                        label=name, color=COLORS[name])
            ax.set(title=f"{mode}: true flow-field alignment", xlabel="Flow time t",
                   ylabel="Cosine(v_pred, z-a) ↑", xlim=(0, 1))
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(output / "flow_field_v3.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if "lag_correlation" in analyses:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), layout="constrained")
        source = examples.get("lag_norms", {})
        for name, norms in source.items():
            axes[0].plot(np.arange(len(norms)), norms, marker="o", markersize=3, lw=1.3,
                         label=name, color=COLORS[name])
        axes[0].set(xlabel="Velocity lag τ", ylabel="Cross-joint covariance norm",
                    title="(a) Hidden motion: lag dependence")
        axes[0].legend(frameon=False, fontsize=8)
        records = summary["lag_correlation"]["long_gap"]
        vals = [records[name]["lag_correlation_relative_error"] for name in ALL_NAMES]
        axes[1].bar(range(len(ALL_NAMES)), [v if v is not None else 0 for v in vals],
                    color=[COLORS[n] for n in ALL_NAMES])
        axes[1].set_xticks(range(len(ALL_NAMES)), ALL_NAMES, rotation=26, ha="right")
        axes[1].set(ylabel="Relative lag-covariance error ↓",
                    title="(b) Multi-joint temporal correlation")
        for i, value in enumerate(vals):
            if value is None:
                axes[1].text(i, 0.005, "N/A", ha="center", fontsize=7)
        fig.savefig(output / "lag_correlation_v3.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def analyze(args):
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    analyses = parse_analyses(args.analyses)
    times = tuple(float(part.strip()) for part in args.flow_times.split(",") if part.strip())
    if not times or any(not np.isfinite(t) or not 0 < t < 1 for t in times) or len(set(times)) != len(times):
        raise ValueError("--flow-times must contain distinct finite numbers strictly between 0 and 1")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing analysis output: {args.output}")
    metadata, prompts, saved = _load_runs(args)
    _set_seed(args.eval_seed)
    cfg = _configure_pi05(Path(metadata["sap"]["base_checkpoint"]), args.device,
                          gradient_checkpointing=False)
    if args.eval_chunks:
        selectors = parse_episode_offsets_many(args.eval_chunks)
        training_episodes = {int(ep) for ep, _ in metadata["sap"]["selectors"]}
        if training_episodes.intersection(int(ep) for ep, _ in selectors):
            raise ValueError("Held-out evaluation must use episodes distinct from training episodes")
        chunks, _, _ = _load_action_chunks_from_selectors(
            Path(args.dataset_root or metadata["sap"]["dataset_root"]), cfg, selectors
        )
        source = "held_out_episodes"
    else:
        selectors = tuple(tuple(pair) for pair in saved["selectors"])
        chunks = saved["actions"].cpu().numpy()
        source = "train_chunks_only"
    count, horizon, dimensions = chunks.shape
    if not 2 <= args.masked_steps <= horizon - 2 or args.repeats < 1 or args.batch_size < 1:
        raise ValueError("Invalid masked steps, repeats, or batch size")
    if args.ode_steps < 1 or not 2 <= args.hankel_window <= args.masked_steps:
        raise ValueError("Invalid ODE steps or Hankel window")
    if not 1 <= args.hankel_rank or not 1 <= args.max_lag < args.masked_steps - 1:
        raise ValueError("Invalid Hankel rank or lag (requires a hidden contiguous interval)")
    event_dim = args.event_dim if args.event_dim >= 0 else dimensions + args.event_dim
    if not 0 <= event_dim < dimensions or dimensions < 2:
        raise ValueError("--event-dim must select one channel leaving at least one continuous dimension")
    motion_channels = [i for i in range(dimensions) if i != event_dim]
    need_ode = bool(set(analyses) & {"temporal_gain", "hankel", "lag_correlation"})
    policy = PI05Policy.from_pretrained(metadata["sap"]["base_checkpoint"], config=cfg,
                                       local_files_only=True)
    policy.eval()
    print(f"[V3] analyses={analyses}, source={source}, chunks={count}, dim={dimensions}", flush=True)

    probes = []
    for mode_index, mode in enumerate(MASK_MODES):
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.eval_seed + 100003 * mode_index + 7919 * repeat)
            mask = select_eval_mask(count, horizon, args.masked_steps, mode, rng)
            noise = rng.standard_normal((count, horizon, cfg.max_action_dim), dtype=np.float32)
            permuted = permute_visible_context(chunks, mask, rng)
            probes.append((mode, repeat, mask, noise, permuted))

    records = {name: [] for name in ANALYSES}
    examples: dict = {"hankel_spectra": {}, "lag_norms": {}}
    for name in ALL_NAMES:
        if name != "linear_interpolation":
            prompt = (prompts["sap"]["prompt_initial"] if name == "init"
                      else prompts[name]["prompt_trained"])
            with torch.no_grad():
                policy.model.prompt_tokens.weight.copy_(prompt.to(args.device))
        for mode, repeat, mask, noise, permuted in probes:
            if name == "linear_interpolation":
                if not need_ode:
                    continue
                predictions = interpolate_visible(chunks, mask)
                corrupt_predictions = interpolate_visible(permuted, mask)
            else:
                predictions = np.empty_like(chunks) if need_ode else None
                corrupt_predictions = np.empty_like(chunks) if "temporal_gain" in analyses else None
            for begin in range(0, count, args.batch_size):
                end = min(begin + args.batch_size, count)
                if name == "linear_interpolation":
                    break
                clean_t = torch.as_tensor(chunks[begin:end], device=args.device, dtype=torch.float32)
                mask_t = torch.as_tensor(mask[begin:end], device=args.device, dtype=torch.bool)
                noise_t = torch.as_tensor(noise[begin:end], device=args.device, dtype=torch.float32)
                permuted_t = torch.as_tensor(permuted[begin:end], device=args.device, dtype=torch.float32)
                if need_ode:
                    predictions[begin:end] = _integrate_inpainting(
                        policy.model, clean_t, noise_t, mask_t, args.ode_steps
                    ).float().cpu().numpy()
                    if corrupt_predictions is not None:
                        corrupt_predictions[begin:end] = _integrate_inpainting(
                            policy.model, clean_t, noise_t, mask_t, args.ode_steps,
                            context_override=permuted_t
                        ).float().cpu().numpy()
                if "flow_field" in analyses:
                    flow_results = _evaluate_flow(
                        policy.model, clean_t, noise_t, mask_t, permuted_t, times, dimensions
                    )
                    for time, batch_result in zip(times, flow_results):
                        for i, scores in enumerate(batch_result):
                            records["flow_field"].append({
                                "model": name, "mask_mode": mode, "repeat": repeat,
                                "chunk": f"{selectors[begin+i][0]}:{selectors[begin+i][1]}",
                                "t": time, **scores,
                            })
            if not need_ode:
                continue
            for i, selector in enumerate(selectors):
                identity = {"model": name, "mask_mode": mode, "repeat": repeat,
                            "chunk": f"{selector[0]}:{selector[1]}"}
                if "temporal_gain" in analyses:
                    scores = temporal_gain_scores(
                        predictions[i:i+1], corrupt_predictions[i:i+1],
                        chunks[i:i+1], mask[i:i+1], motion_channels
                    )
                    records["temporal_gain"].append({**identity, **scores})
                if mode != "long_gap":
                    continue
                hidden_true = chunks[i, mask[i]][:, motion_channels]
                hidden_pred = predictions[i, mask[i]][:, motion_channels]
                if "hankel" in analyses:
                    scores = hankel_structure_scores(hidden_pred, hidden_true,
                                                      window=args.hankel_window, rank=args.hankel_rank)
                    records["hankel"].append({**identity, **scores})
                    if repeat == 0 and i == 0:
                        examples["hankel_spectra"][name] = scores["hankel_prediction_singular_values"]
                        examples["hankel_spectra"]["ground_truth"] = scores["hankel_reference_singular_values"]
                if "lag_correlation" in analyses:
                    scores = lag_correlation_scores(hidden_pred, hidden_true, max_lag=args.max_lag)
                    records["lag_correlation"].append({**identity, **scores})
                    if repeat == 0 and i == 0:
                        examples["lag_norms"][name] = scores["lag_prediction_norms"]
                        examples["lag_norms"]["ground_truth"] = scores["lag_reference_norms"]
        print(f"[DONE] {name}", flush=True)

    fields = {
        "temporal_gain": ("masked_mse", "permuted_mse", "temporal_predictive_gain",
                          "motion_masked_mse", "motion_permuted_mse", "motion_temporal_predictive_gain"),
        "hankel": ("hankel_subspace_distance", "hankel_spectrum_distance"),
        "flow_field": ("flow_cosine", "flow_relative_mse", "flow_permuted_relative_mse",
                       "flow_order_gain", "flow_permuted_cosine"),
        "lag_correlation": ("lag_correlation_relative_error",),
    }
    summary: dict = {}
    for analysis in analyses:
        summary[analysis] = {}
        for mode in MASK_MODES if analysis in ("temporal_gain", "flow_field") else ("long_gap",):
            summary[analysis][mode] = {}
            for name in MODEL_NAMES if analysis == "flow_field" else ALL_NAMES:
                selected = [row for row in records[analysis]
                            if row["model"] == name and row["mask_mode"] == mode]
                value = _aggregate(selected, fields[analysis])
                if analysis == "flow_field":
                    value["by_t"] = {
                        str(t): _aggregate([row for row in selected if row["t"] == t], fields[analysis])
                        for t in times
                    }
                summary[analysis][mode][name] = value
            if analysis == "temporal_gain":
                control = summary[analysis][mode]
                for name in ALL_NAMES:
                    control[name]["order_gain_vs_init"] = (
                        control[name]["temporal_predictive_gain"] -
                        control["init"]["temporal_predictive_gain"]
                    )
                    control[name]["order_gain_vs_shuffled"] = (
                        control[name]["temporal_predictive_gain"] -
                        control["shuffled"]["temporal_predictive_gain"]
                    )

    prompt_change = {}
    for name in MODEL_NAMES:
        base = prompts["sap"]["prompt_initial"].float()
        trained = base if name == "init" else prompts[name]["prompt_trained"].float()
        prompt_change[name] = {
            "absolute_l2": float(torch.linalg.vector_norm(trained - base).item()),
            "relative_l2": float((torch.linalg.vector_norm(trained - base) /
                                  torch.linalg.vector_norm(base).clamp_min(1e-12)).item()),
        }
    summary["prompt_change"] = prompt_change
    summary["protocol"] = {
        "analyses": list(analyses), "training_examples": 2,
        "evaluation_source": source,
        "evaluation_chunks": [f"{ep}:{offset}" for ep, offset in selectors],
        "evaluation_sample_sha256": hashlib.sha256(chunks.tobytes()).hexdigest(),
        "eval_seed": args.eval_seed, "repeats": args.repeats,
        "ode_steps": args.ode_steps if need_ode else None,
        "masked_steps": args.masked_steps, "flow_times": list(times),
        "hankel_window": args.hankel_window, "hankel_rank": args.hankel_rank,
        "max_lag": args.max_lag, "excluded_discrete_channel": event_dim,
        "paired_protocol": "identical masks, Gaussian noise, model budget, and train samples",
        "temporal_gain_warning": (
            "The permuted-context experiment is an intervention, not a direct estimate "
            "of the Bayes risk difference for ordered versus unordered information."
        ),
        "hankel_and_lag_scope": "contiguous hidden region only; continuous dimensions only",
        "flow_field_scope": (
            "Exact pi05 u_t=z-a on masked original dimensions; true a is used only "
            "to construct the evaluation flow path and target, not to condition full-ODE sampling."
        ),
        "independence_warning": "Corruption repeats are NOT independent test trajectories",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics_v3.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output / "per_chunk_v3.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    if not args.skip_plots:
        _draw_figures(args.output, analyses, summary, examples, times)
    for analysis in analyses:
        print(f"\n[{analysis}]")
        field = {"temporal_gain": "temporal_predictive_gain", "hankel": "hankel_subspace_distance",
                 "flow_field": "flow_cosine", "lag_correlation": "lag_correlation_relative_error"}[analysis]
        for mode, results in summary[analysis].items():
            print(f"  {mode}: " + "  ".join(
                f"{model}={results[model][field]:.5f}" if results[model][field] is not None
                else f"{model}=N/A"
                for model in results
            ))
    print(f"\nSaved: {args.output / 'metrics_v3.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sap", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--full-mask", dest="full_mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyses", default="all",
                        help="all, or comma-separated: temporal_gain,hankel,flow_field,lag_correlation")
    parser.add_argument("--eval-chunks", default="", help="held-out episode:offset pairs; cannot reuse training episodes")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--masked-steps", type=int, default=20)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=117)
    parser.add_argument("--event-dim", type=int, default=-1)
    parser.add_argument("--hankel-window", type=int, default=5)
    parser.add_argument("--hankel-rank", type=int, default=3)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--flow-times", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--device", default="cuda")
    analyze(parser.parse_args(argv))


if __name__ == "__main__":
    main()
