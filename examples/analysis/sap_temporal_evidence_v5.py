#!/usr/bin/env python3
"""V5: falsifiable, theory-motivated evidence for SAP's temporal action prior.

Run the subcommands in this order:
  data-check       CPU-only ordered-vs-bag prediction on real LIBERO actions.
  blocked-eval     True future/contiguous-gap masked flow and full-ODE sampling.
  synthetic-prepare  Known AR(2) training/test chunks for original V4 prompt trainer.
  synthetic-score  Recover known modal coefficients from generated trajectories.
  probe            Frozen Action Expert hidden-state future-action linear readout.
  signature-mmd    Paired, context-kernel-weighted time-augmented path discrepancy.

No existing pretrained weights, prompt checkpoints, or V4 metrics are changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from sap_temporal_v5_utils import (
    MASK_MODES,
    _ridge_predict,
    ar2_parameter_error,
    conditional_signature_mmd,
    paired_interval,
    parse_names,
    ridge_forecast,
    swap_visible_order,
    synthetic_ar2_chunks,
    temporal_mask,
)
from two_sample_sap_structure import _configure_pi05, _get_torch, _save_json, _set_seed
from two_sample_sap_structure_v2 import _integrate_inpainting
from two_sample_sap_structure_v3 import flow_alignment_scores
from two_sample_sap_structure_v4 import _load_prepared, _load_prompts, parse_flow_times, parse_positive_ints


MODEL_NAMES = ("init", "sap", "shuffled")


def _nonexistent_output(folder: Path):
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {folder}")
    return folder.resolve()


def _save_output(folder: Path, filename: str, content: dict):
    folder.mkdir(parents=True, exist_ok=False)
    _save_json(folder / filename, content)


def data_check(args):
    """Preflight: detect whether ordered REAL actions predict a hidden future."""
    output = _nonexistent_output(args.output)
    payload = _load_prepared(args.prepared)
    train = payload["train"].detach().cpu().float().numpy()
    test = payload["test"].detach().cpu().float().numpy()
    horizons = parse_positive_ints(args.horizons)
    if args.context + max(horizons) > train.shape[1]:
        raise ValueError("Prediction horizon exceeds prepared 50-step action chunks")
    manifest = payload["manifest"]
    result = {"protocol": "primingvla-temporal-v5-ordered-bag-preflight",
              "prepared_train_sha256": manifest["train_sha256"],
              "prepared_test_sha256": manifest["test_sha256"],
              "train_episodes": len(train), "test_episodes": len(test),
              "context": args.context, "seed": args.seed,
              "claim_limit": "Linear-probe comparison, NOT the Bayes-optimal conditional-variance identity",
              "horizons": {}}
    for horizon in horizons:
        scores = ridge_forecast(train, test, context=args.context,
                                horizon=horizon, seed=args.seed)
        ordered = np.asarray(scores["ordered"].pop("per_episode_mse"))
        unordered = np.asarray(scores["unordered"].pop("per_episode_mse"))
        result["horizons"][str(horizon)] = {
            **scores,
            "bag_minus_ordered": paired_interval(ordered, unordered, seed=args.seed + horizon),
        }
        print(f"[P0] future={horizon:2d}: ordered={scores['ordered']['relative_mse']:.5f}, "
              f"bag={scores['unordered']['relative_mse']:.5f}, "
              f"bag−ordered={float(np.mean(unordered - ordered)):+.6f}", flush=True)
    _save_output(output, "data_order_preflight_v5.json", result)
    print(f"[P0] saved {output}", flush=True)


def synthetic_prepare(args):
    """Create a synthetic V4-compatible payload; reuse exact V4 SAP trainer."""
    torch = _get_torch()
    folder = _nonexistent_output(args.output)
    if args.train_chunks < 2 or args.test_chunks < 2:
        raise ValueError("Need >=2 independent synthetic train/test chunks")
    train, train_meta = synthetic_ar2_chunks(args.train_chunks, seed=args.seed, first_episode=0)
    test, test_meta = synthetic_ar2_chunks(args.test_chunks, seed=args.seed + 500_000,
                                           first_episode=100_000)
    manifest = {
        "protocol": "primingvla-sap-structure-v4-episode-disjoint",
        "source": "synthetic-stable-ar2-with-known-coefficients",
        "horizon": 50, "action_dim": int(train.shape[-1]),
        "sizes": [args.train_chunks], "heldout_chunks": args.test_chunks,
        "seed": args.seed, "train": train_meta, "test": test_meta,
        "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "synthetic_truth": "a[k+1]=2*r*cos(omega)*a[k]-r**2*a[k-1]+innovation",
        "warning": "Synthetic action dynamics, not identified physical robot/environment dynamics",
    }
    folder.mkdir(parents=True)
    torch.save({"train": torch.from_numpy(train), "test": torch.from_numpy(test),
                "manifest": manifest}, folder / "prepared_v4.pt")
    _save_json(folder / "manifest_v4.json", manifest)
    print(f"[P2] AR(2) synthetic chunks: train={len(train)}, test={len(test)}; {folder}", flush=True)


def _instantiate_policy(args, *, seed: int):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    _set_seed(seed)
    cfg = _configure_pi05(args.base_checkpoint, args.device, gradient_checkpointing=False)
    cfg.next_action_masked_steps = args.train_masked_steps
    policy = PI05Policy.from_pretrained(args.base_checkpoint, config=cfg, local_files_only=True)
    policy.eval()
    return policy, cfg


def _validate_v4_inputs(args):
    payload = _load_prepared(args.prepared)
    prompts = _load_prompts(args, payload["manifest"])
    return payload, prompts


def _episode_score_map(rows: list[dict], *, model: str, mode: str,
                       metric: str, condition: str | None = None,
                       flow_time: float | None = None):
    grouping: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] != model or row["mode"] != mode:
            continue
        if condition is not None and row.get("condition") != condition:
            continue
        if flow_time is not None and abs(row.get("flow_time", -1234.) - flow_time) > 1e-7:
            continue
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouping[int(row["episode_index"])].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouping.items()}


def _paired_result(best: dict[int, float], control: dict[int, float], seed: int):
    keys = sorted(set(best).intersection(control))
    if len(keys) < 2 or set(best) != set(control):
        raise ValueError("Paired scores require matching sets of >=2 held-out episodes")
    return paired_interval(np.array([best[k] for k in keys]),
                           np.array([control[k] for k in keys]), seed=seed)


def _summarize_blocked(flow_rows, ode_rows, *, modes, high_noise, seed):
    summary = {}
    for mode in modes:
        source = {name: _episode_score_map(flow_rows, model=name, mode=mode,
                                            metric="flow_mse", condition="ordered",
                                            flow_time=high_noise)
                  for name in MODEL_NAMES}
        corrupted = {name: _episode_score_map(flow_rows, model=name, mode=mode,
                                               metric="flow_mse", condition="permuted",
                                               flow_time=high_noise)
                     for name in MODEL_NAMES}
        model_scores = {name: {"high_noise_flow_mse": float(np.mean(list(source[name].values()))),
                               "permuted_flow_mse": float(np.mean(list(corrupted[name].values()))),
                               "order_gain": float(np.mean([corrupted[name][ep] - source[name][ep]
                                                            for ep in sorted(source[name])]))}
                        for name in MODEL_NAMES}
        ordered_sap = source["sap"]
        ordered_init = source["init"]
        order_gain_sap = {e: corrupted["sap"][e] - source["sap"][e] for e in source["sap"]}
        order_gain_init = {e: corrupted["init"][e] - source["init"][e] for e in source["init"]}
        flow_comparison = {
            "init_minus_sap": _paired_result(ordered_sap, ordered_init, seed + 1),
            "shuffled_minus_sap": _paired_result(ordered_sap, source["shuffled"], seed + 2),
            # order gain is a DIFFERENCE-IN-DIFFERENCES, not a Bayes-information estimate.
            "extra_order_gain_sap_vs_init": _paired_result(order_gain_init, order_gain_sap, seed + 3),
        }
        ode_means = {name: _episode_score_map(ode_rows, model=name, mode=mode,
                                               metric="masked_action_mse", condition="ordered")
                     for name in MODEL_NAMES}
        ode_comparison = None
        if all(ode_means[name] for name in MODEL_NAMES):
            ode_comparison = {
                "init_minus_sap": _paired_result(ode_means["sap"], ode_means["init"], seed + 4),
                "shuffled_minus_sap": _paired_result(ode_means["sap"], ode_means["shuffled"], seed + 5),
            }
            for name in MODEL_NAMES:
                model_scores[name]["full_ode_masked_action_mse"] = float(np.mean(list(ode_means[name].values())))
        summary[mode] = {"models": model_scores, "flow_comparison": flow_comparison,
                         "ode_comparison": ode_comparison,
                         "high_noise_t": high_noise, "episodes": len(ordered_sap)}
    return summary


def blocked_eval(args):
    """Evaluate held-out contiguous/future temporal structure with true PI0.5 flow."""
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import pad_vector
    from two_sample_sap_structure_v2 import second_order_levy_area

    folder = _nonexistent_output(args.output)
    modes = parse_names(args.modes, MASK_MODES)
    times = parse_flow_times(args.flow_times)
    if args.repeats < 1 or args.batch_size < 1 or args.ode_steps < 1 or args.test_limit < 2:
        raise ValueError("Invalid repeats/batch/ODE steps/test limit")
    payload, prompts = _validate_v4_inputs(args)
    manifest = payload["manifest"]
    test = payload["test"].detach().float().cpu().numpy()
    test = np.ascontiguousarray(test[:min(len(test), args.test_limit)])
    if len(test) < 2:
        raise ValueError("At least 2 episode-disjoint test examples are needed")
    test_metadata = manifest["test"][:len(test)]
    policy, cfg = _instantiate_policy(args, seed=args.seed)
    prompt_shape = tuple(policy.model.prompt_tokens.weight.shape)
    if any(tuple(weight.shape) != prompt_shape for weight in prompts.values()):
        raise ValueError("Prompt shape incompatible with this PI0.5 checkpoint")
    device = args.device
    flow_rows: list[dict] = []
    ode_rows: list[dict] = []
    rollouts: dict[str, list] = defaultdict(list)
    dims = test.shape[-1]
    print(f"[P1] episodes={len(test)}, modes={modes}, noise t={times}, "
          f"ODE={args.ode_steps} steps, variants={MODEL_NAMES}", flush=True)
    with torch.inference_mode():
        for mode_index, mode in enumerate(modes):
            for repeat in range(args.repeats):
                rng = np.random.default_rng(args.seed + 10301 * mode_index + 7919 * repeat)
                for first in range(0, len(test), args.batch_size):
                    last = min(first + args.batch_size, len(test))
                    batch_np = test[first:last]
                    batch_count = len(batch_np)
                    hidden_np, scored_np = temporal_mask(batch_count, 50, mode, rng)
                    corrupted_np = swap_visible_order(batch_np, hidden_np, rng)
                    clean = torch.as_tensor(batch_np, device=device, dtype=torch.float32)
                    corrupt = torch.as_tensor(corrupted_np, device=device, dtype=torch.float32)
                    hidden = torch.as_tensor(hidden_np, device=device, dtype=torch.bool)
                    noise_np = rng.standard_normal((batch_count, 50, cfg.max_action_dim), dtype=np.float32)
                    noise = torch.as_tensor(noise_np, device=device)
                    padded = pad_vector(clean, cfg.max_action_dim)
                    masked3d = hidden.unsqueeze(-1)
                    target_velocity = (noise - padded)[:, :, :dims]
                    padding = torch.zeros_like(hidden, dtype=torch.bool)
                    for label, prompt in prompts.items():
                        policy.model.prompt_tokens.weight.copy_(prompt.to(device=device))
                        for flow_time in times:
                            t = torch.full((batch_count,), flow_time, dtype=torch.float32, device=device)
                            # Hidden inputs contain clean targets in this controlled local
                            # velocity diagnostic; FULL ODE never receives hidden truth.
                            hidden_xt = (1. - flow_time) * padded + flow_time * noise
                            for condition, context in (("ordered", clean), ("permuted", corrupt)):
                                visible = pad_vector(context, cfg.max_action_dim)
                                xt = torch.where(masked3d, hidden_xt, visible)
                                prediction = policy.model.predict_inpainting_velocity(
                                    xt, t, padding, hidden
                                )[:, :, :dims]
                                prediction_np = prediction.float().cpu().numpy()
                                goal_np = target_velocity.float().cpu().numpy()
                                for index in range(batch_count):
                                    subset = scored_np[index]
                                    raw_pred = prediction_np[index, subset]
                                    raw_true = goal_np[index, subset]
                                    flow_rows.append({
                                        "episode_index": int(test_metadata[first + index]["episode_index"]),
                                        "stratum": test_metadata[first + index]["stratum"],
                                        "model": label, "mode": mode, "condition": condition,
                                        "repeat": repeat, "flow_time": flow_time,
                                        "flow_mse": float(np.mean((raw_pred - raw_true) ** 2)),
                                        "flow_relative_mse": flow_alignment_scores(raw_pred, raw_true)[
                                            "flow_relative_mse"
                                        ],
                                    })
                        if args.flow_only:
                            continue
                        for condition, context in (("ordered", clean), ("permuted", corrupt)):
                            if condition == "permuted" and not args.permuted_ode:
                                continue
                            reconstructed = _integrate_inpainting(
                                policy.model, clean, noise, hidden, args.ode_steps,
                                context_override=context if condition == "permuted" else None,
                            ).float().cpu().numpy()
                            for index in range(batch_count):
                                selected = scored_np[index]
                                error = reconstructed[index, selected] - batch_np[index, selected]
                                true_area = second_order_levy_area(batch_np[index, :, :-1])
                                fake_area = second_order_levy_area(reconstructed[index, :, :-1])
                                episode_id = int(test_metadata[first + index]["episode_index"])
                                ode_rows.append({
                                    "episode_index": episode_id, "model": label, "mode": mode,
                                    "condition": condition, "repeat": repeat,
                                    "masked_action_mse": float(np.mean(error * error)),
                                    "masked_motion_mse": float(np.mean(error[:, :-1] ** 2)),
                                    "signature_area_l2": float(np.linalg.norm(fake_area - true_area)),
                                })
                                if args.save_rollouts and condition == "ordered":
                                    for key, value in (
                                        ("episode_ids", episode_id), ("models", label),
                                        ("modes", mode), ("repeats", repeat),
                                        ("generated", reconstructed[index]),
                                        ("ground_truth", batch_np[index]),
                                        ("hidden", hidden_np[index]),
                                    ):
                                        rollouts[key].append(value)
                print(f"[P1] completed mode={mode}, repeat={repeat+1}/{args.repeats}", flush=True)

    metrics = {
        "protocol": "primingvla-sap-temporal-evidence-v5",
        "source": manifest.get("source", "real-LIBERO"),
        "prepared_test_sha256": manifest["test_sha256"],
        "train_episodes": int(args.num_chunks), "test_episodes": len(test),
        "seed": args.seed, "modes": list(modes), "flow_times": list(times),
        "repeats": args.repeats, "ode_steps": None if args.flow_only else args.ode_steps,
        "permuted_ode": args.permuted_ode,
        "no_target_leakage": "Flow diagnostic uses noisy target by definition; ODE generation starts from pure noise",
        "interpretation": "Positive order gain without improved correct-order risk does NOT demonstrate temporal learning",
        "results": _summarize_blocked(flow_rows, ode_rows,
                                      modes=modes, high_noise=max(times), seed=args.seed),
    }
    folder.mkdir(parents=True)
    _save_json(folder / "per_episode_flow_temporal_v5.json", flow_rows)
    _save_json(folder / "per_episode_ode_temporal_v5.json", ode_rows)
    if args.save_rollouts and not args.flow_only:
        np.savez_compressed(folder / "rollouts_temporal_v5.npz",
                            **{name: np.asarray(values) for name, values in rollouts.items()})
    # The summary JSON is the LAST write and serves as a resumable-completion marker.
    _save_json(folder / "metrics_temporal_v5.json", metrics)
    for mode in modes:
        record = metrics["results"][mode]
        m = record["models"]
        delta = record["flow_comparison"]["shuffled_minus_sap"]["mean"]
        print(f"[P1] {mode}: ordered high-noise MSE "
              f"Init={m['init']['high_noise_flow_mse']:.5f} "
              f"SAP={m['sap']['high_noise_flow_mse']:.5f} "
              f"Shuffled={m['shuffled']['high_noise_flow_mse']:.5f}, "
              f"Shuffled-SAP={delta:+.5f}", flush=True)
    print(f"[P1] saved {folder}", flush=True)


def synthetic_score(args):
    """Compare known AR(2) coefficients against those fitted to generated paths."""
    folder = _nonexistent_output(args.output)
    prepared = _load_prepared(args.prepared)
    meta_by_id = {int(record["episode_index"]): record
                  for record in prepared["manifest"]["test"]}
    data = np.load(args.rollouts, allow_pickle=False)
    models, modes, episode_ids = data["models"], data["modes"], data["episode_ids"]
    records = []
    for index in range(len(models)):
        if str(modes[index]) != args.mode:
            continue
        episode_id = int(episode_ids[index])
        if episode_id not in meta_by_id or "true_ar2" not in meta_by_id[episode_id]:
            raise ValueError("Synthetic evaluation does not match known AR(2) test episodes")
        truth = meta_by_id[episode_id]["true_ar2"]
        for source, sequence in (("generated", data["generated"][index]),
                                 ("oracle_real", data["ground_truth"][index])):
            scores = ar2_parameter_error(sequence, beta1=truth["beta1"],
                                         beta2=truth["beta2"], start=20, stop=40)
            records.append({"episode_index": episode_id, "model": str(models[index]),
                            "source": source, "mode": args.mode,
                            "true_r": truth["r"], "true_omega": truth["omega"],
                            **scores})
    if not records:
        raise ValueError(f"No {args.mode} rollouts found in {args.rollouts}")
    results = {}
    for label in MODEL_NAMES:
        results[label] = {}
        for source in ("generated", "oracle_real"):
            matching = [row for row in records if row["model"] == label and row["source"] == source]
            values = [row["coefficient_l2"] for row in matching
                      if row["coefficient_l2"] is not None]
            radii = [abs(row["r_hat"] - row["true_r"]) for row in matching
                     if row["r_hat"] is not None]
            frequencies = [abs(row["omega_hat"] - row["true_omega"]) for row in matching
                           if row["omega_hat"] is not None]
            results[label][source] = {
                "mean_coefficient_l2": float(np.mean(values)) if values else None,
                "mean_abs_pole_radius_error": float(np.mean(radii)) if radii else None,
                "mean_abs_pole_frequency_error_rad": float(np.mean(frequencies)) if frequencies else None,
                "valid_episodes": len(values), "valid_complex_poles": len(frequencies),
            }
    identified = {(row["model"], row["episode_index"]): row["coefficient_l2"]
                  for row in records if row["source"] == "generated"
                  and row["coefficient_l2"] is not None}
    common_ids = sorted({e for name, e in identified if name == "init"}
                        & {e for name, e in identified if name == "sap"}
                        & {e for name, e in identified if name == "shuffled"})
    comparisons = {}
    if len(common_ids) >= 2:
        sap = np.asarray([identified[("sap", e)] for e in common_ids])
        for label in ("init", "shuffled"):
            control = np.asarray([identified[(label, e)] for e in common_ids])
            comparisons[f"{label}_minus_sap"] = paired_interval(sap, control, seed=123 + len(common_ids))
    _save_output(folder, "ar2_recovery_v5.json", {
        "protocol": "primingvla-v5-synthetic-AR2-identification",
        "mode": args.mode, "results": results, "paired_comparisons": comparisons,
        "per_episode": records,
        "interpretation": "Known temporal action coefficients; NOT environment-state identification",
    })
    for label in MODEL_NAMES:
        print(f"[P2] {label}: synthetic AR2 coefficient error={results[label]['generated']['mean_coefficient_l2']}",
              flush=True)


def probe(args):
    """Read out future actions from frozen PI0.5 pre-output hidden tokens."""
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import pad_vector

    folder = _nonexistent_output(args.output)
    payload, prompts = _validate_v4_inputs(args)
    test = payload["test"].detach().float().cpu().numpy()[:args.test_limit]
    if len(test) < 30:
        raise ValueError("Probe requires at least 30 adaptation-heldout episodes")
    horizons = parse_positive_ints(args.horizons)
    if any(h > 50 - args.context for h in horizons):
        raise ValueError("Future horizons must fit within 50-step chunks")
    policy, cfg = _instantiate_policy(args, seed=args.seed)
    if any(tuple(weight.shape) != tuple(policy.model.prompt_tokens.weight.shape)
           for weight in prompts.values()):
        raise ValueError("Prompt shape mismatch")
    hidden_np = np.arange(50)[None, :] >= args.context
    mask = torch.as_tensor(np.broadcast_to(hidden_np, (args.batch_size, 50)).copy(),
                           device=args.device, dtype=torch.bool)
    features: dict[str, list[np.ndarray]] = {label: [] for label in MODEL_NAMES}
    captured: list[np.ndarray] = []

    def capture_hidden(_module, inputs):
        # The input to action_out_proj is the *actual expert hidden state*,
        # before its final linear action readout.
        captured.append(inputs[0].detach().float().cpu().numpy())

    hook = policy.model.action_out_proj.register_forward_pre_hook(capture_hidden)
    try:
        rng = np.random.default_rng(args.seed)
        with torch.inference_mode():
            for first in range(0, len(test), args.batch_size):
                batch_np = test[first:first + args.batch_size]
                count = len(batch_np)
                clean = torch.as_tensor(batch_np, device=args.device, dtype=torch.float32)
                noise = torch.as_tensor(
                    rng.standard_normal((count, 50, cfg.max_action_dim), dtype=np.float32),
                    device=args.device,
                )
                mask_batch = mask[:count]
                current = torch.where(mask_batch.unsqueeze(-1), noise,
                                      pad_vector(clean, cfg.max_action_dim))
                step = torch.ones(count, device=args.device, dtype=torch.float32)
                padding = torch.zeros((count, 50), device=args.device, dtype=torch.bool)
                for label, prompt in prompts.items():
                    policy.model.prompt_tokens.weight.copy_(prompt.to(device=args.device))
                    captured.clear()
                    policy.model.predict_inpainting_velocity(current, step, padding, mask_batch)
                    if len(captured) != 1 or captured[0].shape[:2] != (count, 50):
                        raise RuntimeError("Expected one Action Expert pre-readout hidden state")
                    features[label].append(captured[0][:, args.context - 1].copy())
    finally:
        hook.remove()

    vectors = {label: np.concatenate(rows, axis=0) for label, rows in features.items()}
    rng = np.random.default_rng(args.seed + 123)
    indices = rng.permutation(len(test))
    fit_end = max(12, int(.6 * len(test)))
    valid_end = max(fit_end + 6, int(.8 * len(test)))
    fit_idx, validation_idx, test_idx = indices[:fit_end], indices[fit_end:valid_end], indices[valid_end:]
    if len(test_idx) < 2:
        raise ValueError("Not enough heldout episodes for frozen feature probing")
    results = {label: {} for label in MODEL_NAMES}
    for horizon in horizons:
        targets = test[:, args.context + horizon - 1, :-1].astype(np.float64)
        baseline = targets[fit_idx].mean(axis=0, keepdims=True)
        denominator = max(float(np.mean((targets[test_idx] - baseline) ** 2)), 1e-12)
        for label in MODEL_NAMES:
            vectors_train = vectors[label]
            best = None
            for alpha in (.01, .1, 1., 10., 100., 1000.):
                prediction = _ridge_predict(vectors_train[fit_idx], targets[fit_idx],
                                            vectors_train[validation_idx], alpha)
                mse = float(np.mean((prediction - targets[validation_idx]) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, alpha)
            prediction = _ridge_predict(vectors_train[fit_idx], targets[fit_idx],
                                        vectors_train[test_idx], best[1])
            result = float(np.mean((prediction - targets[test_idx]) ** 2) / denominator)
            results[label][str(horizon)] = {"relative_mse": result,
                                            "R2_relative_to_intercept": 1. - result,
                                            "ridge_alpha": best[1]}
        print(f"[P3] τ={horizon}: " + "  ".join(
            f"{label}={results[label][str(horizon)]['relative_mse']:.5f}"
            for label in MODEL_NAMES), flush=True)
    _save_output(folder, "future_probe_v5.json", {
        "protocol": "primingvla-v5-frozen-expert-hidden-state-probing",
        "source": "action_out_proj.forward_pre_hook",
        "information_leakage": "Future inputs pure Gaussian; probe splits disjoint and wholly SAP-heldout",
        "test_episodes": len(test), "fit": len(fit_idx), "validation": len(validation_idx),
        "probe_test": len(test_idx), "context": args.context, "horizons": list(horizons),
        "results": results, "seed": args.seed,
    })


def signature_mmd(args):
    """Finite time-augmented path-signature product-kernel MMD of full ODE samples."""
    folder = _nonexistent_output(args.output)
    data = np.load(args.rollouts, allow_pickle=False)
    models, modes, episode_ids, repeats = (data[key] for key in
                                          ("models", "modes", "episode_ids", "repeats"))
    selected = {}
    for index in range(len(models)):
        if str(modes[index]) != args.mode or int(repeats[index]) != 0:
            continue
        key = (str(models[index]), int(episode_ids[index]))
        if key in selected:
            raise ValueError("Duplicate model/episode in selected path-MMD sample")
        selected[key] = index
    if not selected:
        raise ValueError(f"No {args.mode} ODE rollouts available")
    ids = sorted({episode for _, episode in selected})
    if len(ids) < 4 or any((label, ep) not in selected for ep in ids for label in MODEL_NAMES):
        raise ValueError("All three models need the exact same >=4 heldout episodes")
    reference_ids = [selected[("init", ep)] for ep in ids]
    true = data["ground_truth"][reference_ids]
    hidden = data["hidden"][reference_ids]
    # Different masks do not expose the same context vector. Use a fixed
    # zeroed-where-masked context in the kernel, which is identical per model.
    context = np.where(hidden[:, :, None], 0., true)
    result = {"protocol": "primingvla-v5-finite-signature-conditional-product-MMD",
              "mode": args.mode, "episodes": len(ids),
              "kernel": "RBF(level-1+level-2 time-augmented signature) times RBF(masked-visible-context)",
              "caveat": "Truncated signature joint MMD is NOT full characteristic conditional signature MMD",
              "results": {}}
    for label in MODEL_NAMES:
        prediction = data["generated"][[selected[(label, ep)] for ep in ids]]
        score = conditional_signature_mmd(true, prediction, context)
        result["results"][label] = {"biased_product_mmd_squared": score}
        print(f"[P4] {label}: signature/context MMD²={score:.6g}", flush=True)
    _save_output(folder, "signature_mmd_v5.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("data-check", help="P0: real-data order-vs-bag ridge probe")
    preflight.add_argument("--prepared", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--context", type=int, default=20)
    preflight.add_argument("--horizons", default="5,10,20")
    preflight.add_argument("--seed", type=int, default=117)

    synthetic = commands.add_parser("synthetic-prepare", help="P2: synthesize known stable AR(2) action dynamics")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--train-chunks", type=int, default=256)
    synthetic.add_argument("--test-chunks", type=int, default=120)
    synthetic.add_argument("--seed", type=int, default=2026)

    block = commands.add_parser("blocked-eval", help="P1/P2: exact heldout flow and ODE order intervention")
    readout = commands.add_parser("probe", help="P3: hidden-expert linear future probe")
    for subcommand in (block, readout):
        subcommand.add_argument("--prepared", type=Path, required=True)
        subcommand.add_argument("--base-checkpoint", type=Path, required=True)
        subcommand.add_argument("--sap", type=Path, required=True)
        subcommand.add_argument("--shuffled", type=Path, required=True)
        subcommand.add_argument("--num-chunks", type=int, required=True)
        subcommand.add_argument("--train-masked-steps", type=int, default=40)
        subcommand.add_argument("--output", type=Path, required=True)
        subcommand.add_argument("--test-limit", type=int, default=200)
        subcommand.add_argument("--seed", type=int, default=117)
        subcommand.add_argument("--batch-size", type=int, default=2)
        subcommand.add_argument("--device", default="cuda")
    block.add_argument("--modes", default="future10,middle20,random20")
    block.add_argument("--flow-times", default="0.5,0.99")
    block.add_argument("--ode-steps", type=int, default=16)
    block.add_argument("--repeats", type=int, default=1)
    block.add_argument("--flow-only", action="store_true")
    block.add_argument("--permuted-ode", action="store_true")
    block.add_argument("--save-rollouts", action="store_true")
    readout.add_argument("--context", type=int, default=20)
    readout.add_argument("--horizons", default="1,5,10")

    recovery = commands.add_parser("synthetic-score", help="P2: fit known AR(2) mode parameters")
    recovery.add_argument("--prepared", type=Path, required=True)
    recovery.add_argument("--rollouts", type=Path, required=True)
    recovery.add_argument("--mode", default="future20")
    recovery.add_argument("--output", type=Path, required=True)

    signature = commands.add_parser("signature-mmd", help="P4: time-augmented signature joint-law proxy")
    signature.add_argument("--rollouts", type=Path, required=True)
    signature.add_argument("--mode", default="future10")
    signature.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    runners = {
        "data-check": data_check,
        "blocked-eval": blocked_eval,
        "synthetic-prepare": synthetic_prepare,
        "synthetic-score": synthetic_score,
        "probe": probe,
        "signature-mmd": signature_mmd,
    }
    runners[args.command](args)


if __name__ == "__main__":
    main()
