#!/usr/bin/env python3
"""V4: independent-episode, data-scale diagnostic of action-only SAP.

Subcommands:
  prepare: select nested 2/32/128/512 real action chunks and held-out episodes.
  train:   optimize the original PI0.5 Stage-I action prompts on one subset.
  evaluate: compare exact masked flow-target errors on independent episodes.
  summarize: aggregate four sample sizes into a compact figure and JSON.

All training/model imports are lazy so data/split utilities are CPU-testable.
Only the prompt tensor is saved, never the pretrained model backbone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from two_sample_sap_structure import (  # noqa: E402
    _checkpoint_prompt,
    _configure_pi05,
    _get_torch,
    _save_json,
    _set_seed,
    quantile_normalize,
)
from two_sample_sap_structure_v3 import flow_alignment_scores  # noqa: E402


DEFAULT_SIZES = (2, 32, 128, 512)
DEFAULT_TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
MODEL_NAMES = ("init", "sap", "shuffled")


def parse_positive_ints(value: str, *, minimum: int = 1) -> tuple[int, ...]:
    try:
        numbers = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, got {value!r}") from exc
    if not numbers or any(x < minimum for x in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError(f"Invalid unique positive integers: {value!r}")
    return numbers


def parse_flow_times(value: str) -> tuple[float, ...]:
    try:
        numbers = tuple(float(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid flow time list: {value!r}") from exc
    if not numbers or any(not 0.0 < t < 1.0 for t in numbers):
        raise ValueError("Flow times must lie strictly between zero and one")
    return numbers


def episode_split(records: list[dict], test_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by whole episode, stratifying by task when available."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("Test fraction must lie between zero and one")
    episodes = [int(r["episode_index"]) for r in records]
    if len(episodes) != len(set(episodes)):
        raise ValueError("Duplicate episode metadata")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("task", "unknown"))].append(record)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for task in sorted(groups):
        rows = groups[task]
        order = rng.permutation(len(rows))
        # A task with one episode cannot occur in both sets.
        count = max(1, int(round(len(rows) * test_fraction))) if len(rows) > 1 else 0
        count = min(count, len(rows) - 1)
        test.extend(rows[i] for i in order[:count])
        train.extend(rows[i] for i in order[count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def action_strata(chunks: np.ndarray) -> list[str]:
    """Label action windows *before* selection; thresholds use observed dynamics."""
    x = np.asarray(chunks, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError("Expected [N,T,D>=2] action chunks")
    velocities = np.diff(x[..., :-1], axis=1)
    activity = np.linalg.norm(velocities, axis=-1).mean(axis=1)
    has_event = (np.abs(np.diff(x[..., -1], axis=1)) > 0.5).any(axis=1)
    no_event = activity[~has_event]
    threshold = float(np.median(no_event)) if no_event.size else float(np.median(activity))
    return [
        "transition" if event else "movement" if motion > threshold else "steady"
        for event, motion in zip(has_event, activity, strict=True)
    ]


def balanced_order(strata: list[str], seed: int) -> list[int]:
    """Interleave fixed motion strata, preserving nested subsets across N."""
    rng = np.random.default_rng(seed)
    groups = {}
    for label in ("transition", "movement", "steady"):
        indices = [i for i, group in enumerate(strata) if group == label]
        groups[label] = list(map(int, rng.permutation(indices)))
    output = []
    while any(groups.values()):
        for label in ("transition", "movement", "steady"):
            if groups[label]:
                output.append(groups[label].pop())
    if sorted(output) != list(range(len(strata))):
        raise AssertionError("Each eligible action chunk must be selected once")
    return output


def make_mask(batch: int, horizon: int, hidden: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= hidden < horizon:
        raise ValueError("Hidden length must be 1..T-1")
    mask = np.zeros((batch, horizon), dtype=bool)
    for i in range(batch):
        mask[i, rng.permutation(horizon)[:hidden]] = True
    return mask


def aggregate_by_episode(rows: list[dict], field: str, *, mask_steps: int) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["mask_steps"] == mask_steps:
            scores[row["model"]][int(row["episode_index"])].append(float(row[field]))
    return {model: {eid: float(np.mean(values)) for eid, values in episodes.items()}
            for model, episodes in scores.items()}


def paired_bootstrap_gap(sap: dict[int, float], shuffled: dict[int, float], *,
                         seed: int = 117, iterations: int = 2000) -> dict:
    """Positive gap means SAP has lower held-out error; resample episodes."""
    if set(sap) != set(shuffled) or not sap:
        raise ValueError("SAP/Shuffled must score the exact same independent episodes")
    episodes = sorted(sap)
    diffs = np.array([shuffled[e] - sap[e] for e in episodes], dtype=np.float64)
    if not np.isfinite(diffs).all():
        raise ValueError("Non-finite evaluation score")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(iterations, len(diffs)))
    resampled = diffs[indices].mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {"episodes": len(episodes), "shuffled_minus_sap": float(diffs.mean()),
            "ci95": [float(low), float(high)]}


def _read_episode_records(dataset_root: Path, horizon: int) -> tuple[list[dict], dict, dict]:
    import pyarrow.dataset as pa_ds

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    if not isinstance(info.get("data_path"), str):
        raise ValueError("LIBERO meta/info.json requires a data_path template")
    if "action" not in stats or not {"q01", "q99"}.issubset(stats["action"]):
        raise ValueError("LIBERO meta/stats.json requires action.q01/q99")
    metadata = pa_ds.dataset(str(dataset_root / "meta" / "episodes"), format="parquet")
    required = ["episode_index", "dataset_from_index", "dataset_to_index",
                "data/chunk_index", "data/file_index"]
    optional = [x for x in ("task_index", "tasks", "task_indices")
                if x in metadata.schema.names]
    missing = set(required) - set(metadata.schema.names)
    if missing:
        raise ValueError(f"Episode metadata missing required columns: {sorted(missing)}")
    rows = metadata.to_table(columns=required + optional).to_pylist()
    records = []
    for row in rows:
        length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        if length < horizon:
            continue
        task_value = next((row[key] for key in optional if row.get(key) is not None), "unknown")
        records.append({**row, "task": str(task_value), "episode_length": length})
    return records, info, stats


def _choose_candidates(rows: list[dict], count: int, seed: int, horizon: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Need {count} independent eligible episodes, have only {len(rows)}")
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    for items in groups.values():
        rng.shuffle(items)
    tasks = list(groups)
    rng.shuffle(tasks)
    chosen = []
    while len(chosen) < count:
        for task in tasks:
            if not groups[task]:
                continue
            row = dict(groups[task].pop())
            row["offset"] = int(rng.integers(0, row["episode_length"] - horizon + 1))
            chosen.append(row)
            if len(chosen) == count:
                break
    return chosen


def _read_action_windows(dataset_root: Path, candidates: list[dict], info: dict,
                         stats: dict, horizon: int) -> tuple[np.ndarray, list[dict], list[str]]:
    """Read episode-relative contiguous windows, rejecting malformed candidates.

    dataset_from_index in metadata is not guaranteed to match the index values
    within every Parquet shard. Prefer episode-local frame_index where present;
    otherwise use sorted within-episode index order. Never mix episodes or
    silently accept discontinuous frames.
    """
    import pyarrow.dataset as pa_ds

    result = [None] * len(candidates)
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(candidates):
        path = dataset_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        groups[str(path)].append((i, row))
    rejected = []
    for path, entries in groups.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing action file: {path}")
        wanted = [int(row["episode_index"]) for _, row in entries]
        dataset = pa_ds.dataset(path, format="parquet")
        sequence_key = "frame_index" if "frame_index" in dataset.schema.names else "index"
        if sequence_key not in dataset.schema.names:
            raise ValueError(f"{path} has neither frame_index nor index for ordering")
        table = dataset.to_table(
            columns=["episode_index", sequence_key, "action"],
            filter=pa_ds.field("episode_index").isin(wanted),
        ).to_pydict()
        by_episode: dict[int, list[tuple[int, object]]] = defaultdict(list)
        for ep, index, action in zip(table["episode_index"], table[sequence_key],
                                     table["action"], strict=True):
            by_episode[int(ep)].append((int(index), action))
        for i, row in entries:
            episode_id = int(row["episode_index"])
            offset = int(row["offset"])
            episode_rows = sorted(by_episode[episode_id], key=lambda item: item[0])
            selected = episode_rows[offset:offset + horizon]
            indices = [index for index, _ in selected]
            if (len(indices) != horizon
                    or indices != list(range(indices[0], indices[0] + horizon))):
                rejected.append(
                    f"episode={episode_id}, file={Path(path).name}, "
                    f"key={sequence_key}, offset={offset}, rows={len(episode_rows)}, "
                    f"selected={len(indices)}, first_indices={indices[:5]}"
                )
                continue
            result[i] = np.asarray([action for _, action in selected], dtype=np.float32)
    valid_indices = [i for i, value in enumerate(result) if value is not None]
    if not valid_indices:
        raise ValueError(f"No valid {horizon}-step action chunks; examples: {rejected[:3]}")
    raw = np.stack([result[i] for i in valid_indices], axis=0)
    return (
        quantile_normalize(raw, stats["action"]["q01"], stats["action"]["q99"]),
        [candidates[i] for i in valid_indices],
        rejected,
    )


def prepare(args) -> None:
    torch = _get_torch()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Prepared dataset already exists; refusing overwrite: {output}")
    if args.horizon != 50:
        raise ValueError("PI0.5 action chunk horizon is 50; use --horizon 50")
    sizes = parse_positive_ints(args.sizes, minimum=2)
    max_size = max(sizes)
    if args.test_chunks < 2 or args.candidate_multiplier < 1:
        raise ValueError("Need >=2 held-out episodes and candidate_multiplier >= 1")
    records, info, stats = _read_episode_records(dataset_root, args.horizon)
    train_pool, test_pool = episode_split(records, args.test_fraction, args.seed)
    train_pool_size = min(len(train_pool), max(max_size, math.ceil(max_size * args.candidate_multiplier)))
    test_pool_size = min(len(test_pool), max(args.test_chunks, math.ceil(args.test_chunks * args.candidate_multiplier)))
    train_candidates = _choose_candidates(train_pool, train_pool_size, args.seed + 23, args.horizon)
    test_candidates = _choose_candidates(test_pool, test_pool_size, args.seed + 47, args.horizon)
    train_all, train_candidates, train_rejected = _read_action_windows(
        dataset_root, train_candidates, info, stats, args.horizon
    )
    test_all, test_candidates, test_rejected = _read_action_windows(
        dataset_root, test_candidates, info, stats, args.horizon
    )
    if train_rejected or test_rejected:
        print(f"[PREPARE] Excluded incomplete/noncontiguous episodes: "
              f"train={len(train_rejected)}, test={len(test_rejected)}", flush=True)
        for issue in (train_rejected + test_rejected)[:5]:
            print(f"[PREPARE]   {issue}", flush=True)
    if len(train_candidates) < max_size or len(test_candidates) < args.test_chunks:
        raise ValueError(
            f"After excluding invalid windows, only {len(train_candidates)} train "
            f"and {len(test_candidates)} test episodes remain; need "
            f"{max_size} and {args.test_chunks}. Increase candidate_multiplier "
            f"or inspect the reported Parquet data."
        )
    train_labels = action_strata(train_all)
    test_labels = action_strata(test_all)
    train_order = balanced_order(train_labels, args.seed + 71)[:max_size]
    test_order = balanced_order(test_labels, args.seed + 73)[:args.test_chunks]
    train = np.ascontiguousarray(train_all[train_order], dtype=np.float32)
    test = np.ascontiguousarray(test_all[test_order], dtype=np.float32)
    train_meta = [{**train_candidates[i], "stratum": train_labels[i]} for i in train_order]
    test_meta = [{**test_candidates[i], "stratum": test_labels[i]} for i in test_order]
    train_episodes = {int(row["episode_index"]) for row in train_meta}
    test_episodes = {int(row["episode_index"]) for row in test_meta}
    if train_episodes & test_episodes:
        raise AssertionError("Episode leakage between train and heldout test")
    manifest = {
        "protocol": "primingvla-sap-structure-v4-episode-disjoint",
        "dataset_root": str(dataset_root), "dataset_repo_id": info.get("repo_id"),
        "seed": args.seed, "horizon": args.horizon, "action_dim": int(train.shape[-1]),
        "sizes": sorted(sizes), "heldout_chunks": len(test_meta),
        "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "train": train_meta, "test": test_meta,
        "selection": "one offset per episode, task-round-robin candidate sampling, "
                     "movement/transition/steady interleaving; nested train subsets; "
                     "ordered by episode-local frame_index or within-episode index",
        "excluded_candidate_windows": {
            "train": len(train_rejected), "test": len(test_rejected),
            "examples": (train_rejected + test_rejected)[:10],
        },
        "disclaimer": "Episodes are held out from SAP, not necessarily from pi05_libero_base pretraining",
    }
    output.mkdir(parents=True)
    torch.save({"train": torch.from_numpy(train), "test": torch.from_numpy(test),
                "manifest": manifest}, output / "prepared_v4.pt")
    _save_json(output / "manifest_v4.json", manifest)
    for split in ("train", "test"):
        bins = {name: sum(row["stratum"] == name for row in manifest[split])
                for name in ("transition", "movement", "steady")}
        print(f"[PREPARE] {split}: {len(manifest[split])} independent episodes; strata={bins}", flush=True)
    print(f"[PREPARE] nested sizes {sorted(sizes)}; saved to {output}", flush=True)


def _load_prepared(path: Path):
    torch = _get_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("manifest", {}).get("protocol") != "primingvla-sap-structure-v4-episode-disjoint":
        raise ValueError("Not a prepared V4 experiment")
    manifest = payload["manifest"]
    for key in ("train", "test"):
        actions = payload[key].detach().float().cpu().contiguous().numpy()
        if hashlib.sha256(actions.tobytes()).hexdigest() != manifest[f"{key}_sha256"]:
            raise ValueError(f"Prepared {key} actions do not match manifest SHA256")
        if len(actions) != len(manifest[key]):
            raise ValueError(f"Prepared {key} metadata count mismatch")
    a = {int(x["episode_index"]) for x in manifest["train"]}
    b = {int(x["episode_index"]) for x in manifest["test"]}
    if a & b:
        raise ValueError("Train/test episode leakage in prepared payload")
    return payload


def train(args) -> None:
    torch = _get_torch()
    import lerobot
    from lerobot.policies.pi05.configuration_pi05 import PI05TrainingObjective
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, pad_vector

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {output}")
    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.masked_steps < 50:
        raise ValueError("Invalid training steps, batch size, or masked steps")
    payload = _load_prepared(args.prepared)
    manifest = payload["manifest"]
    n = args.num_chunks
    if n not in manifest["sizes"] or n > len(payload["train"]):
        raise ValueError(f"Number of chunks {n} is not a prepared nested size")
    if args.batch_size > n:
        raise ValueError("For matched data sampling, batch_size must not exceed N")
    _set_seed(args.seed)
    cfg = _configure_pi05(args.base_checkpoint, args.device)
    cfg.next_action_masked_steps = args.masked_steps
    policy = PI05Policy.from_pretrained(args.base_checkpoint, config=cfg, local_files_only=True)
    policy.train()
    for name, param in policy.named_parameters():
        param.requires_grad_(name == "model.prompt_tokens.weight")
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if len(trainable) != 1 or trainable[0][0] != "model.prompt_tokens.weight":
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in trainable]}")
    optimizer = cfg.get_optimizer_preset().build([trainable[0][1]])
    scheduler = cfg.get_scheduler_preset().build(optimizer, max(args.steps, 1_000))
    initial = _checkpoint_prompt(policy)
    data = payload["train"][:n].float().cpu()
    action_dim = data.shape[-1]
    rng = np.random.default_rng(args.seed + 2026)
    index_rng = torch.Generator(device="cpu").manual_seed(args.seed + 3407)
    trace = []
    output.mkdir(parents=True)
    print(f"[TRAIN] lerobot={Path(lerobot.__file__).resolve()}", flush=True)
    print(f"[TRAIN] N={n}, variant={args.variant}, masked={args.masked_steps}/50, "
          f"B={args.batch_size}, updates={args.steps}", flush=True)
    for step in range(1, args.steps + 1):
        batch_ids = torch.randperm(n, generator=index_rng)[:args.batch_size]
        clean = data[batch_ids].to(device=args.device, dtype=torch.float32)
        clean = pad_vector(clean, cfg.max_action_dim)
        if args.variant == "shuffled":
            clean = clean[:, rng.permutation(cfg.chunk_size).copy(), :]
        padding = torch.zeros((args.batch_size, cfg.chunk_size), dtype=torch.bool, device=args.device)
        mask = policy.model.sample_inpainting_mask(padding)
        optimizer.zero_grad(set_to_none=True)
        losses = policy.model.forward(actions=clean, action_is_pad=padding,
                                      inpainting_mask=mask,
                                      objective=PI05TrainingObjective.ACTION_INPAINTING)
        loss = losses[:, :, :action_dim].sum() / (mask.sum() * action_dim)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            value = float(loss.detach())
            trace.append({"step": step, "loss": value, "lr": optimizer.param_groups[0]["lr"]})
            print(f"[{args.variant}/N{n}] {step}/{args.steps}: loss={value:.6f}", flush=True)
    trained = _checkpoint_prompt(policy)
    torch.save({"prompt_initial": initial, "prompt_trained": trained},
               output / "action_prompt.pt")
    difference = torch.linalg.vector_norm(trained - initial)
    metadata = {
        "protocol": "primingvla-sap-structure-v4", "variant": args.variant,
        "num_chunks": n, "batch_size": args.batch_size, "steps": args.steps,
        "seed": args.seed, "masked_steps": args.masked_steps,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "prepared_train_sha256": manifest["train_sha256"],
        "selected_train_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        "training_episode_ids": [int(r["episode_index"]) for r in manifest["train"][:n]],
        "trainable_parameters": trained.numel(),
        "prompt_absolute_change_l2": float(difference),
        "prompt_relative_change_l2": float(difference / torch.linalg.vector_norm(initial).clamp_min(1e-12)),
        "objective": "original PI0.5 action-only masked Flow Matching, no CPC",
        "shuffle_protocol": "independent temporal permutation each optimizer update" if args.variant == "shuffled" else None,
        "loss_trace": trace,
    }
    _save_json(output / "train_metadata.json", metadata)
    print(f"[TRAIN] saved prompt and metadata: {output}", flush=True)


def _load_prompts(args, manifest):
    torch = _get_torch()
    saved = {}
    initial = None
    for label, folder in (("sap", args.sap), ("shuffled", args.shuffled)):
        metadata = json.loads((folder / "train_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("variant") != label or metadata.get("protocol") != "primingvla-sap-structure-v4":
            raise ValueError(f"Invalid V4 {label} training checkpoint: {folder}")
        if metadata["prepared_train_sha256"] != manifest["train_sha256"]:
            raise ValueError(f"Prepared dataset differs from {label} training dataset")
        if metadata["base_checkpoint"] != str(args.base_checkpoint.resolve()):
            raise ValueError("Evaluation must use exactly the same base checkpoint as training")
        if metadata["num_chunks"] != args.num_chunks or metadata["masked_steps"] != args.train_masked_steps:
            raise ValueError("N or train-masked-steps do not match the trained prompts")
        weights = torch.load(folder / "action_prompt.pt", map_location="cpu", weights_only=False)
        if initial is None:
            initial = weights["prompt_initial"].float()
        elif not torch.equal(initial, weights["prompt_initial"].float()):
            raise ValueError("SAP and Shuffled do not share identical initialization")
        saved[label] = weights["prompt_trained"].float()
    return {"init": initial, **saved}


def evaluate(args) -> None:
    torch = _get_torch()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, pad_vector

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    if args.repeats < 1 or args.batch_size < 1 or args.bootstrap_iterations < 100:
        raise ValueError("Invalid evaluation repeats/batch size/bootstrap iterations")
    mask_sizes = parse_positive_ints(args.eval_masks)
    if any(x >= 50 for x in mask_sizes):
        raise ValueError("Evaluation mask sizes must be less than 50")
    times = parse_flow_times(args.flow_times)
    payload = _load_prepared(args.prepared)
    manifest = payload["manifest"]
    prompts = _load_prompts(args, manifest)
    _set_seed(args.seed)
    cfg = _configure_pi05(args.base_checkpoint, args.device, gradient_checkpointing=False)
    cfg.next_action_masked_steps = args.train_masked_steps
    policy = PI05Policy.from_pretrained(args.base_checkpoint, config=cfg, local_files_only=True)
    policy.eval()
    test = payload["test"].float().cpu()
    dims, horizon = test.shape[-1], test.shape[1]
    if any(tuple(weights.shape) != tuple(policy.model.prompt_tokens.weight.shape)
           for weights in prompts.values()):
        raise ValueError("Prompt weights incompatible with pretrained policy")
    observations: list[dict] = []
    with torch.inference_mode():
        for hidden in mask_sizes:
            for repeat in range(args.repeats):
                rng = np.random.default_rng(args.seed + hidden * 100003 + repeat * 7919)
                for first in range(0, len(test), args.batch_size):
                    last = min(first + args.batch_size, len(test))
                    batch = test[first:last].to(device=args.device)
                    count = len(batch)
                    masks_np = make_mask(count, horizon, hidden, rng)
                    noise_np = rng.standard_normal((count, horizon, cfg.max_action_dim), dtype=np.float32)
                    mask = torch.as_tensor(masks_np, dtype=torch.bool, device=args.device)
                    noise = torch.as_tensor(noise_np, dtype=torch.float32, device=args.device)
                    clean = pad_vector(batch, cfg.max_action_dim)
                    visible = ~mask.unsqueeze(-1)
                    action_is_pad = torch.zeros((count, horizon), dtype=torch.bool, device=args.device)
                    target = noise - clean
                    for t in times:
                        x_t = torch.where(visible, clean, (1.0 - t) * clean + t * noise)
                        timestep = torch.full((count,), t, dtype=torch.float32, device=args.device)
                        estimated = {}
                        for label, weights in prompts.items():
                            policy.model.prompt_tokens.weight.copy_(weights.to(device=args.device))
                            estimated[label] = policy.model.predict_inpainting_velocity(
                                x_t, timestep, action_is_pad, mask
                            )[:, :, :dims].float().cpu().numpy()
                        ideal = target[:, :, :dims].float().cpu().numpy()
                        for bi in range(count):
                            episode = manifest["test"][first + bi]
                            selected = masks_np[bi]
                            base_estimate = estimated["init"][bi, selected]
                            for label in MODEL_NAMES:
                                prediction = estimated[label][bi, selected]
                                reference = ideal[bi, selected]
                                flow = flow_alignment_scores(prediction, reference)
                                shift = np.linalg.norm(prediction - base_estimate)
                                base_norm = max(np.linalg.norm(base_estimate), 1e-12)
                                observations.append({
                                    "episode_index": int(episode["episode_index"]),
                                    "stratum": episode["stratum"],
                                    "model": label, "mask_steps": hidden,
                                    "repeat": repeat, "flow_time": t,
                                    "flow_mse": float(np.mean((prediction - reference) ** 2)),
                                    "flow_relative_mse": flow["flow_relative_mse"],
                                    "flow_cosine": flow["flow_cosine"],
                                    "relative_output_shift_vs_init": float(shift / base_norm),
                                })
                print(f"[EVAL] N={args.num_chunks}, mask={hidden}, repeat={repeat + 1}/{args.repeats}", flush=True)
    summary = {"protocol": "primingvla-sap-structure-v4-heldout",
               "num_chunks": args.num_chunks, "test_episodes": len(test),
               "eval_masks": list(mask_sizes), "train_masked_steps": args.train_masked_steps,
               "flow_times": list(times), "repeats": args.repeats, "seed": args.seed,
               "prepared_test_sha256": manifest["test_sha256"], "masks": {}}
    for hidden in mask_sizes:
        by_model = aggregate_by_episode(observations, "flow_relative_mse", mask_steps=hidden)
        shifts = aggregate_by_episode(observations, "relative_output_shift_vs_init", mask_steps=hidden)
        gap = paired_bootstrap_gap(by_model["sap"], by_model["shuffled"],
                                   seed=args.seed + hidden, iterations=args.bootstrap_iterations)
        summary["masks"][str(hidden)] = {
            "models": {label: {
                "flow_relative_mse": float(np.mean(list(by_model[label].values()))),
                "flow_relative_mse_std_across_episodes": float(np.std(list(by_model[label].values()), ddof=1)),
                "relative_output_shift_vs_init": float(np.mean(list(shifts[label].values()))),
            } for label in MODEL_NAMES},
            "paired_gap": gap,
        }
        print(f"[RESULT] N={args.num_chunks}, mask={hidden}: "
              f"SAP={summary['masks'][str(hidden)]['models']['sap']['flow_relative_mse']:.5f}, "
              f"shuffled={summary['masks'][str(hidden)]['models']['shuffled']['flow_relative_mse']:.5f}, "
              f"shuffled-SAP={gap['shuffled_minus_sap']:+.5f} "
              f"95%CI=[{gap['ci95'][0]:+.5f}, {gap['ci95'][1]:+.5f}]", flush=True)
    output.mkdir(parents=True)
    _save_json(output / "metrics_v4.json", summary)
    _save_json(output / "per_episode_v4.json", observations)
    print(f"[EVAL] saved: {output}", flush=True)


def summarize(args) -> None:
    rows = []
    sizes = parse_positive_ints(args.sizes, minimum=2)
    for n in sizes:
        file = args.input_root / "eval" / f"n{n}-seed{args.seed}" / "metrics_v4.json"
        result = json.loads(file.read_text(encoding="utf-8"))
        if result["num_chunks"] != n:
            raise ValueError(f"Unexpected N in {file}")
        record = result["masks"].get(str(args.primary_mask))
        if record is None:
            raise ValueError(f"Primary mask={args.primary_mask} missing in {file}")
        rows.append({"num_chunks": n, "init": record["models"]["init"]["flow_relative_mse"],
                     "sap": record["models"]["sap"]["flow_relative_mse"],
                     "shuffled": record["models"]["shuffled"]["flow_relative_mse"],
                     "gap": record["paired_gap"]["shuffled_minus_sap"],
                     "gap_ci95": record["paired_gap"]["ci95"],
                     "test_episodes": result["test_episodes"],
                     "test_sha256": result["prepared_test_sha256"]})
    if len({r["test_sha256"] for r in rows}) != 1:
        raise ValueError("Different sample-size experiments must use identical heldout episodes")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary output: {output}")
    output.mkdir(parents=True)
    _save_json(output / "sample_size_summary_v4.json",
               {"primary_mask": args.primary_mask, "rows": rows,
                "interpretation": "Positive paired gap means lower heldout flow error for SAP."})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SUMMARY] matplotlib unavailable; JSON summary still saved", flush=True)
        return
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for name, color in (("init", "#89929b"), ("sap", "#006d77"), ("shuffled", "#c57b39")):
        axes[0].plot(x, [row[name] for row in rows], marker="o", lw=2.2,
                     color=color, label=name)
    axes[0].set_title("Held-out flow-field error")
    axes[0].set_ylabel("Relative MSE ↓")
    axes[0].legend(frameon=False)
    delta = np.array([row["gap"] for row in rows])
    intervals = np.array([row["gap_ci95"] for row in rows])
    axes[1].errorbar(x, delta, yerr=np.stack([delta - intervals[:, 0], intervals[:, 1] - delta]),
                     marker="o", lw=1.8, color="#006d77", capsize=4)
    axes[1].axhline(0, color="#87919c", ls="--", lw=1)
    axes[1].set_title("Benefit of ordered SAP")
    axes[1].set_ylabel("Shuffled MSE − SAP MSE ↑")
    for axis in axes:
        axis.set_xticks(x, [str(row["num_chunks"]) for row in rows])
        axis.set_xlabel("Independent training chunks")
        axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"sample_size_v4.{extension}", dpi=220)
    plt.close(fig)
    for row in rows:
        print(f"[SUMMARY] N={row['num_chunks']}: SAP={row['sap']:.5f}, "
              f"shuffled={row['shuffled']:.5f}, gap={row['gap']:+.5f}", flush=True)
    print(f"[SUMMARY] saved: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare independent-episode data splits")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--sizes", default="2,32,128,512")
    prep.add_argument("--test-chunks", type=int, default=200)
    prep.add_argument("--test-fraction", type=float, default=0.25)
    prep.add_argument("--candidate-multiplier", type=float, default=2.0)
    prep.add_argument("--horizon", type=int, default=50)
    prep.add_argument("--seed", type=int, default=0)

    training = commands.add_parser("train", help="Train a single original SAP or shuffled control")
    training.add_argument("--prepared", type=Path, required=True)
    training.add_argument("--base-checkpoint", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--variant", choices=("sap", "shuffled"), required=True)
    training.add_argument("--num-chunks", type=int, required=True)
    training.add_argument("--steps", type=int, default=750)
    training.add_argument("--masked-steps", type=int, default=40)
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cuda")
    training.add_argument("--log-every", type=int, default=50)

    evaluation = commands.add_parser("evaluate", help="Paired held-out masked flow-field test")
    evaluation.add_argument("--prepared", type=Path, required=True)
    evaluation.add_argument("--base-checkpoint", type=Path, required=True)
    evaluation.add_argument("--sap", type=Path, required=True)
    evaluation.add_argument("--shuffled", type=Path, required=True)
    evaluation.add_argument("--num-chunks", type=int, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--eval-masks", default="40,20")
    evaluation.add_argument("--train-masked-steps", type=int, default=40)
    evaluation.add_argument("--flow-times", default="0.1,0.3,0.5,0.7,0.9")
    evaluation.add_argument("--repeats", type=int, default=2)
    evaluation.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluation.add_argument("--batch-size", type=int, default=2)
    evaluation.add_argument("--seed", type=int, default=117)
    evaluation.add_argument("--device", default="cuda")

    summ = commands.add_parser("summarize", help="Figure + JSON for all data sizes")
    summ.add_argument("--input-root", type=Path, required=True)
    summ.add_argument("--output", type=Path, required=True)
    summ.add_argument("--sizes", default="2,32,128,512")
    summ.add_argument("--seed", type=int, default=0)
    summ.add_argument("--primary-mask", type=int, default=40)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
