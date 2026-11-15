#!/usr/bin/env python3
"""Task 2: offline teleoperation / PI05 action prediction plots, with no robot I/O.

Run inside LianYi233/lerobot060, branch piper-partial-ft (or pass --repo-root).
Default checkpoint: task2-fullhead-vlm2-resume10k-to20k / 020000.

Two complementary views are saved:
  1. Episode curves: at recorded frames 0,8,16,... predict 50 actions and retain
     only the first 8 (or the valid tail). Recorded observations are used again
     at every anchor. This is NOT a closed-loop robot rollout.
  2. Individual chunk curves: all valid predicted steps, and a separate h8 zoom.

Only recorded current observations and task text reach the policy. Future action
labels are used for scoring/plotting only. Saved checkpoint processors are loaded;
normalization is never fitted again. No smoothing, clipping, time shifting, or
cross-episode target padding is used. Joint data are assumed to be radians and
plotted in degrees; gripper values keep the dataset's original unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TASK = "2-remove_the_cuboid_from_blue_plate"
TASK_REPO = f"may-pick-and-place/{TASK}"
WORK = Path("/root/autodl-tmp/piper-partial-ft-work")
DEFAULT_POLICY = WORK / "chkpt/2601-lerobot/piper/task2-fullhead-vlm2-resume10k-to20k"
DEFAULT_DATA = Path("/root/autodl-fs/datasets/reality") / TASK
DEFAULT_TOKENIZER = Path("/root/autodl-fs/ckpt/google/paligemma-3b-pt-224")
NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
PROTOCOL = "piper-task2-recorded-observation-h8-v1"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> Any:
    return array(value).item()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def choose_evenly(values: list, count: int) -> list:
    if count == 0 or len(values) <= count:
        return list(values)
    return [values[i] for i in np.linspace(0, len(values) - 1, count, dtype=int)]


def resolve_policy(path: Path, checkpoint_step: int) -> Path:
    """Accept a policy, a checkpoint, a run, or a group containing exactly one run."""
    path = path.expanduser().resolve()
    candidates = [path, path / "pretrained_model",
                  path / "checkpoints" / f"{checkpoint_step:06d}" / "pretrained_model"]
    if path.is_dir():
        candidates += sorted(path.glob(f"*/checkpoints/{checkpoint_step:06d}/pretrained_model"))
    found = sorted({p.resolve() for p in candidates if (p / "config.json").is_file()
                    and (p / "model.safetensors").is_file()})
    if len(found) != 1:
        raise ValueError(f"Expected one full checkpoint at step {checkpoint_step}; found {len(found)} "
                         f"under {path}: {found}. Pass --policy-path explicitly. "
                         "A compact bundle is not a directly loadable policy.")
    return found[0]


def activate_repo(requested: Path | None) -> Path:
    candidates = [requested] if requested else [Path(__file__).resolve().parent, Path.cwd(),
                    Path("/root/autodl-tmp/lerobot060-piper-partial-ft")]
    repo = next((p.expanduser().resolve() for p in candidates if p is not None
                 and (p.expanduser() / "src/lerobot/policies/pi05/modeling_pi05.py").is_file()
                 and (p.expanduser() / "piper_policy_utils.py").is_file()), None)
    if repo is None:
        raise ValueError("Cannot find the piper-partial-ft checkout. Pass --repo-root /path/to/checkout.")
    package = repo / "src/lerobot"
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if (name == "lerobot" or name.startswith("lerobot.")) and origin:
            if not Path(origin).resolve().is_relative_to(package):
                raise RuntimeError(f"Mixed LeRobot checkout: {name} comes from {origin}. Use a new Python process.")
    sys.path[:0] = [str(repo / "src"), str(repo)]
    importlib.invalidate_caches()
    return repo


def find_split(policy: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    # Typical: run/checkpoints/020000/pretrained_model -> run/dataset_split.json.
    for directory in [policy, *list(policy.parents)[:4]]:
        path = directory / "dataset_split.json"
        if path.is_file():
            return path
    return None


def partition_episodes(metadata: Any, training: dict, split_file: Path | None) -> tuple[dict, str]:
    total = int(metadata.total_episodes)
    if split_file:
        raw = read_json(split_file)
        if raw.get("dataset_root") and Path(raw["dataset_root"]).name != TASK:
            raise ValueError(f"Split file belongs to another task: {split_file}")
        groups = {"train": list(map(int, raw.get("train_episodes", []))),
                  "val": list(map(int, raw.get("eval_episodes", [])))}
        origin = str(split_file)
    else:
        base = training.get("episodes")
        base = list(range(total)) if base is None else [int(x) for x in base]
        fraction = training.get("eval_split")
        if fraction is None or not 0 <= float(fraction) < 1:
            raise ValueError("No saved dataset_split.json or valid train_config dataset.eval_split. "
                             "Pass --split-file; the script will not guess the held-out set.")
        by_task: dict[str, list[int]] = {}
        for ep in base:
            tasks = metadata.episodes[ep]["tasks"]
            by_task.setdefault(str(tasks[0]) if tasks else "", []).append(ep)
        groups = {"train": [], "val": []}
        for episodes in by_task.values():
            n_val = math.ceil(len(episodes) * float(fraction))
            cut = len(episodes) - n_val
            groups["train"].extend(episodes[:cut])
            groups["val"].extend(episodes[cut:])
        origin = "reconstructed_from_saved_train_config: last ceil(n*eval_split) per task"
    for key, values in groups.items():
        if len(values) != len(set(values)) or any(ep < 0 or ep >= total for ep in values):
            raise ValueError(f"Invalid {key} episode indices: {values}")
    if set(groups["train"]) & set(groups["val"]):
        raise ValueError("Train/validation episodes overlap.")
    groups["all"] = sorted(set(groups["train"] + groups["val"]))
    return groups, origin


def select_episodes(text: str, groups: dict, count: int) -> tuple[list[int], str]:
    if text in groups:
        eligible = sorted(groups[text])
        selection = text
    else:
        eligible = [int(value.strip()) for value in text.split(",") if value.strip()]
        if len(eligible) != len(set(eligible)) or not set(eligible) <= set(groups["all"]):
            raise ValueError("--episodes must contain distinct episode IDs from this task's saved split.")
        eligible.sort()
        selection = ("val" if set(eligible) <= set(groups["val"]) else
                     "train" if set(eligible) <= set(groups["train"]) else "mixed")
    if not eligible:
        raise ValueError(f"No episodes available for {text!r}.")
    return choose_evenly(eligible, count), selection


def index_episodes(rows: Iterable, selected: list[int], stride: int) -> tuple[list[dict], dict]:
    per_ep = {ep: [] for ep in selected}
    for relative, row in enumerate(rows):
        ep = int(scalar(row["episode_index"]))
        if ep in per_ep:
            per_ep[ep].append({"relative_index": relative, "episode": ep,
                               "frame": int(scalar(row["frame_index"])),
                               "index": int(scalar(row["index"]))})
    anchors = []
    for ep, entries in per_ep.items():
        entries.sort(key=lambda row: row["frame"])
        if not entries or [row["frame"] for row in entries] != list(range(len(entries))):
            raise ValueError(f"Episode {ep} is incomplete or has duplicate frame indices.")
        start = entries[0]["index"]
        if [row["index"] for row in entries] != list(range(start, start + len(entries))):
            raise ValueError(f"Episode {ep} absolute indices are not contiguous.")
        anchors.extend(entries[::stride])
    return anchors, per_ep


def recorded_targets(item: dict, episode: dict, horizon: int, dims: int) -> tuple[np.ndarray, np.ndarray]:
    target = array(item["action"]).astype(np.float32, copy=True)
    padding = array(item["action_is_pad"])
    if target.shape != (horizon, dims) or padding.shape != (horizon,) or padding.dtype != np.bool_:
        raise ValueError(f"Bad target or padding shape/dtype: {target.shape}, {padding.shape}, {padding.dtype}")
    absolute, frame = int(scalar(item["index"])), int(scalar(item["frame_index"]))
    start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if absolute != start + frame or not start <= absolute < end:
        raise ValueError("Dataset global index / episode-local frame mismatch.")
    valid = absolute + np.arange(horizon) < end
    if not np.array_equal(valid, ~padding):
        raise ValueError("Future-action padding does not match the episode boundary.")
    if not np.isfinite(target[valid]).all():
        raise ValueError("Nonfinite recorded action label.")
    return target, valid


def stitch_episode(values: dict, episode: int, execution: int, length: int) -> dict:
    """No averaging of overlapping chunks: retain the actual first-execution window."""
    indices = np.flatnonzero(values["episode_index"] == episode)
    result = {key: np.full((length, values["predicted"].shape[-1]), np.nan, np.float32)
              for key in ("predicted", "target", "hold")}
    result["anchor_frame"] = np.full(length, -1, np.int64)
    for i in indices:
        start = int(values["frame_index"][i])
        offsets = np.flatnonzero(values["valid"][i, :execution])
        slots = start + offsets
        if np.any(slots >= length) or np.any(result["anchor_frame"][slots] >= 0):
            raise ValueError("Stitching would cross an episode or overwrite a predicted frame.")
        result["anchor_frame"][slots] = start
        for key in ("predicted", "target", "hold"):
            result[key][slots] = values[key][i, offsets]
    if np.any(result["anchor_frame"] < 0):
        raise ValueError("Incomplete stitched episode. Anchor stride must equal execution_steps.")
    return result


def metrics(values: dict, names: list[str], horizons: list[int], motion_threshold: float) -> dict:
    result = {}
    joints = [i for i, name in enumerate(names) if name.startswith("joint_")]
    grip = names.index("gripper.pos")
    for horizon in sorted(set(horizons)):
        mask = values["valid"][:, :horizon]
        pred, target, hold = [values[k][:, :horizon].astype(np.float64)[mask]
                              for k in ("predicted", "target", "hold")]
        if not len(target):
            continue
        scale = np.array([180 / math.pi if i in joints else 1.0 for i in range(len(names))])
        error, hold_error = (pred - target) * scale, (hold - target) * scale
        gt_delta, pred_delta = (target - hold) * scale, (pred - hold) * scale
        per_dim = {}
        for i, name in enumerate(names):
            row = {"unit": "deg" if i in joints else "dataset_gripper_unit",
                   "mae": float(np.abs(error[:, i]).mean()),
                   "rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
                   "bias": float(error[:, i].mean()),
                   "hold_mae": float(np.abs(hold_error[:, i]).mean())}
            if i in joints:
                moving = np.abs(gt_delta[:, i]) > motion_threshold
                motion = {"threshold_deg": motion_threshold, "count": int(moving.sum())}
                if moving.any():
                    gt, pd = gt_delta[moving, i], pred_delta[moving, i]
                    gt_amp, pd_amp = float(np.abs(gt).mean()), float(np.abs(pd).mean())
                    motion.update(gt_mean_amplitude_deg=gt_amp, pred_mean_amplitude_deg=pd_amp,
                                  amplitude_ratio=pd_amp / gt_amp,
                                  direction_agreement=float((np.sign(gt) == np.sign(pd)).mean()),
                                  delta_mae_deg=float(np.abs(pd - gt).mean()))
                row["moving_coordinate"] = motion
            per_dim[name] = row
        model_mae = float(np.abs(error[:, joints]).mean())
        hold_mae = float(np.abs(hold_error[:, joints]).mean())
        result[str(horizon)] = {
            "valid_anchor_offset_pairs": int(mask.sum()), "joint_mae_deg": model_mae,
            "hold_joint_mae_deg": hold_mae,
            "joint_mae_ratio_to_hold": model_mae / hold_mae if hold_mae > 1e-12 else None,
            "gripper_mae_dataset_units": float(np.abs(error[:, grip]).mean()),
            "per_dimension": per_dim,
        }
    return result


def draw_pair(stem: Path, seconds: np.ndarray, target: np.ndarray, prediction: np.ndarray,
              name: str, title: str, xlabel: str, *, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joint = name.startswith("joint_")
    scale = 180 / math.pi if joint else 1.0
    label = name.removesuffix(".pos").replace("joint_", "Joint ").replace("gripper", "Gripper")
    mae = float(np.mean(np.abs(prediction - target)) * scale)
    fig, ax = plt.subplots(figsize=(10.8, 3.6), layout="constrained")
    marker = "o" if len(seconds) <= 12 else None
    ax.plot(seconds, target * scale, linewidth=1.8, marker=marker, markersize=3.5,
            label="teleoperation")
    ax.plot(seconds, prediction * scale, linewidth=1.6, linestyle="--", marker=marker,
            markersize=3.0, label="action from model")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{label} (deg)" if joint else "Gripper (dataset unit)")
    ax.set_title(f"{title}\n{label} | MAE = {mae:.3f}" + (" deg" if joint else " dataset unit"), fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", ncol=2, frameon=False)
    if len(seconds) > 1:
        ax.set_xlim(float(seconds[0]), float(seconds[-1]))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def write_episode(output: Path, ep: int, values: dict, names: list[str], fps: float,
                  length: int, execution: int, windows: int, dpi: int) -> None:
    episode_dir = output / f"episode_{ep:06d}"
    episode_dir.mkdir(exist_ok=True)
    stitched = stitch_episode(values, ep, execution, length)
    seconds = np.arange(length, dtype=np.float64) / fps
    with (episode_dir / "episode_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "seconds", "anchor_frame"] +
                        [f"{kind}/{name}" for kind in ("target", "predicted", "hold") for name in names])
        for frame in range(length):
            writer.writerow([frame, seconds[frame], int(stitched["anchor_frame"][frame])] +
                            [float(stitched[kind][frame, j])
                             for kind in ("target", "predicted", "hold") for j in range(len(names))])
    for j, name in enumerate(names):
        draw_pair(episode_dir / "episode" / name.removesuffix(".pos"), seconds,
                  stitched["target"][:, j], stitched["predicted"][:, j], name,
                  f"Task 2 | episode {ep} | first {execution} actions per recorded observation",
                  "Recorded episode time (s)", dpi=dpi)
    ids = [int(i) for i in np.flatnonzero(values["episode_index"] == ep)
           if int(values["valid"][i].sum()) >= execution]
    # Prefer complete H-step windows, so tail padding is not mistaken for a flat target.
    full = [i for i in ids if values["valid"][i].all()]
    if full:
        ids = full
    if windows == 0:
        ids = []
    else:
        ids = choose_evenly(ids, windows)
    for i in ids:
        frame = int(values["frame_index"][i])
        for label, horizon in (("chunk50", values["predicted"].shape[1]), ("first8", execution)):
            offsets = np.flatnonzero(values["valid"][i, :horizon])
            if len(offsets) < 2:
                continue
            for j, name in enumerate(names):
                stem = episode_dir / f"frame_{frame:06d}" / label / name.removesuffix(".pos")
                draw_pair(stem, offsets / fps, values["target"][i, offsets, j],
                          values["predicted"][i, offsets, j], name,
                          f"Task 2 | episode {ep}, anchor {frame} | {len(offsets)} valid predicted actions",
                          "Seconds after recorded observation (offset 0 = current action label)", dpi=dpi)


def write_gallery(output: Path) -> None:
    lines = ["<!doctype html><html><head><meta charset='utf-8'><title>Task 2 offline action plots</title>",
             "<style>body{font-family:sans-serif;margin:24px;max-width:1200px}img{max-width:100%;height:auto}"
             "figure{margin:28px 0}code{overflow-wrap:anywhere}figcaption{font-size:14px}</style></head><body>",
             "<h1>Task 2 — teleoperation vs action from model</h1>",
             "<p>Recorded observations at every replan; first 8 actions retained. "
             "This is offline prediction, not closed-loop robot execution. No smoothing or clipping.</p>"]
    for path in sorted(output.glob("episode_*/*/*.png")) + sorted(output.glob("episode_*/frame_*/*/*.png")):
        relative = path.relative_to(output).as_posix()
        lines.append(f"<figure><figcaption><code>{escape(relative)}</code></figcaption>"
                     f"<a href='{escape(relative)}'><img loading='lazy' src='{escape(relative)}'></a></figure>")
    lines.append("</body></html>")
    (output / "index.html").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", "--repo_root", type=Path)
    parser.add_argument("--policy-path", "--policy_path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--checkpoint-step", type=int, default=20000)
    parser.add_argument("--dataset-root", "--dataset_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tokenizer-path", "--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--episodes", default="val", help="val/train/all, or comma-separated episode IDs")
    parser.add_argument("--num-episodes", type=int, default=3, help="Uniformly choose N eligible episodes; 0 = all")
    parser.add_argument("--execution-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--motion-threshold-deg", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true", help="Validate paths/config/split; no model or GPU inference")
    parser.add_argument("--hash-model", action="store_true", help="SHA256 the large weight file for strict provenance")
    args = parser.parse_args(argv)
    if min(args.num_episodes, args.windows_per_episode, args.seed) < 0:
        parser.error("Episode/window counts and seed must be nonnegative.")
    if args.execution_steps != 8:
        parser.error("This task2 plotting protocol fixes execution_steps=8 for comparable stitching.")
    if args.num_inference_steps < 1 or args.checkpoint_step < 1 or args.dpi < 50:
        parser.error("Inference steps/checkpoint step must be positive; dpi >= 50.")
    if not math.isfinite(args.motion_threshold_deg) or args.motion_threshold_deg <= 0:
        parser.error("motion-threshold-deg must be finite and positive.")
    return args


def run(args: argparse.Namespace) -> Path | None:
    # No Hub fetch and no robot/camera/CAN imports. Offline is mandatory here.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    repo = activate_repo(args.repo_root)
    import torch
    from safetensors import safe_open
    from torch.utils.data import default_collate
    from piper_policy_utils import check_checkpoint_files, check_prompt_weights, deployment_features
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    # Register the exact saved processor implementations before pipeline loading.
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401

    policy_path = resolve_policy(args.policy_path, args.checkpoint_step)
    root, tokenizer = args.dataset_root.expanduser().resolve(), args.tokenizer_path.expanduser().resolve()
    if root.name != TASK:
        raise ValueError(f"Task 2 only: dataset directory must end in {TASK!r}, got {root}")
    info = read_json(root / "meta/info.json")
    if str(info.get("codebase_version", "")).lstrip("v") != "3.0":
        raise ValueError("This script requires the branch's local LeRobot v3.0 dataset.")
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid dataset fps.")
    check_checkpoint_files(policy_path)
    config = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    if not isinstance(config, PI05Config) or config.training_stage != "flow" or config.use_peft:
        raise ValueError("Expected a full flow-stage PI05 model, not an adapter-only checkpoint.")
    if config.chunk_size != 50 or config.action_delta_indices != list(range(50)):
        raise ValueError("Expected 50 action labels at offsets 0..49.")
    if config.n_action_steps != args.execution_steps:
        raise ValueError(f"Saved execution horizon {config.n_action_steps} differs from requested 8.")
    if getattr(config, "use_relative_actions", False) or getattr(config, "piper_gripper_history_steps", 0):
        raise ValueError("This Task 2 protocol is for the current absolute-action, no-history checkpoint.")
    if not (tokenizer / "tokenizer_config.json").is_file() or not any(
            (tokenizer / n).is_file() for n in ("tokenizer.json", "tokenizer.model")):
        raise ValueError(f"Incomplete local tokenizer directory: {tokenizer}")
    features = deployment_features(config, info)
    names, state_names = features["action"]["names"], features["observation.state"]["names"]
    check_prompt_weights(policy_path, config)
    train_config_path = policy_path / "train_config.json"
    train_config = read_json(train_config_path) if train_config_path.is_file() else {}
    training = train_config.get("dataset", {})
    evidence = []
    for key in ("root", "repo_id"):
        if training.get(key):
            if str(training[key]).rstrip("/").rsplit("/", 1)[-1] != TASK:
                raise ValueError(f"Checkpoint's dataset.{key} belongs to a different task: {training[key]}")
            evidence.append(f"train_config.dataset.{key}")
    for path in (policy_path / "dataset_info.json", policy_path.parent.parent.parent / "dataset_info.json"):
        if path.is_file():
            if read_json(path) != info:
                raise ValueError(f"Dataset info differs from checkpoint provenance: {path}")
            evidence.append(str(path))
    if not evidence:
        raise ValueError("Cannot verify task identity. Keep the original train_config.json/dataset_info.json.")
    metadata = LeRobotDatasetMetadata(training.get("repo_id") or TASK_REPO, root=root)
    split_file = find_split(policy_path, args.split_file)
    groups, split_origin = partition_episodes(metadata, training, split_file)
    episodes, selection = select_episodes(args.episodes, groups, args.num_episodes)
    print(f"[CONFIG] repo={repo}\n[POLICY] {policy_path}\n[DATASET] {root}\n"
          f"[SPLIT] {selection}; episodes={episodes}; source={split_origin}\n"
          f"[PROTOCOL] chunk=50, execute=8, flow={args.num_inference_steps}, fps={fps:g}, seed={args.seed}", flush=True)
    if args.preflight_only:
        print("PREFLIGHT_OK: no model instantiated, no robot connection.", flush=True)
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        from PIL import Image  # noqa: F401; needed to save recorded camera observations
    except ImportError as exc:
        raise RuntimeError("Plotting dependencies missing. In the active environment run: "
                           "python -m pip install matplotlib pillow") from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in the active environment.")
    if device.type != "cuda" and config.dtype != "float32":
        raise ValueError("Use CUDA for this BF16 checkpoint; the script does not silently change precision.")
    if config.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected GPU does not support BF16.")
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported PI05 precision: {config.dtype}")
    config.device, config.pretrained_path = str(device), policy_path
    config.compile_model, config.gradient_checkpointing, config.rtc_config = False, False, None
    config.num_inference_steps, config.tokenizer_name = args.num_inference_steps, str(tokenizer)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    dataset = LeRobotDataset(training.get("repo_id") or TASK_REPO, root=root, episodes=episodes,
                            delta_timestamps=resolve_delta_timestamps(config, metadata),
                            video_backend=args.video_backend, return_uint8=True, image_transforms=None)
    anchors, per_episode = index_episodes(
        dataset.select_columns(["episode_index", "frame_index", "index"]), episodes, args.execution_steps)
    # Select observation snapshots without using any prediction/error information.
    snapshot_anchors = set()
    if args.windows_per_episode:
        for ep, entries in per_episode.items():
            available = [a for a in anchors if a["episode"] == ep and len(entries)-a["frame"] >= 8]
            full = [a for a in available if len(entries)-a["frame"] >= 50]
            for anchor in choose_evenly(full or available, args.windows_per_episode):
                snapshot_anchors.add((ep, anchor["frame"]))
    # Preserve fitted statistics. Only override device and the local tokenizer path.
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)},
                                "tokenizer_processor": {"tokenizer_name": str(tokenizer)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    for step in preprocessor.steps:
        if any(k != v for k, v in getattr(step, "rename_map", {}).items()):
            raise ValueError("Nontrivial saved observation rename requires explicit camera-key auditing.")
    output = args.output_dir or WORK / "offline_plots/task2-fullhead-vlm2-20k" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = output.expanduser().resolve()
    if any(output == p or output.is_relative_to(p) for p in (policy_path, root, tokenizer)):
        raise ValueError("Output must be outside the checkpoint, tokenizer and dataset directories.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    stamp = datetime.now(timezone.utc).isoformat()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
    source_hashes = {str(p.relative_to(repo)): sha256(p) for p in (
        repo / "src/lerobot/policies/pi05/modeling_pi05.py",
        repo / "src/lerobot/policies/pi05/processor_pi05.py",
        repo / "src/lerobot/policies/pi_gemma.py")}
    manifest = {"protocol": PROTOCOL, "completed": False, "started_at": stamp,
                "policy_path": str(policy_path), "config_sha256": sha256(policy_path / "config.json"),
                "model_bytes": (policy_path / "model.safetensors").stat().st_size,
                "model_sha256": sha256(policy_path / "model.safetensors") if args.hash_model else None,
                "repository": str(repo), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
                "source_sha256": source_hashes, "script_sha256": sha256(Path(__file__)),
                "torch": torch.__version__, "python": sys.version, "device": str(device),
                "dtype": config.dtype, "bf16_autocast": config.dtype == "bfloat16" and device.type == "cuda",
                "dataset_root": str(root), "dataset_info_sha256": sha256(root / "meta/info.json"),
                "identity_evidence": evidence, "split_source": split_origin, "selection": selection,
                "episodes": episodes, "fps": fps, "seed": args.seed, "chunk_size": 50,
                "execution_steps": args.execution_steps, "num_inference_steps": args.num_inference_steps,
                "action_names": names, "state_names": state_names,
                "units": "joints radians in raw arrays/CSV, degrees in figures/metrics; gripper unchanged",
                "evaluation": "Recorded current RGB+state+language; independent replan every 8 frames. "
                              "Not closed-loop; no future labels in policy input. No clipping or smoothing.",
                "metric_weighting": "h8 uses non-overlapping frame targets; h50 counts anchor/offset pairs, "
                                    "so future targets may occur more than once.",
                "selection_policy": "Uniform episodes and windows by index; no selection using model error."}
    write_json(output / "run_manifest.json", manifest)
    print(f"[OUTPUT] {output}\n[LOAD] Full policy, strict=True", flush=True)
    policy = PI05Policy.from_pretrained(str(policy_path), config=config, local_files_only=True, strict=True)
    policy.to(device).eval()
    # Check that strict loading preserved the learned prompt tensors (never random prompts).
    with safe_open(str(policy_path / "model.safetensors"), framework="pt", device="cpu") as saved:
        for key in ("model.vlm_prompt_tokens.weight", "model.prompt_tokens.weight"):
            actual = policy.state_dict()[key].detach().cpu()
            torch.testing.assert_close(actual, saved.get_tensor(key).to(dtype=actual.dtype), rtol=0, atol=0)
    cameras = list(config.image_features)
    order = [state_names.index(name) for name in names]
    collected = {k: [] for k in ("predicted", "target", "hold", "valid", "episode_index",
                                "frame_index", "index", "task", "inference_seconds")}
    begin = time.monotonic()
    for n, anchor in enumerate(anchors, start=1):
        item = dataset[anchor["relative_index"]]
        ep, frame = int(scalar(item["episode_index"])), int(scalar(item["frame_index"]))
        if (ep, frame, int(scalar(item["index"]))) != (anchor["episode"], anchor["frame"], anchor["index"]):
            raise ValueError("Selected row differs from decoded observation indices.")
        target, valid = recorded_targets(item, metadata.episodes[ep], 50, len(names))
        if not np.array_equal(target[0], array(dataset.get_raw_item(anchor["relative_index"])["action"])):
            raise ValueError("offset=0 does not equal the current recorded action label.")
        current = array(item["observation.state"]).astype(np.float32)
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError("Expected a finite seven-dimensional current state.")
        hold = np.repeat(current[order][None, :], 50, axis=0)
        if not isinstance(item["task"], str) or not item["task"].strip():
            raise ValueError("Recorded task text is missing.")
        observation = {key: deepcopy(item[key]) for key in ["observation.state", *cameras]}
        observation["task"] = item["task"]
        if (ep, frame) in snapshot_anchors:
            from PIL import Image
            destination = output / f"episode_{ep:06d}" / f"frame_{frame:06d}" / "observations"
            destination.mkdir(parents=True, exist_ok=True)
            for camera in cameras:
                pixels = array(item[camera])
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[0] != 3:
                    raise ValueError(f"Expected CHW uint8 RGB observation for {camera}.")
                filename = camera.removeprefix("observation.images.").replace("/", "_")
                Image.fromarray(np.moveaxis(pixels, 0, -1)).save(destination / f"{filename}.png")
        for component in (policy, preprocessor, postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        batch = default_collate([observation])
        for key in cameras:
            if batch[key].dtype != torch.uint8:
                raise ValueError(f"Expected uint8 RGB for {key}; refusing accidental double image normalization.")
            batch[key] = batch[key].float() / 255.0
        batch = preprocessor(batch)
        if batch.pop("action", None) is not None:
            raise ValueError("Future action labels must not enter the policy.")
        generator = torch.Generator(device=device).manual_seed((args.seed + anchor["index"]) % (2**63 - 1))
        noise = torch.randn((1, 50, config.max_action_dim), generator=generator, device=device)
        amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device.type == "cuda" and config.dtype == "bfloat16" else nullcontext())
        t0 = time.monotonic()
        with torch.inference_mode(), amp:
            normalized = policy.predict_action_chunk(batch, noise=noise)
            if tuple(normalized.shape) != (1, 50, 7):
                raise ValueError(f"Expected predicted action [1,50,7], got {tuple(normalized.shape)}")
        # Postprocess outside autocast; per-step [batch,action_dim] interface.
        with torch.inference_mode():
            prediction = torch.stack([postprocessor(normalized[:, k].float()) for k in range(50)], dim=1)
        prediction = prediction[0].detach().float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite action prediction at episode {ep}, frame {frame}")
        for key, value in {"predicted": prediction, "target": target, "hold": hold, "valid": valid,
                           "episode_index": ep, "frame_index": frame, "index": anchor["index"],
                           "task": item["task"], "inference_seconds": time.monotonic() - t0}.items():
            collected[key].append(value)
        if n % 10 == 0 or n == len(anchors):
            print(f"[PREDICT] {n}/{len(anchors)} | episode={ep} frame={frame}", flush=True)
        # Preserve numerical results after each complete episode, even if plotting later fails.
        if n == len(anchors) or anchors[n]["episode"] != ep:
            partial = {key: np.asarray(value) for key, value in collected.items()}
            mask_ep = partial["episode_index"] == ep
            ep_dir = output / f"episode_{ep:06d}"
            ep_dir.mkdir(exist_ok=True)
            np.savez_compressed(ep_dir / "predictions.npz",
                                **{key: val[mask_ep] for key, val in partial.items()}, action_names=np.asarray(names))
    values = {key: np.asarray(value) for key, value in collected.items()}
    np.savez_compressed(output / "predictions.npz", **values, action_names=np.asarray(names))
    aggregate = metrics(values, names, [1, 8, 50], args.motion_threshold_deg)
    per_episode = {ep: {"length": len(entries)} for ep, entries in per_episode.items()}
    for ep in episodes:
        mask_ep = values["episode_index"] == ep
        subset = {key: val[mask_ep] for key, val in values.items()}
        per_episode[ep]["metrics"] = metrics(subset, names, [8], args.motion_threshold_deg)["8"]
        write_episode(output, ep, values, names, fps, per_episode[ep]["length"], 8,
                      args.windows_per_episode, args.dpi)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon", "dimension", "unit", "mae", "rmse", "bias", "hold_mae"])
        for h, scores in aggregate.items():
            for name, row in scores["per_dimension"].items():
                writer.writerow([h, name, row["unit"], row["mae"], row["rmse"], row["bias"], row["hold_mae"]])
    write_gallery(output)
    summary = {**manifest, "completed": True, "finished_at": datetime.now(timezone.utc).isoformat(),
               "wall_seconds_inference_and_plotting": time.monotonic() - begin,
               "num_observations": len(anchors), "horizons": aggregate, "per_episode": per_episode}
    write_json(output / "summary.json", summary)
    write_json(output / "run_manifest.json", {**manifest, "completed": True})
    h8 = aggregate["8"]
    print(f"[H8] joint MAE={h8['joint_mae_deg']:.4f} deg | hold={h8['hold_joint_mae_deg']:.4f} deg | "
          f"gripper MAE={h8['gripper_mae_dataset_units']:.5f}")
    for key in ("joint_2.pos", "joint_3.pos"):
        row = h8["per_dimension"][key]
        print(f"[{key}] MAE={row['mae']:.4f} deg; moving={row.get('moving_coordinate')}")
    print(f"TASK2_PLOT_OK: {output / 'index.html'}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"TASK2_PLOT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
