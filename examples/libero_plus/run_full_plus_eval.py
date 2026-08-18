#!/usr/bin/env python
"""Resume-safe full LIBERO-plus evaluation (~10,030 perturbation tasks).

Official LIBERO-plus protocol (sylvestf/LIBERO-plus): 1 trial per perturbation
task across the four suites (spatial / object / goal / libero_10). That is *not*
the 10-task × 10-episode vanilla LIBERO protocol.

Creates one task env at a time so peak RAM stays at a single MuJoCo instance
plus the policy. Progress is appended after every task and the run can be
restarted with the same ``--output_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoPlusEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import _get_suite, create_libero_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_one
from lerobot.utils.random_utils import set_seed

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

CATEGORY_TO_COL = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
LEADERBOARD_COLS = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")


def _numpy2_compat() -> None:
    """LIBERO-plus sensor-noise fog() still uses NumPy 1 aliases (np.float_)."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int_"):
        np.int_ = np.int64  # type: ignore[attr-defined]


def _patch_libero_plus_noise_corruptions() -> None:
    """Make LIBERO-plus ImageNet-C fog() work on 360x360 LeRobot images.

    Upstream ``plasma_fractal`` defaults to 256 and NumPy 1's ``np.float_``.
    """
    _numpy2_compat()
    try:
        from libero.libero.envs import env_wrapper
    except Exception as exc:
        print(f"[{_utc_now()}] skip libero-plus noise patch: {exc}", flush=True)
        return

    orig_plasma = env_wrapper.plasma_fractal

    def fog(x, severity=1):
        table = [
            (0.5, 3),
            (1.0, 2.8),
            (1.5, 2.5),
            (2.0, 2.2),
            (2.5, 2.0),
            (3.0, 1.8),
            (3.5, 1.6),
            (4.0, 1.5),
            (4.5, 1.4),
            (5.0, 1.3),
        ]
        c = table[severity - 1]
        arr = np.array(x) / 255.0
        max_val = arr.max()
        height_x, width_x = int(arr.shape[0]), int(arr.shape[1])
        mapsize = 1 << (max(height_x, width_x) - 1).bit_length()
        fractal = orig_plasma(mapsize=mapsize, wibbledecay=c[1])[:height_x, :width_x]
        arr = arr + c[0] * fractal[..., np.newaxis]
        return np.clip(arr * max_val / (max_val + c[0]), 0, 1) * 255

    env_wrapper.fog = fog
    print(f"[{_utc_now()}] patched libero-plus fog() for non-256 images / NumPy 2", flush=True)


def _close_quiet(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_classification(path: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for suite, entries in raw.items():
        by_tid: dict[int, dict[str, Any]] = {}
        for entry in entries:
            # Official JSON ids are 1-indexed; LeRobot task_ids are 0-indexed.
            by_tid[int(entry["id"]) - 1] = entry
        out[suite] = by_tid
    return out


def _agg(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=float)))


def _success_pct(successes: list[bool]) -> float:
    if not successes:
        return float("nan")
    return float(np.mean(np.asarray(successes, dtype=float)) * 100.0)


def build_eval_info(
    rows: list[dict[str, Any]],
    classification: dict[str, dict[int, dict[str, Any]]],
    eval_s: float,
) -> dict[str, Any]:
    per_group_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {"sum_rewards": [], "max_rewards": [], "successes": []}
    )
    overall = {"sum_rewards": [], "max_rewards": [], "successes": []}
    dim_acc: dict[str, list[bool]] = defaultdict(list)
    per_task: list[dict[str, Any]] = []

    for row in rows:
        suite = row["suite"]
        tid = int(row["task_id"])
        metrics = {
            "sum_rewards": row["sum_rewards"],
            "max_rewards": row["max_rewards"],
            "successes": row["successes"],
        }
        per_task.append({"task_group": suite, "task_id": tid, "metrics": metrics, **row.get("extra", {})})
        per_group_acc[suite]["sum_rewards"].extend(row["sum_rewards"])
        per_group_acc[suite]["max_rewards"].extend(row["max_rewards"])
        per_group_acc[suite]["successes"].extend(row["successes"])
        overall["sum_rewards"].extend(row["sum_rewards"])
        overall["max_rewards"].extend(row["max_rewards"])
        overall["successes"].extend(row["successes"])
        entry = classification.get(suite, {}).get(tid)
        if entry is not None:
            col = CATEGORY_TO_COL.get(entry.get("category", ""), entry.get("category"))
            dim_acc[col].extend(row["successes"])

    per_group = {
        suite: {
            "avg_sum_reward": _agg(acc["sum_rewards"]),
            "avg_max_reward": _agg(acc["max_rewards"]),
            "pc_success": _success_pct(acc["successes"]),
            "n_episodes": len(acc["successes"]),
        }
        for suite, acc in per_group_acc.items()
    }
    by_dimension = {
        col: {
            "pc_success": _success_pct(dim_acc[col]),
            "n_episodes": len(dim_acc[col]),
        }
        for col in LEADERBOARD_COLS
        if col in dim_acc
    }
    return {
        "protocol": {
            "benchmark": "libero-plus",
            "n_episodes_per_task": 1,
            "suites": list(SUITES),
            "note": (
                "Official LIBERO-plus uses 1 trial per perturbation task "
                "(~10,030 tasks). This is not the vanilla 10-task x 10-episode protocol."
            ),
        },
        "per_task": per_task,
        "per_group": per_group,
        "by_dimension": by_dimension,
        "overall": {
            "avg_sum_reward": _agg(overall["sum_rewards"]),
            "avg_max_reward": _agg(overall["max_rewards"]),
            "pc_success": _success_pct(overall["successes"]),
            "n_episodes": len(overall["successes"]),
            "eval_s": eval_s,
            "eval_ep_s": eval_s / max(1, len(overall["successes"])),
        },
        "updated_at": _utc_now(),
    }


def format_summary(info: dict[str, Any]) -> str:
    overall = info["overall"]
    lines = [
        f"LIBERO-plus full eval  n={overall['n_episodes']}  "
        f"success={overall['pc_success']:.1f}%  "
        f"elapsed_s={overall['eval_s']:.1f}",
        "",
        "Per suite:",
    ]
    for suite, stats in info.get("per_group", {}).items():
        lines.append(
            f"  {suite:16s}  {stats['pc_success']:6.1f}%  n={stats['n_episodes']}"
        )
    if info.get("by_dimension"):
        lines.append("")
        lines.append("Per perturbation dimension (official leaderboard columns):")
        header = "  ".join(f"{c:>12s}" for c in LEADERBOARD_COLS)
        values = "  ".join(
            f"{info['by_dimension'].get(c, {}).get('pc_success', float('nan')):11.1f}%"
            if c in info.get("by_dimension", {})
            else f"{'n/a':>12s}"
            for c in LEADERBOARD_COLS
        )
        lines.append("  " + header)
        lines.append("  " + values)
        lines.append(f"  {'Total':>12s}  {overall['pc_success']:11.1f}%")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy.path", dest="policy_path", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--env.task", dest="task", default=",".join(SUITES))
    p.add_argument("--env.control_mode", dest="control_mode", default="relative")
    p.add_argument("--eval.n_episodes", dest="n_episodes", type=int, default=1)
    p.add_argument("--policy.n_action_steps", dest="n_action_steps", type=int, default=10)
    p.add_argument("--policy.use_amp", dest="use_amp", type=lambda s: str(s).lower() == "true", default=False)
    p.add_argument("--policy.device", dest="device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--classification_json",
        type=Path,
        default=None,
        help="LIBERO-plus task_classification.json (auto-detected when omitted).",
    )
    return p.parse_args()


def _find_classification_json() -> Path | None:
    import libero

    roots: list[Path] = []
    if getattr(libero, "__file__", None):
        roots.append(Path(libero.__file__).resolve().parent)
    for entry in getattr(libero, "__path__", []):
        roots.append(Path(entry))
    for root in roots:
        candidate = root / "libero" / "benchmark" / "task_classification.json"
        if candidate.exists():
            return candidate
        candidate = root / "benchmark" / "task_classification.json"
        if candidate.exists():
            return candidate
    env_path = os.environ.get("LIBERO_PLUS_CLASSIFICATION")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    return None


def main() -> None:
    _patch_libero_plus_noise_corruptions()
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    errors_path = output_dir / "errors.jsonl"
    eval_info_path = output_dir / "eval_info.json"
    summary_path = output_dir / "summary.txt"

    suites = [s.strip() for s in str(args.task).split(",") if s.strip()]
    classification_path = args.classification_json or _find_classification_json()
    classification = _load_classification(classification_path)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg = LiberoPlusEnv(task=suites[0], control_mode=args.control_mode)
    policy_cfg = PreTrainedConfig.from_pretrained(str(args.policy_path))
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.n_action_steps = args.n_action_steps
    policy_cfg.use_amp = args.use_amp
    policy_cfg.device = args.device

    print(f"[{_utc_now()}] loading policy from {args.policy_path}", flush=True)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    done_rows = _load_jsonl(progress_path)
    done_keys = {(row["suite"], int(row["task_id"])) for row in done_rows}
    started = time.time()
    # Preserve elapsed time from a previous partial run when present.
    prior_eval_s = 0.0
    if eval_info_path.exists():
        try:
            prior_eval_s = float(json.loads(eval_info_path.read_text())["overall"].get("eval_s", 0.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            prior_eval_s = 0.0

    print(
        f"[{_utc_now()}] suites={suites} n_episodes={args.n_episodes} "
        f"already_done={len(done_keys)} classification={classification_path}",
        flush=True,
    )

    with torch.no_grad():
        for suite_name in suites:
            suite = _get_suite(suite_name)
            n_tasks = len(suite.tasks)
            print(f"[{_utc_now()}] suite={suite_name} n_tasks={n_tasks}", flush=True)
            # One lazy vec-env per task: only the current task's MuJoCo env is constructed.
            envs = create_libero_envs(
                task=suite_name,
                n_envs=1,
                camera_name=env_cfg.camera_name,
                init_states=env_cfg.init_states,
                gym_kwargs=dict(env_cfg.gym_kwargs),
                env_cls=gym.vector.SyncVectorEnv,
                control_mode=args.control_mode,
                episode_length=env_cfg.episode_length,
                camera_name_mapping=env_cfg.camera_name_mapping,
                is_libero_plus=True,
            )
            try:
                for tid in range(n_tasks):
                    vec = envs[suite_name][tid]
                    if (suite_name, tid) in done_keys:
                        _close_quiet(vec)
                        continue
                    t0 = time.time()
                    metrics = None
                    last_error: str | None = None
                    for attempt in range(2):
                        try:
                            metrics = eval_one(
                                vec,
                                policy=policy,
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                n_episodes=args.n_episodes,
                                max_episodes_rendered=0,
                                videos_dir=None,
                                return_episode_data=False,
                                start_seed=args.seed,
                            )
                            break
                        except Exception as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            traceback.print_exc()
                            _close_quiet(vec)
                            print(
                                f"[{_utc_now()}] {suite_name} task {tid} "
                                f"attempt {attempt + 1}/2 failed: {last_error}",
                                flush=True,
                            )
                    _close_quiet(vec)
                    if metrics is None:
                        _append_jsonl(
                            errors_path,
                            {
                                "suite": suite_name,
                                "task_id": tid,
                                "error": last_error,
                                "finished_at": _utc_now(),
                            },
                        )
                        print(
                            f"[{_utc_now()}] {suite_name} task {tid} skipped after errors; "
                            f"see {errors_path}",
                            flush=True,
                        )
                        continue

                    row = {
                        "suite": suite_name,
                        "task_id": tid,
                        "sum_rewards": list(metrics["sum_rewards"]),
                        "max_rewards": list(metrics["max_rewards"]),
                        "successes": list(metrics["successes"]),
                        "elapsed_s": time.time() - t0,
                        "finished_at": _utc_now(),
                    }
                    extra = classification.get(suite_name, {}).get(tid)
                    if extra:
                        row["extra"] = {
                            "name": extra.get("name"),
                            "category": extra.get("category"),
                            "difficulty_level": extra.get("difficulty_level"),
                        }
                    _append_jsonl(progress_path, row)
                    done_rows.append(row)
                    done_keys.add((suite_name, tid))

                    n_done = len(done_keys)
                    elapsed = prior_eval_s + (time.time() - started)
                    info = build_eval_info(done_rows, classification, elapsed)
                    if n_done % 10 == 0 or tid + 1 == n_tasks:
                        _atomic_write_json(eval_info_path, info)
                        summary_path.write_text(format_summary(info))

                    ok = sum(bool(s) for s in row["successes"])
                    n_ep = len(row["successes"])
                    print(
                        f"[{_utc_now()}] {suite_name} task {tid}/{n_tasks - 1} "
                        f"success={ok}/{n_ep}  {row['elapsed_s']:.1f}s  "
                        f"overall={info['overall']['pc_success']:.1f}% "
                        f"done={n_done}",
                        flush=True,
                    )
            finally:
                for vec in envs.get(suite_name, {}).values():
                    _close_quiet(vec)

    elapsed = prior_eval_s + (time.time() - started)
    info = build_eval_info(done_rows, classification, elapsed)
    _atomic_write_json(eval_info_path, info)
    summary_path.write_text(format_summary(info))
    print(format_summary(info), flush=True)
    print(f"[{_utc_now()}] wrote {eval_info_path}", flush=True)


if __name__ == "__main__":
    main()
