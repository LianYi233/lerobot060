"""Read-only audit of raw Piper parquet coordinates, before allocating policy weights.

This cannot determine whether actions were commanded or measured, or verify camera
exposure synchronization. It never shifts labels, recomputes stats, or edits the dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.utils.piper_training_eval import PIPER_NAMES, piper_names


def coordinate_error(action, state):
    error = np.asarray(action, dtype=np.float64) - np.asarray(state, dtype=np.float64)
    if len(error) == 0:
        return {"pairs": 0}
    return {
        "pairs": len(error),
        "joint_mae_deg": float(np.rad2deg(np.abs(error[:, :6])).mean()),
        "joint_mae_deg_per_coordinate": np.rad2deg(np.abs(error[:, :6])).mean(axis=0).tolist(),
        "gripper_mae_dataset_units": float(np.abs(error[:, 6]).mean()),
        "all_coordinates_equal_fraction_at_1e-6": float((np.abs(error).max(axis=1) <= 1e-6).mean()),
    }


def audit_arrays(action, state, episodes, frames, timestamps, fps):
    """Arrays use canonical Piper name order; adjacent comparisons never cross episodes."""
    action, state = np.asarray(action, dtype=np.float64), np.asarray(state, dtype=np.float64)
    episodes, frames, timestamps = (np.asarray(x) for x in (episodes, frames, timestamps))
    size = len(action)
    if size == 0 or action.shape != (size, 7) or state.shape != action.shape:
        raise ValueError("Expected nonempty raw action/state arrays with seven coordinates")
    if any(x.shape != (size,) for x in (episodes, frames, timestamps)):
        raise ValueError("Invalid episode/frame/timestamp shape")
    if not all(np.isfinite(x).all() for x in (action, state, episodes, frames, timestamps)):
        raise ValueError("Nonfinite raw coordinates or row identifiers")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be positive")
    if any(not np.equal(x, np.floor(x)).all() or (x < 0).any() for x in (episodes, frames)):
        raise ValueError("Episode and frame indices must be nonnegative integers")
    order = np.lexsort((frames, episodes))
    action, state, episodes, frames, timestamps = (
        x[order] for x in (action, state, episodes, frames, timestamps)
    )
    same_episode = episodes[1:] == episodes[:-1]
    if not (np.diff(frames)[same_episode] == 1).all():
        raise ValueError("Duplicate or missing frame indices within a raw episode")
    starts = np.r_[True, ~same_episode]
    if not (frames[starts] == 0).all():
        raise ValueError("Raw episodes must start at frame_index=0")
    dt = np.diff(timestamps)[same_episode]
    if len(dt) and (dt <= 0).any():
        raise ValueError("Timestamps must increase inside each episode")
    previous, following = np.flatnonzero(same_episode), np.flatnonzero(same_episode) + 1
    report = {
        "frames": size,
        "episodes": int(starts.sum()),
        "fps": fps,
        "coordinate_order": PIPER_NAMES,
        "comparisons": {
            "action_t_vs_state_t": coordinate_error(action, state),
            "action_t_vs_state_t_plus_1": coordinate_error(action[previous], state[following]),
            "action_t_vs_state_t_minus_1": coordinate_error(action[following], state[previous]),
        },
        "per_episode": [],
        "timestamp_interval_seconds": {
            "median": float(np.median(dt)) if len(dt) else None,
            "max_abs_error_from_1_over_fps": float(np.max(np.abs(dt - 1 / fps))) if len(dt) else None,
        },
        "ranges": {
            key: {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
            for key, values in (("action", action), ("state", state))
        },
        "limitations": [
            "Joint metrics assume absolute radians; gripper remains in dataset units.",
            "Equality does not prove a labeling bug; verify commanded versus measured action semantics.",
            "Parquet timestamps do not verify camera exposure synchronization; inspect recorded video.",
            "No dataset or normalization statistics were modified.",
        ],
    }
    for episode in np.unique(episodes):
        mask = episodes == episode
        report["per_episode"].append({"episode": int(episode), **coordinate_error(action[mask], state[mask])})
    return report


def audit_dataset(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    action_names, state_names = piper_names(info["features"])
    columns = ["action", "observation.state", "episode_index", "frame_index", "timestamp"]
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files under {root}")
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in files])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action = action[:, [action_names.index(name) for name in PIPER_NAMES]]
    state = state[:, [state_names.index(name) for name in PIPER_NAMES]]
    report = audit_arrays(
        action, state, *(table[name].to_numpy() for name in columns[2:]), fps=float(info["fps"])
    )
    if report["frames"] != info["total_frames"] or report["episodes"] != info["total_episodes"]:
        raise ValueError("Raw parquet row/episode counts disagree with meta/info.json")
    return {"dataset_root": str(root), "scope": "all raw episodes before train/val split", **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PIPER_DATA_AUDIT:", json.dumps(report["comparisons"], ensure_ascii=False))
    print(f"DATA_AUDIT_SAVED: {args.output}")
    if report["comparisons"]["action_t_vs_state_t"]["all_coordinates_equal_fraction_at_1e-6"] > 0.95:
        print(
            "DATA_SEMANTICS_NOTE: >95% action[t] equals state[t]. Verify the recording convention; "
            "this is not automatically an error. Labels are unchanged."
        )


if __name__ == "__main__":
    main()
