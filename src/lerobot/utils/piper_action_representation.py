"""Training-split normalization for the opt-in Piper action representation experiment.

Reads numeric columns only. Neither labels on disk nor dataset-wide metadata are modified.
Every real future action in a chunk contributes; padded tails and cross-episode pairs do not.
"""

from copy import deepcopy

import numpy as np

from lerobot.utils.piper_training_eval import piper_names


def _statistics(values):
    quantiles = np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0)
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "count": np.array([len(values)]),
        **dict(zip(("q01", "q10", "q50", "q90", "q99"), quantiles, strict=True)),
    }


def fit_piper_numeric_stats(columns, features, chunk_size, relative):
    """Fit exact quantiles on selected rows, using one current state per predicted chunk."""
    names, state_names = piper_names(features)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    actions = np.asarray(columns["action"], dtype=np.float64)
    states = np.asarray(columns["observation.state"], dtype=np.float64)
    episodes = np.asarray(columns["episode_index"], dtype=np.int64)
    frames = np.asarray(columns["frame_index"], dtype=np.int64)
    if (
        len(actions) == 0
        or actions.shape != (len(episodes), 7)
        or states.shape != actions.shape
        or frames.shape != episodes.shape
        or not np.isfinite(actions).all()
        or not np.isfinite(states).all()
    ):
        raise ValueError("Piper statistics require nonempty, finite seven-dimensional rows")
    order = np.lexsort((frames, episodes))
    actions, states, episodes, frames = (x[order] for x in (actions, states, episodes, frames))
    state_in_action_order = states[:, [state_names.index(name) for name in names]]
    mask = np.array([name != "gripper.pos" for name in names])
    chunks = []
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        if not np.array_equal(frames[indices], np.arange(len(indices))):
            raise ValueError("Piper statistics require complete episodes with contiguous frame indices")
        for offset in range(min(chunk_size, len(indices))):
            starts = indices[: len(indices) - offset]
            values = actions[starts + offset].copy()
            if relative:
                values[:, mask] -= state_in_action_order[starts][:, mask]
            chunks.append(values)
    targets = np.concatenate(chunks, axis=0)
    stats = {"observation.state": _statistics(states), "action": _statistics(targets)}
    report = {
        "representation": "relative_joints_absolute_gripper" if relative else "absolute",
        "reference": "state at the observation time for the entire chunk; no label shift",
        "chunk_size": chunk_size,
        "training_episodes": np.unique(episodes).tolist(),
        "training_frames": len(states),
        "valid_action_pairs": len(targets),
        "action_names": names,
        "state_names": state_names,
        "stats": {key: {name: value.tolist() for name, value in item.items()} for key, item in stats.items()},
    }
    return stats, report


def fit_piper_train_normalization(dataset, config):
    """Use the filtered HF dataset, not its underlying Arrow table or full dataset metadata."""
    keys = ["action", "observation.state", "episode_index", "frame_index"]
    # with_format(None) avoids video decoding/transforms and select/episode indices are respected.
    columns = dataset.hf_dataset.select_columns(keys).with_format(None)[:]
    stats, report = fit_piper_numeric_stats(
        columns, dataset.meta.features, config.chunk_size, config.use_relative_actions
    )
    if dataset.episodes is not None and set(report["training_episodes"]) != set(dataset.episodes):
        raise ValueError("Normalization rows do not match the selected training episodes")
    result = deepcopy(dataset.meta.stats)
    result.update(stats)
    return result, report
