"""Optional attention-video adapter used by run_eval_libero-full_with_attention.sh."""

import functools
import hashlib
import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.pi05 import attention_visualization
from lerobot.policies.pi05.attention_visualization import (
    AttentionVideoConfig,
    PI05AttentionRecorder,
    load_attention_font,
)


def install_attention_evaluation(evaluator):
    """Install before the launcher's existing progress/resume wrappers."""
    config = AttentionVideoConfig.from_env()
    font = load_attention_font(config.font_path)  # Validate before loading a multi-GB checkpoint.
    original_one = evaluator.run_one
    original_rollout = evaluator.rollout
    one_signature = inspect.signature(original_one)
    rollout_signature = inspect.signature(original_rollout)
    active = {}

    @functools.wraps(original_rollout)
    def rollout(*args, **kwargs):
        bound = rollout_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        recorder = active["recorder"]
        if bound.arguments["env"].num_envs != 1:
            raise ValueError("Attention videos require eval.batch_size=1")
        recorder.reset_episode()
        result = original_rollout(*args, **kwargs)
        if not recorder.maps:
            raise RuntimeError("No prediction attention captured in this episode")
        episode = active["episode"]
        stem = active["directory"] / f"eval_episode_{episode}"
        temporary = stem.with_suffix(".attention.npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                maps=np.stack(recorder.maps),
                input_steps=np.asarray(recorder.steps, dtype=np.int64),
                denoise_pass_counts=np.asarray(recorder.counts, dtype=np.int64),
                image_attention_mass=np.asarray([m.sum() for m in recorder.maps]),
            )
        os.replace(temporary, stem.with_suffix(".attention.npz"))
        metadata = {
            **recorder.metadata(),
            "task_group": active["group"],
            "task_id": active["task_id"],
            "episode": episode,
            "environment_seeds": bound.arguments["seeds"],
        }
        temporary = stem.with_suffix(".attention.json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, stem.with_suffix(".attention.json"))
        active["episode"] += 1
        return result

    @functools.wraps(original_one)
    def run_one(*args, **kwargs):
        bound = one_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if active:
            raise RuntimeError("Attention visualization requires max_parallel_tasks=1")
        if values["videos_dir"] is None or values["recording_dir"] is not None:
            raise ValueError(
                "Attention evaluation requires ordinary evaluation videos, not dataset recording"
            )
        directory = Path(values["videos_dir"]) / f"{values['task_group']}_{values['task_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        # Save every episode, including failures. Do not select only appealing heatmaps.
        values["max_episodes_rendered"] = values["n_episodes"]
        try:
            with PI05AttentionRecorder(values["policy"], config) as recorder:
                active.update(
                    recorder=recorder,
                    directory=directory,
                    episode=0,
                    group=values["task_group"],
                    task_id=values["task_id"],
                )
                result = original_one(*bound.args, **bound.kwargs)
            metrics = result[2]
            paths = metrics.get("video_paths", [])
            if len(paths) != values["n_episodes"]:
                raise RuntimeError("Missing attention videos; task will not be marked complete")
            renamed = []
            for path, success in zip(paths, metrics["successes"], strict=True):
                source = Path(path)
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(f"Video encoding failed: {source}")
                target = source.with_name(f"{source.stem}_{'TRUE' if success else 'FALSE'}_attention.mp4")
                source.replace(target)
                for suffix in (".attention.npz", ".attention.json"):
                    source.with_suffix(suffix).replace(target.with_suffix(suffix))
                renamed.append(str(target))
            metrics["video_paths"] = renamed
            return result
        finally:
            active.clear()

    evaluator.rollout = rollout
    evaluator.run_one = run_one
    print(f"Attention videos enabled: {asdict(config)}", flush=True)
    return {
        "attention": asdict(config),
        "attention_font": {
            "family": font.getname()[0],
            "path": str(font.path),
            "sha256": hashlib.sha256(Path(font.path).read_bytes()).hexdigest(),
        },
        "attention_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(attention_visualization.__file__).read_bytes()
        ).hexdigest(),
    }


def require_saved_attention_videos(metrics):
    """Do not skip a saved task whose diagnostic artifacts were deleted or lost."""
    paths = metrics.get("video_paths", [])
    if len(paths) != len(metrics["successes"]):
        raise RuntimeError("Saved task lacks attention videos; use a fresh BASE_OUTPUT")
    for path in paths:
        video = Path(path)
        for artifact in (video, video.with_suffix(".attention.npz"), video.with_suffix(".attention.json")):
            if not artifact.is_file() or not artifact.stat().st_size:
                raise RuntimeError(f"Missing saved attention artifact: {artifact}; use a fresh BASE_OUTPUT")
