"""Exercise real evaluator/video encoding with a tiny policy and a toy Gym env."""

import json
from pathlib import Path

import av
import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.scripts import lerobot_eval
from lerobot.scripts.libero_attention import install_attention_evaluation, require_saved_attention_videos
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.policies.pi0_pi05.test_pi05_attention_visualization import (
    _test_font,  # noqa: F401
    make_policy_and_batch,
)
from tests.policies.pi0_pi05.test_pi05_prompt import (
    _restore_matmul_precision,  # noqa: F401
    _tiny_real_pi05,  # noqa: F401
)


class TinyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    task_description = "move the object"
    task = "test"
    _max_episode_steps = 5

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8) for key in ("image", "image2")}
                ),
            }
        )
        self.action_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)

    def _observation(self):
        return {"pixels": {key: self.render() for key in ("image", "image2")}}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.succeed = seed % 2 == 0
        return self._observation(), {}

    def step(self, action):
        self.step_index += 1
        return self._observation(), 1.0, self.step_index == 5, False, {"is_success": self.succeed}

    def render(self):
        return np.full((16, 16, 3), self.step_index * 30, dtype=np.uint8)


def test_actual_eval_saves_success_failure_videos_and_resumable_raw_maps(tmp_path, monkeypatch):
    original_one, original_rollout = lerobot_eval.run_one, lerobot_eval.rollout
    monkeypatch.setattr(lerobot_eval, "run_one", original_one)
    monkeypatch.setattr(lerobot_eval, "rollout", original_rollout)
    monkeypatch.setenv("ATTENTION_SOURCE", "action")
    manifest = install_attention_evaluation(lerobot_eval)
    assert manifest["attention"]["source"] == "action"
    policy, _ = make_policy_and_batch()
    env = gym.vector.SyncVectorEnv([TinyEnv])

    def preprocess(batch):
        return {
            **batch,
            OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 0]]),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False]]),
        }

    def identity(value):
        return value

    try:
        result = lerobot_eval.run_one(
            "test_suite",
            0,
            env,
            policy=policy,
            env_preprocessor=identity,
            env_postprocessor=identity,
            preprocessor=preprocess,
            postprocessor=identity,
            n_episodes=2,
            max_episodes_rendered=1,
            videos_dir=tmp_path,
            return_episode_data=False,
            start_seed=0,
        )
    finally:
        env.close()
    metrics = result[2]
    assert metrics["successes"] == [True, False]
    assert len(metrics["video_paths"]) == 2  # save every episode despite incoming limit
    assert "TRUE_attention" in metrics["video_paths"][0]
    assert "FALSE_attention" in metrics["video_paths"][1]
    for path in metrics["video_paths"]:
        video = Path(path)
        with av.open(str(video)) as reader:
            frames = list(reader.decode(video=0))
            assert len(frames) == 5  # preserve existing evaluator's frame count
            assert (frames[0].width, frames[0].height) == (768, 320)
        with np.load(video.with_suffix(".attention.npz")) as data:
            np.testing.assert_array_equal(data["input_steps"], [0, 2, 4])
            assert data["maps"].shape == (3, 2, 2)
            np.testing.assert_allclose(data["image_attention_mass"], data["maps"].sum(axis=(1, 2)))
        meta = json.loads(video.with_suffix(".attention.json").read_text())
        assert meta["camera_feature"] == "observation.images.image"
        assert meta["resolved_layer_zero_based"] == 1
    require_saved_attention_videos(metrics)
    from lerobot.scripts.replot_libero_attention import replot_video

    original = Path(metrics["video_paths"][0])
    restyled = tmp_path / "restyled" / original.name
    assert replot_video(original, restyled) == "5 frames"
    assert (
        original.with_suffix(".attention.npz").read_bytes()
        == restyled.with_suffix(".attention.npz").read_bytes()
    )
    with av.open(str(restyled)) as reader:
        assert len(list(reader.decode(video=0))) == 5
    assert replot_video(original, restyled) == "already done"
    with pytest.raises(ValueError, match="Different output"):
        replot_video(original, restyled, alpha=0.9)
    Path(metrics["video_paths"][0]).with_suffix(".attention.npz").unlink()
    with pytest.raises(RuntimeError, match="Missing saved attention artifact"):
        require_saved_attention_videos(metrics)
    assert not hasattr(policy, "_eval_attention_visualizer")
