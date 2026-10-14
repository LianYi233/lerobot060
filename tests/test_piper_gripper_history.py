"""Causality, selected-episode indexing, normalization persistence and command history."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.processor_pi05 import (
    Pi05GripperHistoryProcessorStep,
    validate_pi05_action_processors,
)
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, UnnormalizerProcessorStep
from lerobot.utils.piper_gripper_history import (
    CONTEXT_KEY,
    HISTORY_KEY,
    ExecutedGripperHistory,
    RecordedGripperHistory,
    configure_gripper_history,
)

NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def columns():
    n = 30
    return {
        "episode_index": [0] * n + [9] * n,
        "frame_index": list(range(n)) * 2,
        "action": [[0] * 6 + [i / 100] for i in range(n * 2)],
        "observation.state": [[0] * 6 + [0.8]] * n + [[0] * 6 + [0.9]] * n,
    }


def test_history_is_strictly_past_and_episode_local():
    raw = columns()
    history = RecordedGripperHistory(raw, NAMES, NAMES)
    np.testing.assert_allclose(history[0], [0.8] * 20)
    np.testing.assert_allclose(history[1], [0.8] * 19 + [0.0])
    np.testing.assert_allclose(history[25], np.arange(5, 25) / 100)
    np.testing.assert_allclose(history[30], [0.9] * 20)
    np.testing.assert_allclose(history[31], [0.9] * 19 + [0.30])
    changed = deepcopy(raw)
    for i in range(25, 60):
        changed["action"][i] = [999] * 7
    np.testing.assert_array_equal(history[25], RecordedGripperHistory(changed, NAMES, NAMES)[25])
    with pytest.raises(ValueError, match="complete episodes"):
        RecordedGripperHistory({k: v[1:] for k, v in raw.items()}, NAMES, NAMES)


def test_hf_selected_rows_and_dataset_getitem_keep_targets_unchanged():
    # HF select has an indices table: do not use hf_dataset.data directly.
    hf = Dataset.from_dict(columns()).select(range(30, 60))
    features = {key: {"shape": [7], "names": NAMES} for key in ("action", "observation.state")}
    dataset = SimpleNamespace(hf_dataset=hf, meta=SimpleNamespace(features=features))
    configure_gripper_history(dataset, SimpleNamespace(piper_gripper_history_steps=20))
    np.testing.assert_allclose(dataset._piper_gripper_history[0], [0.9] * 20)
    target = torch.full((16, 7), 123.0)
    dataset.writer = None
    dataset._ensure_reader = lambda: SimpleNamespace(hf_dataset=hf, get_item=lambda idx: {"action": target})
    item = LeRobotDataset.__getitem__(dataset, 1)
    assert item["action"] is target
    torch.testing.assert_close(item[HISTORY_KEY], torch.tensor([0.9] * 19 + [0.30]))


def test_named_gripper_order():
    raw = columns()
    permutation = [6, 0, 1, 2, 3, 4, 5]
    raw["action"] = np.asarray(raw["action"])[:, permutation].tolist()
    reordered = [NAMES[i] for i in permutation]
    np.testing.assert_allclose(RecordedGripperHistory(raw, reordered, NAMES)[25], np.arange(5, 25) / 100)


def test_command_history_reset_clamp_and_copy():
    buffer = ExecutedGripperHistory(0.7)
    before = buffer.observation()
    buffer.append(-1)
    buffer.append(2)
    np.testing.assert_allclose(buffer.observation(), [0.7] * 18 + [0, 1])
    np.testing.assert_allclose(before, [0.7] * 20)
    np.testing.assert_allclose(ExecutedGripperHistory(0.3).observation(), [0.3] * 20)


def test_context_normalization_and_saved_processor_roundtrip(tmp_path):
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    modes = {FeatureType.STATE: NormalizationMode.QUANTILES, FeatureType.ACTION: NormalizationMode.QUANTILES}
    stats = {
        "observation.state": {"q01": torch.zeros(7), "q99": torch.ones(7) * 4},
        "action": {"q01": torch.ones(7) * 0.2, "q99": torch.ones(7) * 0.8},
    }
    normalizer = NormalizerProcessorStep(features=features, norm_map=modes, stats=stats)
    history = Pi05GripperHistoryProcessorStep(
        features={"action": features["action"]}, norm_map=modes, stats={"action": stats["action"]}
    )
    pipeline = PolicyProcessorPipeline(steps=[normalizer, history], name="history_test")
    # No action label is passed to inference preprocessing.
    batch = {"observation.state": torch.ones(2, 7), HISTORY_KEY: torch.ones(2, 20) * 0.8}
    result = pipeline(deepcopy(batch))[CONTEXT_KEY]
    assert result.shape == (2, 32)
    torch.testing.assert_close(result[:, :7], torch.full((2, 7), -0.5))
    torch.testing.assert_close(result[:, 7:27], torch.ones(2, 20))
    torch.testing.assert_close(result[:, 27:], torch.zeros(2, 5), atol=0, rtol=0)
    pipeline.save_pretrained(tmp_path)
    restored = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="history_test.json")
    torch.testing.assert_close(restored(deepcopy(batch))[CONTEXT_KEY], result, atol=0, rtol=0)
    post = SimpleNamespace(steps=[UnnormalizerProcessorStep(features=features, norm_map=modes, stats=stats)])
    config = SimpleNamespace(
        piper_gripper_history_steps=20, action_feature_names=NAMES, use_relative_actions=False
    )
    validate_pi05_action_processors(config, restored, post)
    with pytest.raises(ValueError, match="causal"):
        restored({"observation.state": torch.ones(2, 7)})
    with pytest.raises(ValueError, match="disagree"):
        validate_pi05_action_processors(config, SimpleNamespace(steps=[normalizer]), post)
    restored.steps[1]._tensor_stats["action"]["q01"].add_(1)
    with pytest.raises(ValueError, match="statistics"):
        validate_pi05_action_processors(config, restored, post)


def test_episode_records_only_executed_commands_and_resets(monkeypatch):
    import deploy_piper_vlaa as deploy

    monkeypatch.setattr(deploy, "build_observation", lambda *args: {})
    holds = []
    monkeypatch.setattr(
        deploy, "_hold_position", lambda robot, action, *args: holds.append(action["gripper.pos"])
    )
    seen = []
    calls = []
    robot = SimpleNamespace(
        action_features=NAMES,
        get_observation=lambda: dict.fromkeys(NAMES, 0.7),
        send_action=lambda action: calls.append(action),
    )
    worker = SimpleNamespace(
        _policy_cfg=SimpleNamespace(piper_gripper_history_steps=20),
        trigger=lambda obs: seen.append(obs[HISTORY_KEY].copy()),
        next_chunk=lambda **kwargs: ([dict.fromkeys(NAMES, i / 20) for i in range(8)], 1.0),
    )
    arguments = (robot, worker, {}, 10, 1e6, 1, 1, {}, "rgb", 5)
    deploy.run_episode(*arguments)
    assert len(calls) == 10
    np.testing.assert_allclose(seen[0], [0.7] * 20)
    np.testing.assert_allclose(seen[1], [0.7] * 12 + [i / 20 for i in range(8)])
    np.testing.assert_allclose(holds, [0.7, 0.35])
    deploy.run_episode(*arguments)
    np.testing.assert_array_equal(seen[2], seen[0])


def test_failed_send_does_not_enter_history(monkeypatch):
    import deploy_piper_vlaa as deploy
    from lerobot.utils import piper_gripper_history as history_module

    buffer = ExecutedGripperHistory(0.7)
    monkeypatch.setattr(history_module, "ExecutedGripperHistory", lambda initial: buffer)
    monkeypatch.setattr(deploy, "build_observation", lambda *args: {})
    monkeypatch.setattr(deploy, "_hold_position", lambda *args: None)

    def fail_send(action):
        raise RuntimeError("simulated send failure")

    robot = SimpleNamespace(
        action_features=NAMES, get_observation=lambda: dict.fromkeys(NAMES, 0.7), send_action=fail_send
    )
    worker = SimpleNamespace(
        _policy_cfg=SimpleNamespace(piper_gripper_history_steps=20),
        trigger=lambda obs: None,
        next_chunk=lambda **kwargs: ([dict.fromkeys(NAMES, 0.1)], 1),
    )
    with pytest.raises(RuntimeError, match="send failure"):
        deploy.run_episode(robot, worker, {}, 1, 1e6, 1, 1, {}, "rgb", 5)
    np.testing.assert_allclose(buffer.observation(), [0.7] * 20)
