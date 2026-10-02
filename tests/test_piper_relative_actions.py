"""CPU checks for Piper representation, selected-row stats and checkpoint inference parity."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.piper_action_representation import fit_piper_train_normalization
from lerobot.utils.piper_training_eval import PIPER_NAMES, PiperActionEvaluator


@pytest.fixture
def tokenizer(monkeypatch):
    class Tokenizer:
        padding_side = "right"

        def __call__(self, text, **kwargs):
            shape = (len(text), 4)
            return {"input_ids": torch.ones(shape, dtype=torch.long), "attention_mask": torch.ones(shape)}

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer()
    )


def fixture_dataset():
    features = {
        "action": {"shape": (7,), "names": PIPER_NAMES[::-1]},
        "observation.state": {"shape": (7,), "names": PIPER_NAMES},
    }
    state = np.array([np.arange(7) * 0.1 + t for t in (0, 1, 2, 10000, 10, 11)])
    state[:, -1] = [0.1, 0.3, 0.8, 9999, 0.2, 0.6]
    data = Dataset.from_dict(
        {
            "action": state[:, ::-1].tolist(),
            "observation.state": state.tolist(),
            "episode_index": [2, 2, 2, 99, 7, 7],
            "frame_index": [0, 1, 2, 0, 0, 1],
        }
    )
    # Retain a HF indices mapping with a held-out outlier in the underlying table.
    return SimpleNamespace(
        hf_dataset=data.select([4, 5, 0, 1, 2]),
        episodes=[2, 7],
        meta=SimpleNamespace(features=features, stats={"action": {"q99": [99999] * 7}}),
    )


def make_config(relative=True):
    config = PI05Config(
        device="cpu",
        chunk_size=3,
        n_action_steps=2,
        cabo_enabled=False,
        next_action_pretrain_steps=0,
        next_action_bridge_steps=0,
        piper_train_normalization=True,
        use_relative_actions=relative,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        normalization_mapping={
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
            "VISUAL": NormalizationMode.IDENTITY,
        },
    )
    config.set_dataset_feature_metadata(fixture_dataset().meta.features)
    return config


@pytest.mark.parametrize("relative", [False, True])
def test_stats_use_only_selected_episodes_and_unpadded_pairs(relative):
    dataset, config = fixture_dataset(), make_config(relative)
    original = deepcopy(dataset.meta.stats)
    stats, report = fit_piper_train_normalization(dataset, config)
    assert dataset.meta.stats == original
    assert report["training_episodes"] == [2, 7]
    assert report["valid_action_pairs"] == 9  # 3+2+1 plus 2+1, no repeated padded tails
    assert report["training_frames"] == 5
    assert max(stats["observation.state"]["max"]) < 12
    assert stats["action"]["max"][0] == 0.8  # reversed order: gripper kept absolute
    if relative:
        np.testing.assert_allclose(stats["action"]["max"][1:], 2)
        np.testing.assert_allclose(stats["action"]["min"][1:], 0)
        np.testing.assert_allclose(stats["action"]["mean"][1:], 5 / 9)
    else:
        np.testing.assert_allclose(stats["action"]["max"][1:], (np.arange(6) * 0.1 + 11)[::-1])
    # Feature names are validated, rather than inferring a seven-dimensional order.
    dataset.meta.features["action"]["names"] = None
    with pytest.raises(ValueError, match="explicitly named"):
        fit_piper_train_normalization(dataset, config)


@pytest.mark.parametrize("relative", [False, True])
def test_pipeline_roundtrip_saved_load_and_reset(tokenizer, tmp_path, relative):
    config = make_config(relative)
    stats, _ = fit_piper_train_normalization(fixture_dataset(), config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    state = torch.tensor([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.3]])
    actions = state.flip(-1).unsqueeze(1).repeat(1, 3, 1)
    actions[:, :, 1:] += torch.tensor([0, 0.5, 1.0]).reshape(1, 3, 1)
    actions[:, :, 0] = torch.tensor([0.3, 0.5, 0.7])
    raw = {"observation.state": state, "action": actions, "task": ["pick apple"]}
    normalized = pre(deepcopy(raw))["action"]
    torch.testing.assert_close(post(normalized), actions)
    config.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_config = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    loaded_pre, loaded_post = make_pre_post_processors(loaded_config, pretrained_path=tmp_path)
    # Inference has no labels. Every output uses the SAME observed state, not accumulated deltas.
    loaded_pre({"observation.state": state, "task": ["pick apple"]})
    actual = torch.stack([loaded_post(normalized[:, h]) for h in range(3)], dim=1)
    torch.testing.assert_close(actual, actions)
    if relative:
        new_state = state.clone()
        new_state[:, :6] += 2
        loaded_pre({"observation.state": new_state, "task": ["pick apple"]})
        expected = actions.clone()
        expected[:, :, 1:] += 2
        torch.testing.assert_close(loaded_post(normalized), expected)
        loaded_pre.reset()
        with pytest.raises(RuntimeError, match="no state"):
            loaded_post(normalized)
        loaded_pre({"observation.state": state, "task": ["pick apple"]})
        with pytest.raises(ValueError, match="every preprocessing"):
            loaded_pre({"observation.images.camera": torch.zeros(1, 3, 4, 4), "task": ["pick apple"]})
        with pytest.raises(RuntimeError, match="no state"):
            loaded_post(normalized)
    # Old absolute processors cannot masquerade as relative checkpoints, and vice versa.
    loaded_config.use_relative_actions = not relative
    with pytest.raises(ValueError, match="PI05"):
        make_pre_post_processors(loaded_config, pretrained_path=tmp_path)


def test_evaluation_copies_keep_shared_reference_and_do_not_change_training_state(tokenizer, tmp_path):
    from tests.test_piper_training_eval import FakeDataset, Policy

    dataset = FakeDataset()
    config = make_config()
    config.action_feature_names = config.state_feature_names = PIPER_NAMES.copy()
    config.input_features.update(
        {
            f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4))
            for name in ("cam_high", "cam_wrist")
        }
    )
    stats = {key: {"q01": np.full(7, -1), "q99": np.full(7, 1)} for key in ("action", "observation.state")}
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    pre({"observation.state": torch.full((1, 7), 42.0), "task": ["training"]})
    options = SimpleNamespace(seed=10, samples=2, execution_steps=2, moving_threshold_deg=1)
    evaluator = PiperActionEvaluator(
        {"train": dataset}, config, pre, post, options, tmp_path, torch.device("cpu")
    )
    relative = next(s for s in evaluator.preprocessor.steps if isinstance(s, RelativeActionsProcessorStep))
    absolute = next(s for s in evaluator.postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert absolute.relative_step is relative
    # A full fixed-sample prediction calls the cloned preprocessor before inverse conversion.
    scores = evaluator.evaluate(Policy(config), 0, nullcontext)
    assert np.isfinite(scores["action_train/h2/joint_mae_deg"])
    original = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    torch.testing.assert_close(original.get_cached_state(), torch.full((1, 7), 42.0))
    assert relative.get_cached_state().eq(0).all()


def test_config_refuses_relative_with_absolute_stats_or_changed_coordinate_order():
    with pytest.raises(ValueError, match="piper_train_normalization"):
        PI05Config(use_relative_actions=True)
    config = make_config()
    features = deepcopy(fixture_dataset().meta.features)
    features["action"]["names"] = PIPER_NAMES.copy()
    with pytest.raises(ValueError, match="refusing to reorder"):
        config.set_dataset_feature_metadata(features)


@pytest.mark.parametrize("resume", [False, True])
def test_trainer_builds_fresh_target_processors_and_resume_keeps_saved_stats(
    tokenizer, tmp_path, monkeypatch, resume
):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as train

    dataset, config = fixture_dataset(), make_config()
    config.pretrained_path = tmp_path / "base"
    expected_stats, _ = fit_piper_train_normalization(dataset, config)
    if resume:
        config.pretrained_path.mkdir()
        pre, post = make_pre_post_processors(config, dataset_stats=expected_stats)
        pre.save_pretrained(config.pretrained_path)
        post.save_pretrained(config.pretrained_path)

        # Recomputing stats during resume is forbidden, even if the current metadata is different.
        def no_refit(*args):
            pytest.fail("Resume must load normalization from checkpoint")

        monkeypatch.setattr(
            "lerobot.utils.piper_action_representation.fit_piper_train_normalization", no_refit
        )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/piper"),
        policy=config,
        output_dir=tmp_path / "run",
        resume=resume,
        seed=None,
    )
    cfg.wandb.enable = False
    accelerator = SimpleNamespace(
        is_main_process=True, device=torch.device("cpu"), wait_for_everyone=lambda: None
    )
    monkeypatch.setattr(train, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(train, "make_train_eval_datasets", lambda cfg: (dataset, None))
    monkeypatch.setattr(train, "make_policy", lambda **kwargs: SimpleNamespace(config=config))
    captured = []

    def processors(*args, **kwargs):
        result = make_pre_post_processors(*args, **kwargs)
        captured.append(result)
        return result

    class StopBeforeOptimizerError(Exception):
        pass

    def stop(*args):
        raise StopBeforeOptimizerError

    monkeypatch.setattr(train, "make_pre_post_processors", processors)
    monkeypatch.setattr(train, "make_optimizer_and_scheduler", stop)
    with pytest.raises(StopBeforeOptimizerError):
        train._train_single_stage(cfg, accelerator)
    from lerobot.processor import NormalizerProcessorStep

    pre, post = captured[0]
    norm = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    np.testing.assert_allclose(norm.stats["action"]["q99"], expected_stats["action"]["q99"])
    assert not resume or not (cfg.output_dir / "piper_normalization.json").exists()
    assert any(isinstance(s, AbsoluteActionsProcessorStep) for s in post.steps)


def test_streaming_is_rejected_by_train_validation(tmp_path):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/piper", streaming=True),
        policy=make_config(),
        output_dir=tmp_path / "run",
    )
    with pytest.raises(ValueError, match="non-streaming"):
        cfg.validate()
