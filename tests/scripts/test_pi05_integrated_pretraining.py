#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig, PeftConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets import EpisodeAwareSampler  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.scripts import lerobot_train as train_module  # noqa: E402


class _FakeAccelerator:
    distributed_type = object()

    def __init__(self):
        self.wait_calls = 0
        self.free_memory_calls = 0
        self.end_training_calls = 0

    def wait_for_everyone(self):
        self.wait_calls += 1

    def free_memory(self):
        self.free_memory_calls += 1

    def end_training(self):
        self.end_training_calls += 1


class _PartiallyVisibleAccelerator(_FakeAccelerator):
    num_processes = 2
    device = torch.device("cpu")

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value.new_tensor(1)


def _make_flow_config(tmp_path: Path, **policy_kwargs) -> TrainPipelineConfig:
    policy = PI05Config(
        device="cpu",
        push_to_hub=False,
        optimizer_lr=7.5e-5,
        optimizer_betas=(0.8, 0.9),
        optimizer_eps=1e-7,
        optimizer_weight_decay=0.2,
        scheduler_warmup_steps=17,
        scheduler_decay_steps=23,
        scheduler_decay_lr=9e-6,
        time_sampling_offset=0.25,
        **policy_kwargs,
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/dataset"),
        policy=policy,
        output_dir=tmp_path / "flow",
        job_name="pi05_test",
        steps=123,
    )


def test_integrated_pretraining_config_is_isolated_and_uses_fixed_recipe(tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.validate()

    pretrain_cfg = train_module._make_pi05_next_action_pretraining_config(cfg)

    assert cfg.policy.training_stage == "flow"
    assert cfg.steps == 123
    assert cfg.output_dir == tmp_path / "flow"
    assert cfg.optimizer.lr == pytest.approx(7.5e-5)
    assert cfg.policy.drop_n_last_frames == 0

    assert pretrain_cfg.policy.training_stage == "next_action"
    assert pretrain_cfg.policy.next_action_pretrain_steps == 0
    assert not pretrain_cfg.policy.next_action_pretraining_active
    assert pretrain_cfg.policy.drop_n_last_frames == 25
    assert not pretrain_cfg.cabo_active
    assert pretrain_cfg.policy.time_sampling_offset == pytest.approx(0.001)
    assert pretrain_cfg.steps == 3_000
    assert pretrain_cfg.output_dir == tmp_path / "flow_next_action_pretrain"
    assert pretrain_cfg.save_checkpoint
    assert pretrain_cfg.save_freq == 3_000
    assert not pretrain_cfg.save_checkpoint_to_hub
    assert not pretrain_cfg.policy.push_to_hub
    assert not pretrain_cfg.wandb.enable
    assert pretrain_cfg.env is None
    assert pretrain_cfg.env_eval_freq == 0
    assert pretrain_cfg.sample_weighting is None
    assert pretrain_cfg.peft is None
    assert pretrain_cfg.job.target == "local"

    assert pretrain_cfg.optimizer is not cfg.optimizer
    assert pretrain_cfg.optimizer.lr == pytest.approx(2.5e-4)
    assert pretrain_cfg.optimizer.betas == (0.9, 0.95)
    assert pretrain_cfg.optimizer.eps == pytest.approx(1e-8)
    assert pretrain_cfg.optimizer.weight_decay == pytest.approx(0.01)
    assert pretrain_cfg.optimizer.grad_clip_norm == 0.0
    assert pretrain_cfg.scheduler.peak_lr == pytest.approx(2.5e-4)
    assert pretrain_cfg.scheduler.decay_lr == pytest.approx(2.5e-5)
    assert pretrain_cfg.scheduler.num_warmup_steps == 1_000
    assert pretrain_cfg.scheduler.num_decay_steps == 30_000

    assert train_module._pi05_next_action_pretrained_model_dir(pretrain_cfg) == (
        tmp_path / "flow_next_action_pretrain" / "checkpoints" / "003000" / "pretrained_model"
    )


def test_next_action_sampler_keeps_only_anchors_with_a_future_target(tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.validate()
    pretrain_cfg = train_module._make_pi05_next_action_pretraining_config(cfg)

    sampler = EpisodeAwareSampler(
        dataset_from_indices=[0, 25, 51],
        dataset_to_indices=[25, 51, 78],
        drop_n_last_frames=pretrain_cfg.policy.drop_n_last_frames,
    )

    assert sampler.indices == [25, 51, 52]
    assert all(
        anchor + pretrain_cfg.policy.next_action_context_steps < episode_end
        for anchor, episode_end in ((25, 51), (51, 78), (52, 78))
    )


def test_one_command_runs_next_action_then_flow_with_fresh_stage_configs(monkeypatch, tmp_path):
    cfg = _make_flow_config(tmp_path)
    accelerator = _FakeAccelerator()
    stages = []
    validate_calls = []
    original_validate = TrainPipelineConfig.validate

    def tracked_validate(stage_cfg):
        validate_calls.append(stage_cfg)
        return original_validate(stage_cfg)

    def fake_train_single_stage(stage_cfg, stage_accelerator):
        stages.append((stage_cfg, stage_accelerator))
        if stage_cfg.policy.training_stage == "next_action":
            train_module._pi05_next_action_pretrained_model_dir(stage_cfg).mkdir(
                parents=True,
                exist_ok=True,
            )

    monkeypatch.setattr(TrainPipelineConfig, "validate", tracked_validate)
    monkeypatch.setattr(train_module, "_train_single_stage", fake_train_single_stage)

    train_module.train(cfg, accelerator=accelerator)

    assert validate_calls == [cfg]
    assert [stage_cfg.policy.training_stage for stage_cfg, _ in stages] == ["next_action", "flow"]
    assert all(stage_accelerator is accelerator for _, stage_accelerator in stages)
    assert stages[0][0] is not cfg
    assert not stages[0][0].cabo_active
    assert stages[0][0].optimizer is not cfg.optimizer
    assert stages[1][0] is cfg
    assert stages[1][0].cabo_active
    expected_model_dir = (
        tmp_path / "flow_next_action_pretrain" / "checkpoints" / "003000" / "pretrained_model"
    )
    assert cfg.policy.pretrained_path == expected_model_dir
    assert cfg.policy.pretrained_revision is None
    assert not cfg.output_dir.exists()
    assert accelerator.wait_calls == 1
    assert accelerator.free_memory_calls == 1
    assert accelerator.end_training_calls == 1


def test_zero_pretraining_steps_runs_only_flow(monkeypatch, tmp_path):
    cfg = _make_flow_config(tmp_path, next_action_pretrain_steps=0)
    accelerator = _FakeAccelerator()
    stages = []

    monkeypatch.setattr(
        train_module,
        "_train_single_stage",
        lambda stage_cfg, stage_accelerator: stages.append((stage_cfg, stage_accelerator)),
    )

    train_module.train(cfg, accelerator=accelerator)

    assert [stage_cfg.policy.training_stage for stage_cfg, _ in stages] == ["flow"]
    assert accelerator.free_memory_calls == 0
    assert accelerator.end_training_calls == 1


def test_distributed_pretraining_requires_checkpoint_visible_to_every_rank(monkeypatch, tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.validate()
    accelerator = _PartiallyVisibleAccelerator()

    def fake_train_single_stage(stage_cfg, stage_accelerator):
        assert stage_accelerator is accelerator
        train_module._pi05_next_action_pretrained_model_dir(stage_cfg).mkdir(
            parents=True,
            exist_ok=True,
        )

    monkeypatch.setattr(train_module, "_train_single_stage", fake_train_single_stage)

    with pytest.raises(RuntimeError, match="filesystem shared by all ranks"):
        train_module._run_pi05_next_action_pretraining(cfg, accelerator)

    assert accelerator.free_memory_calls == 0


def test_resume_never_activates_integrated_pretraining(tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.resume = True

    assert not train_module._pi05_next_action_pretraining_active(cfg)


@pytest.mark.parametrize("peft_mode", ["source_adapter", "formal_adapter"])
def test_integrated_pretraining_rejects_peft(tmp_path, peft_mode):
    cfg = _make_flow_config(tmp_path, use_peft=True)
    if peft_mode == "formal_adapter":
        cfg.policy.use_peft = False
        cfg.peft = PeftConfig()

    with pytest.raises(ValueError, match="does not support PEFT"):
        train_module._make_pi05_next_action_pretraining_config(cfg)


def test_integrated_pretraining_rejects_streaming_dataset(tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.dataset.streaming = True

    with pytest.raises(ValueError, match="requires a map-style dataset"):
        train_module._make_pi05_next_action_pretraining_config(cfg)


def test_direct_next_action_training_rejects_streaming_dataset(tmp_path):
    cfg = _make_flow_config(
        tmp_path,
        training_stage="next_action",
        next_action_pretrain_steps=0,
    )
    cfg.dataset.streaming = True

    with pytest.raises(ValueError, match="requires a map-style dataset"):
        train_module._train_single_stage(cfg, _FakeAccelerator())


def test_reward_model_training_never_activates_pi05_pretraining(tmp_path):
    cfg = _make_flow_config(tmp_path)
    cfg.reward_model = object()

    assert not train_module._pi05_next_action_pretraining_active(cfg)


def test_integrated_pretraining_revalidates_25_plus_25_split(tmp_path):
    cfg = _make_flow_config(tmp_path, chunk_size=40, n_action_steps=40)

    with pytest.raises(ValueError, match="must equal chunk_size"):
        train_module._make_pi05_next_action_pretraining_config(cfg)
