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

from types import SimpleNamespace

import draccus
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402


def _make_module() -> nn.Module:
    return nn.Linear(2, 2, bias=False)


def _make_policy_with_gradients(
    *, action_gradient: float, vlm_gradient: float, action_head_grad_clip_ratio: float = 10.0
) -> tuple[SimpleNamespace, list[nn.Parameter], list[nn.Parameter]]:
    paligemma_with_expert = SimpleNamespace(
        paligemma=_make_module(),
        gemma_expert=_make_module(),
    )
    action_modules = [
        paligemma_with_expert.gemma_expert,
        _make_module(),
        _make_module(),
        _make_module(),
        _make_module(),
    ]
    model = SimpleNamespace(
        paligemma_with_expert=paligemma_with_expert,
        action_in_proj=action_modules[1],
        action_out_proj=action_modules[2],
        time_mlp_in=action_modules[3],
        time_mlp_out=action_modules[4],
    )
    policy = SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            clip_action_head_by_vlm=True,
            action_head_grad_clip_ratio=action_head_grad_clip_ratio,
        ),
    )

    action_parameters = [parameter for module in action_modules for parameter in module.parameters()]
    vlm_parameters = list(paligemma_with_expert.paligemma.parameters())
    for parameter in action_parameters:
        parameter.grad = torch.full_like(parameter, action_gradient)
    for parameter in vlm_parameters:
        parameter.grad = torch.full_like(parameter, vlm_gradient)

    return policy, action_parameters, vlm_parameters


def _gradient_rms(parameters: list[nn.Parameter]) -> float:
    gradients = torch.cat([parameter.grad.detach().float().flatten() for parameter in parameters])
    return gradients.square().mean().sqrt().item()


def test_pi05_uses_action_only_gradient_clipping_by_default():
    config = PI05Config()
    optimizer_config = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()
    parameters = [nn.Parameter(torch.zeros(())), nn.Parameter(torch.ones(()))]
    optimizer = optimizer_config.build(parameters)

    assert config.optimizer_grad_clip_norm == 0.0
    assert optimizer_config.grad_clip_norm == 0.0
    assert optimizer_config.lr == pytest.approx(2.5e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(parameter) for parameter in parameters
    ]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.9, 0.95))
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1e-8)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert scheduler.peak_lr == pytest.approx(2.5e-4)
    assert scheduler.decay_lr == pytest.approx(2.5e-5)
    assert scheduler.num_warmup_steps == 1_000
    assert scheduler.num_decay_steps == 30_000
    assert config.clip_action_head_by_vlm
    assert config.action_head_grad_clip_ratio == pytest.approx(10.0)
    assert not config.cabo_enabled
    assert config.cabo_control_mode == "budget"
    assert config.cabo_balance_max_scale == pytest.approx(2.0)
    assert config.cabo_action_drift_ratio == pytest.approx(0.1)
    assert config.cabo_probe_interval == 8
    assert config.cabo_probe_batch_size == 1
    assert config.cabo_num_projections == 4
    assert config.cabo_base_action_scale == pytest.approx(0.1)
    assert config.cabo_negative_cross_discount == pytest.approx(0.5)


@pytest.mark.parametrize(
    "action_head_grad_clip_ratio",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_pi05_rejects_invalid_action_head_grad_clip_ratio(action_head_grad_clip_ratio: float):
    with pytest.raises(ValueError, match="action_head_grad_clip_ratio"):
        PI05Config(action_head_grad_clip_ratio=action_head_grad_clip_ratio)


@pytest.mark.parametrize("cabo_action_drift_ratio", [0.0, -1.0, 1.1])
def test_pi05_rejects_invalid_cabo_action_drift_ratio(cabo_action_drift_ratio: float):
    with pytest.raises(ValueError, match="cabo_action_drift_ratio"):
        PI05Config(cabo_action_drift_ratio=cabo_action_drift_ratio)


def test_pi05_rejects_invalid_cabo_control_mode():
    with pytest.raises(ValueError, match="cabo_control_mode"):
        PI05Config(cabo_control_mode="unknown")


def test_pi05_cabo_control_mode_decodes_from_nested_cli_argument():
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=user/repo",
            "--policy.type=pi05",
            "--policy.cabo_control_mode=balance",
        ],
    )

    assert isinstance(config.policy, PI05Config)
    assert config.policy.cabo_control_mode == "balance"


@pytest.mark.parametrize(
    "cabo_balance_max_scale",
    [0.0, 0.5, float("nan"), float("inf"), float("-inf")],
)
def test_pi05_rejects_invalid_cabo_balance_max_scale(cabo_balance_max_scale: float):
    with pytest.raises(ValueError, match="cabo_balance_max_scale"):
        PI05Config(cabo_balance_max_scale=cabo_balance_max_scale)


@pytest.mark.parametrize("cabo_num_projections", [0, 1, -1])
def test_pi05_rejects_too_few_cabo_projections(cabo_num_projections: int):
    with pytest.raises(ValueError, match="cabo_num_projections"):
        PI05Config(cabo_num_projections=cabo_num_projections)


@pytest.mark.parametrize("cabo_base_action_scale", [-0.1, 1.1])
def test_pi05_rejects_invalid_cabo_base_action_scale(cabo_base_action_scale: float):
    with pytest.raises(ValueError, match="cabo_base_action_scale"):
        PI05Config(cabo_base_action_scale=cabo_base_action_scale)


@pytest.mark.parametrize("cabo_negative_cross_discount", [-0.1, 1.1])
def test_pi05_rejects_invalid_cabo_negative_cross_discount(cabo_negative_cross_discount: float):
    with pytest.raises(ValueError, match="cabo_negative_cross_discount"):
        PI05Config(cabo_negative_cross_discount=cabo_negative_cross_discount)


def test_pi05_rejects_cabo_with_expert_only_training():
    with pytest.raises(ValueError, match="train_expert_only"):
        PI05Config(cabo_enabled=True, train_expert_only=True)


def test_pi05_cabo_requires_policy_training_preset(tmp_path):
    policy_config = PI05Config(cabo_enabled=True, push_to_hub=False)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy_config,
        output_dir=tmp_path / "new-output",
        use_policy_training_preset=False,
        optimizer=policy_config.get_optimizer_preset(),
        scheduler=policy_config.get_scheduler_preset(),
    )

    with pytest.raises(ValueError, match="use_policy_training_preset"):
        config.validate()


def test_pi05_cabo_disables_gradient_clipping_hook():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
    )
    policy.config.cabo_enabled = True
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    assert metrics["cabo/gradient_clip_disabled"] == 1.0
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_leaves_action_gradient_below_ten_times_vlm_rms_unchanged():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=6.0,
        vlm_gradient=1.0,
    )
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    assert metrics["action_head_grad_rms_before_clip"] == pytest.approx(6.0)
    assert metrics["vlm_grad_rms"] == pytest.approx(1.0)
    assert metrics["action_head_clip_threshold_rms"] == pytest.approx(10.0)
    assert metrics["action_head_grad_rms_after_clip"] == pytest.approx(6.0)
    assert metrics["action_head_clip_scale"] == pytest.approx(1.0)
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_clips_action_gradient_to_ten_times_vlm_rms_without_modifying_vlm():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=20.0,
        vlm_gradient=1.0,
    )
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 1.0
    assert metrics["action_head_grad_rms_before_clip"] == pytest.approx(20.0)
    assert metrics["vlm_grad_rms"] == pytest.approx(1.0)
    assert metrics["action_head_clip_threshold_rms"] == pytest.approx(10.0)
    assert _gradient_rms(action_parameters) == pytest.approx(10.0, abs=1e-6)
    assert metrics["action_head_grad_rms_after_clip"] == pytest.approx(10.0, abs=1e-6)
    assert metrics["action_head_clip_scale"] == pytest.approx(0.5, abs=1e-6)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
