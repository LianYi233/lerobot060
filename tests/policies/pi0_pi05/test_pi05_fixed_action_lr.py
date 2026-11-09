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

"""CPU-only contracts for the fixed action-prompt LR baseline against CPC/CABO."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.optim.cabo import (  # noqa: E402
    CABO_ACTION_PROMPT_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_PROMPT_GROUP,
    temporary_optimizer_group_lr_scales,
)
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402


class _NoCollectiveAccelerator:
    num_processes = 2

    def reduce(self, *_args, **_kwargs):
        raise AssertionError("A fixed LR multiplier must not estimate or synchronize candidate updates")


class _TinyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_prompt_tokens = nn.Embedding(1, 2, dtype=torch.float64)
        self.prompt_tokens = nn.Embedding(1, 2, dtype=torch.float64)
        self.backbone = nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            self.vlm_prompt_tokens.weight.fill_(10.0)
            self.prompt_tokens.weight.fill_(1.0)

    def _freeze_vlm(self):
        self.backbone.requires_grad_(False)


class _TinyFixedLRPolicy(nn.Module):
    # Deliberately do not bind the new helper: class-qualified calls retain compatibility
    # with existing lightweight policies whose configs do not carry the new field.
    _cabo_parameter_groups = PI05Policy._cabo_parameter_groups
    get_optim_params = PI05Policy.get_optim_params
    compute_optimizer_step_control = PI05Policy.compute_optimizer_step_control
    validate_optimizer_step_control = PI05Policy.validate_optimizer_step_control

    def __init__(self, *, scale=0.25, stage="flow", bridge_active=False):
        super().__init__()
        self.model = _TinyPromptModel()
        self.config = SimpleNamespace(
            cabo_enabled=False,
            action_prompt_lr_scale=scale,
            training_stage=stage,
            num_prompt_tokens=1,
        )
        self._stage1_bridge_active = bridge_active

    @property
    def prompts(self):
        return (self.model.vlm_prompt_tokens.weight, self.model.prompt_tokens.weight)


def _optimizer(policy, *, weight_decay=0.0, betas=(0.9, 0.95)):
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=0.1, betas=betas, eps=1e-8, weight_decay=weight_decay
    )
    policy.validate_optimizer_step_control(optimizer)
    return optimizer


def _gradients(policy, value=1.0):
    for parameter in policy.prompts:
        parameter.grad = torch.full_like(parameter, value)


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_fixed_action_lr_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="action_prompt_lr_scale must be finite and in"):
        PI05Config(device="cpu", cabo_enabled=False, action_prompt_lr_scale=scale)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
def test_fixed_action_lr_config_defaults_and_controller_property(scale):
    config = PI05Config(device="cpu", cabo_enabled=False, action_prompt_lr_scale=scale)
    assert config.action_prompt_lr_scale == scale
    assert config.cabo_active is False
    assert config.optimizer_step_control_active is (scale != 1.0)
    assert PI05Config(device="cpu").action_prompt_lr_scale == 1.0
    assert PI05Config(device="cpu").optimizer_step_control_active is True


def test_fixed_action_lr_rejects_cabo_and_single_prompt_bank():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PI05Config(device="cpu", cabo_enabled=True, action_prompt_lr_scale=0.5)
    for missing_bank in ("num_prompt_tokens", "num_vlm_prompt_tokens"):
        with pytest.raises(ValueError, match="Fixed action-prompt LR scaling requires"):
            PI05Config(device="cpu", cabo_enabled=False, action_prompt_lr_scale=0.5, **{missing_bank: 0})


def test_fixed_action_lr_keeps_named_disjoint_prompt_groups_and_cabo_adamw_mode():
    policy = _TinyFixedLRPolicy()
    optimizer = _optimizer(policy)

    assert [group[CABO_GROUP_NAME] for group in optimizer.param_groups] == [
        CABO_VLM_PROMPT_GROUP,
        CABO_ACTION_PROMPT_GROUP,
    ]
    for group, prompt in zip(optimizer.param_groups, policy.prompts, strict=True):
        assert len(group["params"]) == 1
        assert group["params"][0] is prompt
        assert group["lr"] == 0.1
        assert group["foreach"] is False
        assert group["fused"] is False
    assert all(not parameter.requires_grad for parameter in policy.model.backbone.parameters())


@pytest.mark.parametrize("stage,bridge_active", [("flow", False), ("next_action", True)])
def test_fixed_action_lr_scales_only_action_in_bridge_and_flow_without_touching_gradients(
    stage, bridge_active, monkeypatch
):
    policy = _TinyFixedLRPolicy(stage=stage, bridge_active=bridge_active)
    optimizer = _optimizer(policy)
    _gradients(policy)
    gradients_before = [parameter.grad.clone() for parameter in policy.prompts]

    def unexpected_candidate(*_args, **_kwargs):
        raise AssertionError("Fixed LR must not depend on the relative-update candidate estimator")

    monkeypatch.setattr(modeling_pi05, "adamw_group_relative_update_moments", unexpected_candidate)
    control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())

    assert control.group_scales == {CABO_ACTION_PROMPT_GROUP: 0.25}
    assert not control.skip_optimizer_step
    assert control.metrics["cabo/active"] == 0.0
    assert control.metrics["fixed_action_lr/active"] == 1.0
    assert control.metrics["fixed_action_lr/action_only_bypass"] == 0.0
    assert control.metrics["fixed_action_lr/effective_vlm_prompt_lr"] == 0.1
    assert control.metrics["fixed_action_lr/effective_action_prompt_lr"] == 0.025
    assert not optimizer.state
    for parameter, old_grad in zip(policy.prompts, gradients_before, strict=True):
        torch.testing.assert_close(parameter.grad, old_grad, rtol=0, atol=0)
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.1]


def test_fixed_action_lr_bypasses_sap_without_lowering_its_action_lr():
    policy = _TinyFixedLRPolicy(stage="next_action")
    optimizer = _optimizer(policy, weight_decay=0.1)
    reference = _TinyFixedLRPolicy(stage="next_action")
    reference_optimizer = _optimizer(reference, weight_decay=0.1)
    policy.prompts[1].grad = torch.ones_like(policy.prompts[1])
    reference.prompts[1].grad = torch.ones_like(reference.prompts[1])

    control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()
    reference_optimizer.step()

    assert not control.group_scales
    assert control.metrics["fixed_action_lr/active"] == 0.0
    assert control.metrics["fixed_action_lr/action_only_bypass"] == 1.0
    assert control.metrics["fixed_action_lr/configured_scale"] == 0.25
    assert control.metrics["fixed_action_lr/action_prompt_scale"] == 1.0
    assert control.metrics["fixed_action_lr/effective_action_prompt_lr"] == 0.1
    for parameter, expected in zip(policy.prompts, reference.prompts, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_fixed_action_lr_matches_independent_adamw_lr_groups_and_leaves_base_scheduler_unchanged():
    policy = _TinyFixedLRPolicy()
    optimizer = _optimizer(policy, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.8**step)
    reference = _TinyFixedLRPolicy()
    reference_optimizer = _optimizer(reference, weight_decay=0.1)

    for step, gradient in enumerate((1.0, -0.3, 0.7)):
        _gradients(policy, gradient)
        _gradients(reference, gradient)
        scheduled_lr = 0.1 * 0.8**step
        reference_optimizer.param_groups[0]["lr"] = scheduled_lr
        reference_optimizer.param_groups[1]["lr"] = scheduled_lr * 0.25
        control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())
        with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduled_lr)
            assert optimizer.param_groups[1]["lr"] == pytest.approx(scheduled_lr * 0.25)
            optimizer.step()
        reference_optimizer.step()
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([scheduled_lr] * 2)
        for parameter, expected in zip(policy.prompts, reference.prompts, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    optimizer.state[parameter][key], reference_optimizer.state[expected][key], rtol=0, atol=0
                )
        scheduler.step()
        assert scheduler.get_last_lr() == pytest.approx([0.1 * 0.8 ** (step + 1)] * 2)
        assert scheduler.base_lrs == [0.1, 0.1]


def test_fixed_action_lr_scales_actual_adamw_weight_decay_as_well_as_learning():
    policy = _TinyFixedLRPolicy()
    optimizer = _optimizer(policy, weight_decay=0.5)
    _gradients(policy, 0.0)
    before = [parameter.detach().clone() for parameter in policy.prompts]

    control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    torch.testing.assert_close(policy.prompts[0], before[0] * (1.0 - 0.1 * 0.5), rtol=0, atol=0)
    torch.testing.assert_close(policy.prompts[1], before[1] * (1.0 - 0.1 * 0.25 * 0.5), rtol=0, atol=0)


def test_fixed_action_lr_restores_scheduled_rates_even_if_optimizer_step_raises():
    policy = _TinyFixedLRPolicy()
    optimizer = _optimizer(policy)
    control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())

    with pytest.raises(RuntimeError, match="simulated step failure"):
        with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            assert optimizer.param_groups[1]["lr"] == 0.025
            raise RuntimeError("simulated step failure")
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.1]


@pytest.mark.parametrize("legacy_config", [False, True])
def test_default_scale_keeps_existing_uncontrolled_no_cabo_behavior(legacy_config):
    policy = _TinyFixedLRPolicy(scale=1.0)
    if legacy_config:
        del policy.config.action_prompt_lr_scale
    parameters = policy.get_optim_params()
    assert all(isinstance(parameter, nn.Parameter) for parameter in parameters)
    optimizer = torch.optim.SGD(parameters, lr=0.1)

    policy.validate_optimizer_step_control(optimizer)
    control = policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())

    assert not control.group_scales
    assert not control.metrics
    assert not control.skip_optimizer_step


def test_runtime_fixed_action_lr_rejects_cabo_conflict_and_invalid_overrides():
    policy = _TinyFixedLRPolicy()
    optimizer = _optimizer(policy)
    policy.config.cabo_enabled = True
    with pytest.raises(ValueError, match="requires cabo_enabled=False"):
        policy.compute_optimizer_step_control({}, optimizer, _NoCollectiveAccelerator())
    policy.config.cabo_enabled = False
    policy.config.action_prompt_lr_scale = 0.0
    with pytest.raises(ValueError, match="action_prompt_lr_scale"):
        policy.validate_optimizer_step_control(optimizer)


def test_fixed_action_lr_requires_the_same_adamw_contract_as_cabo():
    policy = _TinyFixedLRPolicy()
    parameters = policy.get_optim_params()
    with pytest.raises(TypeError, match="AdamW"):
        policy.validate_optimizer_step_control(torch.optim.SGD(parameters, lr=0.1))
    with pytest.raises(ValueError, match="two named"):
        policy.validate_optimizer_step_control(torch.optim.AdamW(list(policy.prompts), lr=0.1))
    with pytest.raises(RuntimeError, match="foreach"):
        policy.validate_optimizer_step_control(torch.optim.AdamW(parameters, lr=0.1, foreach=True))
