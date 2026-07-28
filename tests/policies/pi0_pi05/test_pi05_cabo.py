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

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.optim.cabo import (  # noqa: E402
    CABO_ACTION_EXPERT_GROUP,
    CABO_ACTION_PROJECTION_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_GROUP,
    temporary_optimizer_group_lr_scales,
)
from lerobot.policies.pi05 import PI05Policy  # noqa: E402


class _FakeAccelerator:
    num_processes = 1

    def reduce(self, value, reduction="mean"):
        raise AssertionError(f"single-process CABO must not reduce moments ({reduction=}, {value=})")


class _ReducingFakeAccelerator:
    num_processes = 2

    def __init__(self):
        self.reduce_calls = 0

    def reduce(self, value, reduction="mean"):
        assert reduction == "sum"
        self.reduce_calls += 1
        return value * self.num_processes


class _TinyCABOPolicy(nn.Module):
    compute_optimizer_step_control = PI05Policy.compute_optimizer_step_control
    validate_optimizer_step_control = PI05Policy.validate_optimizer_step_control

    def __init__(self, *, warmup_steps: int = 0):
        super().__init__()
        self.vlm_weight = nn.Parameter(torch.tensor([10.0], dtype=torch.float64))
        self.expert_weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.projection_weight = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.config = SimpleNamespace(
            cabo_enabled=True,
            cabo_expert_update_ratio=2.0,
            cabo_projection_update_ratio=5.0,
            cabo_vlm_update_ema_decay=0.0,
            cabo_update_warmup_steps=warmup_steps,
            cabo_vlm_update_floor_ratio=0.0,
        )

    def _cabo_parameter_groups(self):
        return [self.vlm_weight], [self.expert_weight], [self.projection_weight]


def _make_optimizer(policy: _TinyCABOPolicy, *, weight_decay: float = 0.0):
    return torch.optim.AdamW(
        [
            {
                "params": [policy.vlm_weight],
                CABO_GROUP_NAME: CABO_VLM_GROUP,
            },
            {
                "params": [policy.expert_weight],
                CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP,
            },
            {
                "params": [policy.projection_weight],
                CABO_GROUP_NAME: CABO_ACTION_PROJECTION_GROUP,
            },
        ],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=weight_decay,
        foreach=False,
    )


def _set_unit_gradients(policy: _TinyCABOPolicy) -> None:
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)


def test_pi05_cabo_scales_action_groups_from_relative_adamw_updates_without_touching_gradients():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    gradients_before = [parameter.grad.clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    # The first AdamW learning delta is 0.1 for every scalar. Relative rates are
    # therefore VLM=0.01, expert=0.1, projection=0.05.
    assert control.group_scales == pytest.approx(
        {
            CABO_ACTION_EXPERT_GROUP: 0.2,
            CABO_ACTION_PROJECTION_GROUP: 1.0,
        }
    )
    assert control.metrics["cabo/vlm_relative_update_rate"] == pytest.approx(0.01)
    assert control.metrics["cabo/expert_relative_update_rate"] == pytest.approx(0.1)
    assert control.metrics["cabo/projection_relative_update_rate"] == pytest.approx(0.05)
    assert not control.skip_optimizer_step
    for parameter, gradient_before in zip(policy.parameters(), gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_cabo_action_scales_limit_the_real_optimizer_step_and_leave_vlm_full():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    with temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
        optimizer.step()

    deltas = [parameter.detach() - old for parameter, old in zip(policy.parameters(), before, strict=True)]
    assert deltas[0].item() == pytest.approx(-0.1)
    assert deltas[1].item() == pytest.approx(-0.02)
    assert deltas[2].item() == pytest.approx(-0.1)


def test_pi05_cabo_warmup_collects_reference_without_limiting_action_groups():
    policy = _TinyCABOPolicy(warmup_steps=2)
    optimizer = _make_optimizer(policy)

    for expected_step in (1, 2):
        _set_unit_gradients(policy)
        control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
        assert control.group_scales == pytest.approx(
            {
                CABO_ACTION_EXPERT_GROUP: 1.0,
                CABO_ACTION_PROJECTION_GROUP: 1.0,
            }
        )
        assert control.metrics["cabo/warmup_active"] == 1.0
        assert get_vlm_group(optimizer)["cabo_step"] == expected_step

    _set_unit_gradients(policy)
    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())
    assert control.metrics["cabo/warmup_active"] == 0.0
    assert control.group_scales[CABO_ACTION_EXPERT_GROUP] == pytest.approx(0.2)


def get_vlm_group(optimizer):
    return next(group for group in optimizer.param_groups if group.get(CABO_GROUP_NAME) == CABO_VLM_GROUP)


def test_pi05_cabo_excludes_weight_decay_from_measured_learning_rate():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy, weight_decay=0.5)
    for parameter in policy.parameters():
        parameter.grad = torch.zeros_like(parameter)

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.metrics["cabo/vlm_relative_update_rate"] == 0.0
    assert control.metrics["cabo/expert_relative_update_rate"] == 0.0
    assert control.metrics["cabo/projection_relative_update_rate"] == 0.0
    assert control.group_scales[CABO_ACTION_EXPERT_GROUP] == 1.0
    assert control.group_scales[CABO_ACTION_PROJECTION_GROUP] == 1.0


def test_pi05_cabo_reduces_only_six_update_moments_for_multi_process_training():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)
    accelerator = _ReducingFakeAccelerator()
    _set_unit_gradients(policy)

    control = policy.compute_optimizer_step_control({}, optimizer, accelerator)

    assert accelerator.reduce_calls == 1
    assert control.group_scales[CABO_ACTION_EXPERT_GROUP] == pytest.approx(0.2)
    assert control.group_scales[CABO_ACTION_PROJECTION_GROUP] == pytest.approx(1.0)


def test_pi05_cabo_nonfinite_update_skips_without_mutating_controller_state():
    policy = _TinyCABOPolicy()
    policy.vlm_weight.data.zero_()
    optimizer = _make_optimizer(policy)
    _set_unit_gradients(policy)
    vlm_group = get_vlm_group(optimizer)
    state_before = dict(vlm_group)

    control = policy.compute_optimizer_step_control({}, optimizer, _FakeAccelerator())

    assert control.skip_optimizer_step
    assert control.metrics["cabo/update_nonfinite"] == 1.0
    assert control.metrics["optimizer_step/nonfinite_cabo_update"] == 1.0
    assert "cabo_step" not in vlm_group
    for key, value in state_before.items():
        if key != "params":
            assert vlm_group[key] == value


def test_pi05_cabo_optimizer_contract_is_validated_before_training():
    policy = _TinyCABOPolicy()
    optimizer = _make_optimizer(policy)

    policy.validate_optimizer_step_control(optimizer)

    assert all(group["foreach"] is False for group in optimizer.param_groups)
    assert all(group["fused"] is False for group in optimizer.param_groups)


def test_pi05_cabo_optimizer_contract_rejects_wrong_optimizer_or_groups():
    policy = _TinyCABOPolicy()
    with pytest.raises(TypeError, match="AdamW"):
        policy.validate_optimizer_step_control(torch.optim.SGD(policy.parameters(), lr=0.1))

    unnamed_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="three named"):
        policy.validate_optimizer_step_control(unnamed_optimizer)

    swapped_optimizer = torch.optim.AdamW(
        [
            {"params": [policy.expert_weight], CABO_GROUP_NAME: CABO_VLM_GROUP},
            {"params": [policy.vlm_weight], CABO_GROUP_NAME: CABO_ACTION_EXPERT_GROUP},
            {
                "params": [policy.projection_weight],
                CABO_GROUP_NAME: CABO_ACTION_PROJECTION_GROUP,
            },
        ],
        lr=0.1,
        foreach=False,
    )
    with pytest.raises(ValueError, match="does not match"):
        policy.validate_optimizer_step_control(swapped_optimizer)
