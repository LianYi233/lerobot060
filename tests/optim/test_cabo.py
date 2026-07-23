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

import pytest
import torch
from torch import nn

from lerobot.optim.cabo import (
    CABO_ACTION_GROUP,
    CABO_GROUP_NAME,
    CABO_VLM_GROUP,
    adamw_candidate_parameter_delta,
    get_named_param_group,
    temporary_optimizer_group_lr_scales,
    update_cabo_budget,
    update_cabo_influence_balance,
    update_cabo_residual_compensation,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("existing_steps", [0, 2])
def test_adamw_candidate_delta_matches_real_optimizer_step(existing_steps: int, dtype: torch.dtype):
    parameter = nn.Parameter(torch.tensor([1.5, -0.75], dtype=dtype))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_GROUP}],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-9,
        weight_decay=0.2,
        amsgrad=True,
    )

    for index in range(existing_steps):
        parameter.grad = torch.tensor([0.2 + index, -0.4 - index], dtype=dtype)
        optimizer.step()
        optimizer.zero_grad()

    parameter.grad = torch.tensor([0.7, -1.1], dtype=dtype)
    group = get_named_param_group(optimizer, CABO_ACTION_GROUP)
    expected_delta = adamw_candidate_parameter_delta(parameter, group, optimizer.state.get(parameter, {}))
    before = parameter.detach().clone()

    optimizer.step()

    actual_delta = parameter.detach().to(dtype=expected_delta.dtype) - before.to(dtype=expected_delta.dtype)
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0.0, atol=0.0)


def test_temporary_group_lr_scale_scales_complete_adamw_step_and_restores_lr():
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_GROUP}],
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay=0.3,
    )
    parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    group = get_named_param_group(optimizer, CABO_ACTION_GROUP)
    full_delta = adamw_candidate_parameter_delta(parameter, group, optimizer.state.get(parameter, {}))
    before = parameter.detach().clone()

    with temporary_optimizer_group_lr_scales(optimizer, {CABO_ACTION_GROUP: 0.25}) as effective_lrs:
        assert effective_lrs[CABO_ACTION_GROUP] == pytest.approx(0.025)
        optimizer.step()

    torch.testing.assert_close(parameter.detach() - before, full_delta * 0.25)
    assert group["lr"] == pytest.approx(0.1)


def test_temporary_group_lr_scales_can_amplify_one_group_and_attenuate_another():
    vlm_parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    action_parameter = nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [
            {"params": [vlm_parameter], CABO_GROUP_NAME: CABO_VLM_GROUP},
            {"params": [action_parameter], CABO_GROUP_NAME: CABO_ACTION_GROUP},
        ],
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay=0.0,
    )
    vlm_parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    action_parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    vlm_before = vlm_parameter.detach().clone()
    action_before = action_parameter.detach().clone()

    with temporary_optimizer_group_lr_scales(
        optimizer,
        {CABO_VLM_GROUP: 2.0, CABO_ACTION_GROUP: 0.5},
    ) as effective_lrs:
        assert effective_lrs == pytest.approx({CABO_VLM_GROUP: 0.2, CABO_ACTION_GROUP: 0.05})
        optimizer.step()

    torch.testing.assert_close(
        vlm_parameter.detach() - vlm_before,
        torch.tensor([-0.2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        action_parameter.detach() - action_before,
        torch.tensor([-0.05], dtype=torch.float64),
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)


def test_cabo_residual_compensation_uses_vlm_only_for_post_action_residual():
    controller_group = {}

    vlm_scale, metrics = update_cabo_residual_compensation(
        controller_group,
        vlm_drift=4.0,
        action_drift=9.0,
        cross_drift=2.0,
        residual_energy=16.0,
        residual_vlm_alignment=-3.0,
        residual_action_alignment=-5.0,
        ema_decay=0.0,
        regularization=0.0,
        max_vlm_scale=1.0,
    )

    # After the full action update, <e + da, dv> = -3 + 2 = -1. The VLM therefore
    # supplies exactly one quarter of its candidate update: -(-1) / Dv = 0.25.
    assert vlm_scale == pytest.approx(0.25)
    assert metrics["cabo/action_scale"] == pytest.approx(1.0)
    assert metrics["cabo/post_action_vlm_alignment_ema"] == pytest.approx(-1.0)
    assert metrics["cabo/predicted_action_only_residual_ema"] == pytest.approx(15.0)
    assert metrics["cabo/predicted_joint_residual_ema"] == pytest.approx(14.75)
    assert metrics["cabo/predicted_vlm_improvement_ema"] == pytest.approx(0.25)
    assert controller_group["cabo_vlm_scale"] == pytest.approx(0.25)
    assert controller_group["cabo_action_scale"] == pytest.approx(1.0)


def test_cabo_residual_compensation_rejects_vlm_update_that_worsens_action_residual():
    vlm_scale, metrics = update_cabo_residual_compensation(
        {},
        vlm_drift=1.0,
        action_drift=1.0,
        cross_drift=0.5,
        residual_energy=4.0,
        residual_vlm_alignment=0.25,
        residual_action_alignment=-1.0,
        ema_decay=0.0,
        regularization=0.1,
        max_vlm_scale=1.0,
    )

    assert vlm_scale == pytest.approx(0.0)
    assert metrics["cabo/post_action_vlm_alignment_ema"] == pytest.approx(0.75)
    assert metrics["cabo/predicted_vlm_improvement_ema"] == pytest.approx(0.0)
    assert metrics["cabo/residual_scale_clamped"] == 1.0


def test_cabo_residual_compensation_can_preserve_a_vlm_learning_floor():
    vlm_scale, metrics = update_cabo_residual_compensation(
        {},
        vlm_drift=1.0,
        action_drift=1.0,
        cross_drift=0.5,
        residual_energy=4.0,
        residual_vlm_alignment=0.25,
        residual_action_alignment=-1.0,
        ema_decay=0.0,
        regularization=0.1,
        min_vlm_scale=0.1,
        max_vlm_scale=1.0,
    )

    assert vlm_scale == pytest.approx(0.1)
    assert metrics["cabo/residual_scale_at_floor"] == 1.0
    assert metrics["cabo/residual_scale_clamped"] == 1.0


def test_cabo_residual_compensation_regularizes_and_caps_vlm_scale():
    vlm_scale, metrics = update_cabo_residual_compensation(
        {},
        vlm_drift=1.0,
        action_drift=1.0,
        cross_drift=0.0,
        residual_energy=9.0,
        residual_vlm_alignment=-3.0,
        residual_action_alignment=-1.0,
        ema_decay=0.0,
        regularization=1.0,
        max_vlm_scale=1.0,
    )

    # The regularized unconstrained scale is 1.5, so the trust-region cap wins.
    assert vlm_scale == pytest.approx(1.0)
    assert metrics["cabo/residual_scale_clamped"] == 1.0


def test_cabo_residual_compensation_preserves_state_on_nonfinite_probe():
    controller_group = {"cabo_vlm_scale": 0.25, "cabo_action_scale": 1.0}

    vlm_scale, metrics = update_cabo_residual_compensation(
        controller_group,
        vlm_drift=1.0,
        action_drift=1.0,
        cross_drift=0.0,
        residual_energy=float("nan"),
        residual_vlm_alignment=0.0,
        residual_action_alignment=0.0,
        ema_decay=0.0,
        regularization=0.1,
        max_vlm_scale=1.0,
    )

    assert vlm_scale == pytest.approx(0.25)
    assert controller_group == {"cabo_vlm_scale": 0.25, "cabo_action_scale": 1.0}
    assert metrics["cabo/probe_nonfinite"] == 1.0


def test_cabo_influence_balance_symmetrically_equalizes_marginal_drift():
    controller_group = {}

    vlm_scale, action_scale, metrics = update_cabo_influence_balance(
        controller_group,
        vlm_drift=1.0,
        action_drift=16.0,
        cross_drift=-2.0,
        ema_decay=0.0,
        max_scale=4.0,
    )

    assert vlm_scale == pytest.approx(2.0)
    assert action_scale == pytest.approx(0.5)
    assert vlm_scale * action_scale == pytest.approx(1.0)
    assert metrics["cabo/balanced_vlm_drift_ema"] == pytest.approx(4.0)
    assert metrics["cabo/balanced_action_drift_ema"] == pytest.approx(4.0)
    assert metrics["cabo/balanced_influence_ratio"] == pytest.approx(1.0)
    assert metrics["cabo/balance_clamped"] == 0.0
    assert controller_group["cabo_vlm_scale"] == pytest.approx(2.0)
    assert controller_group["cabo_action_scale"] == pytest.approx(0.5)


def test_cabo_influence_balance_is_symmetric_when_vlm_is_stronger():
    vlm_scale, action_scale, metrics = update_cabo_influence_balance(
        {},
        vlm_drift=16.0,
        action_drift=1.0,
        cross_drift=0.0,
        ema_decay=0.0,
        max_scale=4.0,
    )

    assert vlm_scale == pytest.approx(0.5)
    assert action_scale == pytest.approx(2.0)
    assert metrics["cabo/balanced_vlm_drift_ema"] == pytest.approx(4.0)
    assert metrics["cabo/balanced_action_drift_ema"] == pytest.approx(4.0)


def test_cabo_influence_balance_caps_zero_drift_without_dividing_by_zero():
    vlm_scale, action_scale, metrics = update_cabo_influence_balance(
        {},
        vlm_drift=0.0,
        action_drift=9.0,
        cross_drift=0.0,
        ema_decay=0.0,
        max_scale=2.0,
    )

    assert vlm_scale == pytest.approx(2.0)
    assert action_scale == pytest.approx(0.5)
    assert metrics["cabo/balance_clamped"] == 1.0
    assert metrics["cabo/balanced_vlm_drift_ema"] == pytest.approx(0.0)
    assert metrics["cabo/balanced_action_drift_ema"] == pytest.approx(2.25)


def test_cabo_influence_balance_preserves_state_on_nonfinite_probe():
    controller_group = {"cabo_vlm_scale": 2.0, "cabo_action_scale": 0.5}

    vlm_scale, action_scale, metrics = update_cabo_influence_balance(
        controller_group,
        vlm_drift=float("nan"),
        action_drift=1.0,
        cross_drift=0.0,
        ema_decay=0.0,
        max_scale=2.0,
    )

    assert vlm_scale == 2.0
    assert action_scale == 0.5
    assert controller_group == {"cabo_vlm_scale": 2.0, "cabo_action_scale": 0.5}
    assert metrics["cabo/probe_nonfinite"] == 1.0


def test_cabo_budget_enforces_rms_drift_ratio_and_persists_state():
    action_group = {"lr": 1e-3}

    scale, metrics = update_cabo_budget(
        action_group,
        vlm_drift=4.0,
        action_drift=9.0,
        cross_drift=6.0,
        action_drift_ratio=0.1,
        base_action_scale=0.0,
        negative_cross_discount=0.5,
        probe_interval=8,
        ema_decay=0.9,
        budget_decay=0.95,
        budget_cap_windows=4.0,
    )

    # Perfectly aligned updates recover the old worst-case marginal ratio.
    assert scale == pytest.approx(0.1 * (4.0 / 9.0) ** 0.5)
    assert metrics["cabo/total_drift_ema"] == pytest.approx((1.0 + 0.1) ** 2 * metrics["cabo/vlm_drift_ema"])
    assert metrics["cabo/cross_correlation_ema"] == pytest.approx(1.0)
    assert action_group["cabo_action_scale"] == pytest.approx(scale)
    assert action_group["cabo_ema_initialized"] is True


def test_cabo_joint_budget_uses_cross_drift_geometry():
    common = {
        "vlm_drift": 1.0,
        "action_drift": 9.0,
        "action_drift_ratio": 0.1,
        "base_action_scale": 0.0,
        "negative_cross_discount": 1.0,
        "probe_interval": 1,
        "ema_decay": 0.0,
        "budget_decay": 0.0,
        "budget_cap_windows": 1.0,
    }

    aligned_scale, _ = update_cabo_budget({}, cross_drift=3.0, **common)
    orthogonal_scale, _ = update_cabo_budget({}, cross_drift=0.0, **common)
    cancelling_scale, cancelling_metrics = update_cabo_budget({}, cross_drift=-3.0, **common)

    assert aligned_scale == pytest.approx(1.0 / 30.0)
    assert orthogonal_scale == pytest.approx((0.21 / 9.0) ** 0.5)
    assert cancelling_scale == pytest.approx(0.7)
    assert aligned_scale < orthogonal_scale < cancelling_scale
    assert cancelling_metrics["cabo/total_drift_ema"] == pytest.approx(1.21)


def test_cabo_base_action_scale_prevents_stationary_vlm_from_starving_action():
    scale, metrics = update_cabo_budget(
        {},
        vlm_drift=0.0,
        action_drift=9.0,
        cross_drift=0.0,
        action_drift_ratio=0.1,
        base_action_scale=0.2,
        negative_cross_discount=0.5,
        probe_interval=8,
        ema_decay=0.9,
        budget_decay=0.95,
        budget_cap_windows=4.0,
    )

    assert scale == pytest.approx(0.2)
    assert metrics["cabo/base_action_allowance"] == pytest.approx(0.2**2 * 9.0)


def test_cabo_discounts_negative_cross_drift_credit():
    common = {
        "vlm_drift": 1.0,
        "action_drift": 4.0,
        "cross_drift": -2.0,
        "action_drift_ratio": 0.1,
        "base_action_scale": 0.0,
        "probe_interval": 1,
        "ema_decay": 0.0,
        "budget_decay": 0.0,
        "budget_cap_windows": 1.0,
    }

    full_credit_scale, _ = update_cabo_budget({}, negative_cross_discount=1.0, **common)
    discounted_scale, metrics = update_cabo_budget({}, negative_cross_discount=0.5, **common)

    assert discounted_scale < full_credit_scale
    assert metrics["cabo/effective_cross_drift_ema"] == pytest.approx(-1.0)


def test_cabo_budget_preserves_state_on_nonfinite_probe():
    action_group = {"cabo_action_scale": 0.5, "cabo_budget": 2.0}

    scale, metrics = update_cabo_budget(
        action_group,
        vlm_drift=float("nan"),
        action_drift=1.0,
        cross_drift=0.0,
        action_drift_ratio=0.1,
        base_action_scale=0.1,
        negative_cross_discount=0.5,
        probe_interval=8,
        ema_decay=0.9,
        budget_decay=0.95,
        budget_cap_windows=4.0,
    )

    assert scale == 0.5
    assert action_group["cabo_action_scale"] == 0.5
    assert action_group["cabo_budget"] == 2.0
    assert metrics["cabo/probe_nonfinite"] == 1.0


@pytest.mark.parametrize("mode", ["foreach", "fused", "capturable", "differentiable"])
def test_adamw_candidate_delta_rejects_unsupported_optimizer_modes(mode: str):
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.5])
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    group = optimizer.param_groups[0]
    group[mode] = True

    with pytest.raises(RuntimeError, match=mode):
        adamw_candidate_parameter_delta(parameter, group, optimizer.state.get(parameter, {}))
