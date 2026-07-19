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
    adamw_candidate_parameter_delta,
    get_named_param_group,
    temporary_optimizer_group_lr_scales,
    update_cabo_budget,
)


@pytest.mark.parametrize("existing_steps", [0, 2])
def test_adamw_candidate_delta_matches_real_optimizer_step(existing_steps: int):
    parameter = nn.Parameter(torch.tensor([1.5, -0.75], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], CABO_GROUP_NAME: CABO_ACTION_GROUP}],
        lr=3e-3,
        betas=(0.8, 0.95),
        eps=1e-9,
        weight_decay=0.2,
        amsgrad=True,
    )

    for index in range(existing_steps):
        parameter.grad = torch.tensor([0.2 + index, -0.4 - index], dtype=torch.float64)
        optimizer.step()
        optimizer.zero_grad()

    parameter.grad = torch.tensor([0.7, -1.1], dtype=torch.float64)
    group = get_named_param_group(optimizer, CABO_ACTION_GROUP)
    expected_delta = adamw_candidate_parameter_delta(parameter, group, optimizer.state.get(parameter, {}))
    before = parameter.detach().clone()

    optimizer.step()

    torch.testing.assert_close(parameter.detach() - before, expected_delta, rtol=1e-12, atol=1e-12)


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


def test_cabo_budget_enforces_rms_drift_ratio_and_persists_state():
    action_group = {"lr": 1e-3}

    scale, metrics = update_cabo_budget(
        action_group,
        vlm_drift=4.0,
        action_drift=9.0,
        action_drift_ratio=0.1,
        probe_interval=8,
        ema_decay=0.9,
        budget_decay=0.95,
        budget_cap_windows=4.0,
    )

    # scale^2 * Da == ratio^2 * Dv on the first probe because no prior budget exists.
    assert scale == pytest.approx(0.1 * (4.0 / 9.0) ** 0.5)
    assert scale**2 * metrics["cabo/action_drift_ema"] == pytest.approx(
        0.1**2 * metrics["cabo/vlm_drift_ema"]
    )
    assert action_group["cabo_action_scale"] == pytest.approx(scale)
    assert action_group["cabo_ema_initialized"] is True


def test_cabo_budget_freezes_action_on_nonfinite_probe():
    action_group = {"cabo_action_scale": 0.5, "cabo_budget": 2.0}

    scale, metrics = update_cabo_budget(
        action_group,
        vlm_drift=float("nan"),
        action_drift=1.0,
        action_drift_ratio=0.1,
        probe_interval=8,
        ema_decay=0.9,
        budget_decay=0.95,
        budget_cap_windows=4.0,
    )

    assert scale == 0.0
    assert action_group["cabo_action_scale"] == 0.0
    assert metrics["cabo/probe_nonfinite"] == 1.0
