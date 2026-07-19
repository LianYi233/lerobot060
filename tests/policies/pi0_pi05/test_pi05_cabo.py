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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.optim.cabo import CABO_ACTION_GROUP, CABO_VLM_GROUP  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


class _FakeAccelerator:
    def autocast(self):
        return nullcontext()

    def reduce(self, value, reduction="mean"):
        assert reduction == "mean"
        return value


class _TinyCABOPolicy(nn.Module):
    compute_optimizer_step_control = PI05Policy.compute_optimizer_step_control

    def __init__(self):
        super().__init__()
        self.vlm_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.action_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.config = SimpleNamespace(
            cabo_enabled=True,
            cabo_probe_interval=8,
            cabo_action_drift_ratio=0.1,
            cabo_drift_ema_decay=0.9,
            cabo_budget_decay=0.95,
            cabo_budget_cap_windows=4.0,
        )

    def _cabo_probe_velocity(self, batch, *, step):
        _ = batch, step
        # Both branches affect the same scalar action output, with action twice as sensitive.
        return self.vlm_weight + 2.0 * self.action_weight


def test_pi05_cabo_probe_preserves_training_gradients_and_reuses_scale():
    policy = _TinyCABOPolicy()
    optimizer = torch.optim.AdamW(
        [
            {"params": [policy.vlm_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.action_weight], "name": CABO_ACTION_GROUP},
        ],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-12,
        weight_decay=0.0,
    )
    policy.vlm_weight.grad = torch.tensor(1.0, dtype=torch.float64)
    policy.action_weight.grad = torch.tensor(1.0, dtype=torch.float64)
    gradients_before = [parameter.grad.clone() for parameter in policy.parameters()]
    batch = {ACTION: torch.ones(1, 1, 1)}

    scales, metrics = policy.compute_optimizer_step_control(batch, optimizer, _FakeAccelerator())

    # Candidate AdamW deltas are both -0.1; the probe Jacobians are 1 and 2, so drift is 0.01/0.04.
    assert metrics["cabo/vlm_drift"] == pytest.approx(0.01)
    assert metrics["cabo/action_drift"] == pytest.approx(0.04)
    assert scales[CABO_ACTION_GROUP] == pytest.approx(0.05)
    assert metrics["cabo/probe_applied"] == 1.0
    for parameter, gradient_before in zip(policy.parameters(), gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)

    reused_scales, reused_metrics = policy.compute_optimizer_step_control(
        batch, optimizer, _FakeAccelerator()
    )
    assert reused_scales == scales
    assert reused_metrics["cabo/probe_applied"] == 0.0
