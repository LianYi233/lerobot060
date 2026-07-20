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
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


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
            cabo_num_projections=4,
            cabo_base_action_scale=0.0,
            cabo_negative_cross_discount=0.5,
            cabo_drift_ema_decay=0.9,
            cabo_budget_decay=0.95,
            cabo_budget_cap_windows=4.0,
        )

    def _cabo_probe_velocity(self, batch, *, step, process_index=0):
        _ = batch, step, process_index
        # Both branches affect the same scalar action output, with action twice as sensitive.
        velocity = self.vlm_weight + 2.0 * self.action_weight
        return torch.stack([velocity, -velocity, velocity, -velocity])


class _ProbeModel:
    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.full((batch_size,), 0.5, device=device)

    def predict_velocity(self, images, img_masks, tokens, masks, x_t, sample_time):
        _ = images, img_masks, tokens, masks, sample_time
        return x_t


class _ProbeBatchPolicy:
    _cabo_probe_velocity = PI05Policy._cabo_probe_velocity

    def __init__(self):
        self.config = SimpleNamespace(
            cabo_probe_batch_size=2,
            cabo_num_projections=4,
            output_features={ACTION: SimpleNamespace(shape=(1,))},
        )
        self.model = _ProbeModel()
        self.selected_actions = None

    def _preprocess_images(self, batch):
        _ = batch
        return [], []

    def prepare_action(self, batch):
        self.selected_actions = batch[ACTION]
        return batch[ACTION]


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

    # Candidate AdamW deltas are both -0.1; the probe Jacobians are 1 and 2, so the moments are
    # Dv=0.01, Da=0.04, Cva=0.02. Perfect alignment recovers the old worst-case action scale.
    assert metrics["cabo/vlm_drift"] == pytest.approx(0.01)
    assert metrics["cabo/action_drift"] == pytest.approx(0.04)
    assert metrics["cabo/cross_drift"] == pytest.approx(0.02)
    assert metrics["cabo/cross_correlation_ema"] == pytest.approx(1.0)
    assert scales[CABO_ACTION_GROUP] == pytest.approx(0.05)
    assert metrics["cabo/probe_applied"] == 1.0
    assert metrics["cabo/num_projections"] == 4.0
    for parameter, gradient_before in zip(policy.parameters(), gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)

    reused_scales, reused_metrics = policy.compute_optimizer_step_control(
        batch, optimizer, _FakeAccelerator()
    )
    assert reused_scales == scales
    assert reused_metrics["cabo/probe_applied"] == 0.0


def test_pi05_cabo_probe_excludes_rows_with_padded_action_horizons():
    policy = _ProbeBatchPolicy()
    batch = {
        ACTION: torch.tensor([[[1.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]]),
        f"{ACTION}_is_pad": torch.tensor([[False, True], [False, False], [False, False]]),
        OBS_LANGUAGE_TOKENS: torch.ones(3, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(3, 2, dtype=torch.bool),
    }

    probe_scalars = policy._cabo_probe_velocity(batch, step=0)

    assert probe_scalars is not None
    assert probe_scalars.shape == (4,)
    torch.testing.assert_close(policy.selected_actions, batch[ACTION][1:])


def test_pi05_cabo_probe_skips_an_all_padding_batch():
    policy = _ProbeBatchPolicy()
    batch = {
        ACTION: torch.ones(2, 3, 1),
        f"{ACTION}_is_pad": torch.ones(2, 3, dtype=torch.bool),
    }

    assert policy._cabo_probe_velocity(batch, step=0) is None
    assert policy.selected_actions is None
