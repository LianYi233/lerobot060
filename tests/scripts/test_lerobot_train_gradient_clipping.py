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

pytest.importorskip("datasets")

from lerobot.scripts.lerobot_train import _clip_policy_gradients, update_policy  # noqa: E402


class _PolicyWithCustomGradientClipping(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.clip_calls = 0

    def forward(self, batch):
        return self.weight * batch, {"forward_metric": 2.0}

    @torch.no_grad()
    def clip_gradients(self, accelerator=None) -> dict[str, float]:
        if accelerator is not None:
            accelerator.events.append("custom_clip")
        self.clip_calls += 1
        self.weight.grad.mul_(0.25)
        return {"custom_clip_applied": 1.0}


class _PolicyWithoutCustomGradientClipping(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


class _FakeAccelerator:
    def __init__(self):
        self.unscale_calls = 0
        self.global_clip_calls = 0
        self.events = []

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        self.events.append("backward")
        loss.backward()

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        _ = keep_fp32_wrapper
        return policy

    def unscale_gradients(self, optimizer):
        _ = optimizer
        self.unscale_calls += 1
        self.events.append("unscale")

    def clip_grad_norm_(self, parameters, max_norm):
        self.global_clip_calls += 1
        self.events.append("global_clip")
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def test_policy_specific_hook_replaces_global_gradient_clipping():
    policy = _PolicyWithCustomGradientClipping()
    policy.weight.grad = torch.tensor(4.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    grad_norm, metrics = _clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert grad_norm.item() == pytest.approx(4.0)
    assert policy.weight.grad.item() == pytest.approx(1.0)
    assert metrics == {"custom_clip_applied": 1.0}
    assert policy.clip_calls == 1
    assert accelerator.unscale_calls == 1
    assert accelerator.global_clip_calls == 0
    assert accelerator.events == ["unscale", "custom_clip"]


def test_policy_without_custom_hook_keeps_global_gradient_clipping():
    policy = _PolicyWithoutCustomGradientClipping()
    policy.weight.grad = torch.tensor(4.0)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    accelerator = _FakeAccelerator()

    grad_norm, metrics = _clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert grad_norm.item() == pytest.approx(4.0)
    assert policy.weight.grad.item() == pytest.approx(1.0)
    assert metrics == {}
    assert accelerator.unscale_calls == 0
    assert accelerator.global_clip_calls == 1


def test_update_policy_calls_custom_hook_and_merges_clipping_metrics():
    policy = _PolicyWithCustomGradientClipping()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
    accelerator = _FakeAccelerator()
    train_metrics = SimpleNamespace()

    train_metrics, output_dict = update_policy(
        train_metrics=train_metrics,
        policy=policy,
        batch=torch.tensor(4.0),
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
    )

    assert policy.clip_calls == 1
    assert accelerator.events == ["backward", "unscale", "custom_clip"]
    assert accelerator.global_clip_calls == 0
    assert train_metrics.grad_norm == pytest.approx(4.0)
    assert output_dict == {"forward_metric": 2.0, "custom_clip_applied": 1.0}
