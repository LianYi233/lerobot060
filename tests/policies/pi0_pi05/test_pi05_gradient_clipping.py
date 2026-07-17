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

from lerobot.policies.pi05 import PI05Config, PI05Policy  # noqa: E402


def _make_module() -> nn.Module:
    return nn.Linear(2, 2, bias=False)


def _make_policy_with_gradients(
    *, action_gradient: float, vlm_gradient: float
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
            action_head_grad_clip_ratio=1.0,
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

    assert config.optimizer_grad_clip_norm == 0.0
    assert config.get_optimizer_preset().grad_clip_norm == 0.0
    assert config.clip_action_head_by_vlm
    assert config.action_head_grad_clip_ratio == 1.0


def test_pi05_clips_action_gradient_rms_to_vlm_rms_without_modifying_vlm():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=4.0,
        vlm_gradient=2.0,
    )
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 1.0
    assert metrics["action_head_grad_rms_before_clip"] == pytest.approx(4.0)
    assert metrics["vlm_grad_rms"] == pytest.approx(2.0)
    assert _gradient_rms(action_parameters) == pytest.approx(2.0, abs=1e-6)
    assert metrics["action_head_grad_rms_after_clip"] == pytest.approx(2.0, abs=1e-6)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)


def test_pi05_leaves_both_groups_unchanged_when_action_rms_is_not_larger():
    policy, action_parameters, vlm_parameters = _make_policy_with_gradients(
        action_gradient=1.0,
        vlm_gradient=2.0,
    )
    action_gradients_before = [parameter.grad.clone() for parameter in action_parameters]
    vlm_gradients_before = [parameter.grad.clone() for parameter in vlm_parameters]

    metrics = PI05Policy.clip_gradients(policy)

    assert metrics["action_head_clip_applied"] == 0.0
    for parameter, gradient_before in zip(action_parameters, action_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
    for parameter, gradient_before in zip(vlm_parameters, vlm_gradients_before, strict=True):
        assert torch.equal(parameter.grad, gradient_before)
