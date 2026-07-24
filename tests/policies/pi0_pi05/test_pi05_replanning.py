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

from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05 import PI05Policy  # noqa: E402


def test_select_action_replans_old_fifty_step_checkpoints_after_ten_steps():
    """A legacy chunk stays shape-compatible but is not executed fully open loop."""
    policy = SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=50,
            n_action_steps=50,
            replan_interval=10,
        )
    )
    policy._execution_horizon = MethodType(PI05Policy._execution_horizon, policy)
    policy._rtc_enabled = lambda: False
    policy.eval = lambda: None

    prediction_calls = 0

    def predict_action_chunk(_batch):
        nonlocal prediction_calls
        prediction_calls += 1
        return torch.full((1, 50, 1), float(prediction_calls))

    policy.predict_action_chunk = predict_action_chunk
    PI05Policy.reset(policy)

    first_ten = [PI05Policy.select_action(policy, {}) for _ in range(10)]
    eleventh = PI05Policy.select_action(policy, {})

    assert policy._action_queue.maxlen == 10
    assert prediction_calls == 2
    assert torch.stack(first_ten).flatten().tolist() == [1.0] * 10
    assert eleventh.item() == 2.0


def test_replan_interval_never_extends_n_action_steps():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            n_action_steps=4,
            replan_interval=10,
        )
    )

    assert PI05Policy._execution_horizon(policy) == 4
