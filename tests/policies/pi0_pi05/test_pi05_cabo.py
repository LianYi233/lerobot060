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

import datetime as dt
import os
import socket
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
    num_processes = 1

    def autocast(self):
        return nullcontext()

    def reduce(self, value, reduction="mean"):
        assert reduction == "sum"
        return value


class _TinyCABOPolicy(nn.Module):
    compute_optimizer_step_control = PI05Policy.compute_optimizer_step_control
    validate_optimizer_step_control = PI05Policy.validate_optimizer_step_control

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
        return torch.stack([velocity, -velocity, velocity, -velocity]), 1

    def forward(self, value):
        return value * (self.vlm_weight + self.action_weight)

    def _cabo_parameter_groups(self):
        return [self.vlm_weight], [self.action_weight]


class _RankSparseCABOPolicy(_TinyCABOPolicy):
    def _cabo_probe_velocity(self, batch, *, step, process_index=0):
        if process_index == 0:
            return None, 0
        return super()._cabo_probe_velocity(batch, step=step, process_index=process_index)


class _DistributedFakeAccelerator(_FakeAccelerator):
    def __init__(self, rank: int, world_size: int):
        self.process_index = rank
        self.num_processes = world_size

    def reduce(self, value, reduction="sum"):
        assert reduction == "sum"
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value


def _run_rank_sparse_cabo_probe(rank, world_size, init_method, result_queue):
    interface_names = {name for _, name in socket.if_nameindex()}
    loopback_interface = "lo0" if "lo0" in interface_names else "lo"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback_interface)
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(seconds=30),
    )
    try:
        policy = _RankSparseCABOPolicy()
        ddp_policy = torch.nn.parallel.DistributedDataParallel(policy)
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
        ddp_policy(torch.tensor(0.5, dtype=torch.float64)).backward()
        gradients_before = [parameter.grad.clone() for parameter in policy.parameters()]
        with ddp_policy.no_sync():
            control = policy.compute_optimizer_step_control(
                {ACTION: torch.ones(1, 1, 1)},
                optimizer,
                _DistributedFakeAccelerator(rank, world_size),
            )
        gradients_preserved = all(
            torch.equal(parameter.grad, before)
            for parameter, before in zip(policy.parameters(), gradients_before, strict=True)
        )
        result_queue.put(
            (
                rank,
                control.group_scales[CABO_ACTION_GROUP],
                control.metrics["cabo/budget"],
                control.metrics["cabo/probe_valid_elements"],
                gradients_preserved,
            )
        )
    finally:
        torch.distributed.destroy_process_group()


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
            cabo_probe_batch_size=3,
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

    control = policy.compute_optimizer_step_control(batch, optimizer, _FakeAccelerator())
    scales, metrics = control.group_scales, control.metrics

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

    reused_control = policy.compute_optimizer_step_control(batch, optimizer, _FakeAccelerator())
    assert reused_control.group_scales == scales
    assert reused_control.metrics["cabo/probe_applied"] == 0.0


def test_pi05_cabo_probe_masks_padded_action_steps_without_excluding_partial_rows():
    policy = _ProbeBatchPolicy()
    batch = {
        ACTION: torch.tensor([[[1.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]]),
        f"{ACTION}_is_pad": torch.tensor([[False, True], [False, False], [False, False]]),
        OBS_LANGUAGE_TOKENS: torch.ones(3, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(3, 2, dtype=torch.bool),
    }

    probe_scalars, valid_output_elements = policy._cabo_probe_velocity(batch, step=0)

    assert probe_scalars is not None
    assert probe_scalars.shape == (4,)
    assert valid_output_elements == 5
    assert sorted(policy.selected_actions[:, 0, 0].tolist()) == [1.0, 2.0, 4.0]


def test_pi05_cabo_probe_skips_an_all_padding_batch():
    policy = _ProbeBatchPolicy()
    batch = {
        ACTION: torch.ones(2, 3, 1),
        f"{ACTION}_is_pad": torch.ones(2, 3, dtype=torch.bool),
    }

    probe_scalars, valid_output_elements = policy._cabo_probe_velocity(batch, step=0)

    assert probe_scalars is None
    assert valid_output_elements == 0
    assert policy.selected_actions is None


def test_pi05_cabo_probe_sampling_is_reproducible_for_step_and_rank():
    batch = {
        ACTION: torch.arange(8, dtype=torch.float32).reshape(4, 2, 1),
        OBS_LANGUAGE_TOKENS: torch.ones(4, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 2, dtype=torch.bool),
    }
    first = _ProbeBatchPolicy()
    second = _ProbeBatchPolicy()
    first.config.cabo_probe_batch_size = 2
    second.config.cabo_probe_batch_size = 2

    training_rng_state = torch.random.get_rng_state()
    first_scalars, first_count = first._cabo_probe_velocity(batch, step=7, process_index=1)
    assert torch.equal(torch.random.get_rng_state(), training_rng_state)
    second_scalars, second_count = second._cabo_probe_velocity(batch, step=7, process_index=1)

    torch.testing.assert_close(first.selected_actions, second.selected_actions)
    torch.testing.assert_close(first_scalars, second_scalars)
    assert first_count == second_count == 4


def test_pi05_cabo_probe_preserves_none_training_gradients():
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
    policy.action_weight.grad = None

    control = policy.compute_optimizer_step_control(
        {ACTION: torch.ones(1, 1, 1)}, optimizer, _FakeAccelerator()
    )

    assert policy.action_weight.grad is None
    assert policy.vlm_weight.grad.item() == pytest.approx(1.0)
    assert control.metrics["cabo/action_drift"] == pytest.approx(0.0)


def test_pi05_cabo_nonfinite_probe_skips_without_mutating_controller_state():
    policy = _TinyCABOPolicy()

    def nonfinite_probe(batch, *, step, process_index=0):
        _ = batch, step, process_index
        scalar = policy.vlm_weight * torch.tensor(float("nan"), dtype=torch.float64)
        return torch.stack([scalar, scalar, scalar, scalar]), 1

    policy._cabo_probe_velocity = nonfinite_probe
    optimizer = torch.optim.AdamW(
        [
            {"params": [policy.vlm_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.action_weight], "name": CABO_ACTION_GROUP},
        ],
        lr=0.1,
    )
    policy.vlm_weight.grad = torch.tensor(1.0, dtype=torch.float64)
    policy.action_weight.grad = torch.tensor(1.0, dtype=torch.float64)

    control = policy.compute_optimizer_step_control(
        {ACTION: torch.ones(1, 1, 1)}, optimizer, _FakeAccelerator()
    )

    action_group = optimizer.param_groups[1]
    assert control.skip_optimizer_step
    assert control.metrics["optimizer_step/nonfinite_probe"] == 1.0
    assert "cabo_step" not in action_group
    assert "cabo_budget" not in action_group
    assert "cabo_action_scale" not in action_group


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_pi05_cabo_ddp_rank_without_valid_probe_still_joins_collective(tmp_path):
    world_size = 2
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'cabo_gloo_init'}"

    torch.multiprocessing.spawn(
        _run_rank_sparse_cabo_probe,
        args=(world_size, init_method, result_queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(result_queue.get() for _ in range(world_size))

    assert results[0][1:4] == pytest.approx(results[1][1:4])
    assert results[0][3] == pytest.approx(1.0)
    assert results[0][4] is results[1][4] is True


def test_pi05_cabo_optimizer_contract_is_validated_before_training():
    policy = _TinyCABOPolicy()
    optimizer = torch.optim.AdamW(
        [
            {"params": [policy.vlm_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.action_weight], "name": CABO_ACTION_GROUP},
        ]
    )

    policy.validate_optimizer_step_control(optimizer)

    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[1]["foreach"] is False


def test_pi05_cabo_optimizer_contract_rejects_wrong_optimizer_or_groups():
    policy = _TinyCABOPolicy()
    with pytest.raises(TypeError, match="AdamW"):
        policy.validate_optimizer_step_control(torch.optim.SGD(policy.parameters(), lr=0.1))

    unnamed_optimizer = torch.optim.AdamW(policy.parameters())
    with pytest.raises(ValueError, match="exactly two"):
        policy.validate_optimizer_step_control(unnamed_optimizer)

    swapped_optimizer = torch.optim.AdamW(
        [
            {"params": [policy.action_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.vlm_weight], "name": CABO_ACTION_GROUP},
        ]
    )
    with pytest.raises(ValueError, match="does not match"):
        policy.validate_optimizer_step_control(swapped_optimizer)

    duplicate_named_groups = torch.optim.AdamW(
        [
            {"params": [policy.vlm_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.action_weight], "name": CABO_VLM_GROUP},
        ]
    )
    with pytest.raises(ValueError, match="group named"):
        policy.validate_optimizer_step_control(duplicate_named_groups)

    optimizer_with_extra_empty_group = torch.optim.AdamW(
        [
            {"params": [policy.vlm_weight], "name": CABO_VLM_GROUP},
            {"params": [policy.action_weight], "name": CABO_ACTION_GROUP},
            {"params": [], "name": "extra"},
        ]
    )
    with pytest.raises(ValueError, match="exactly two"):
        policy.validate_optimizer_step_control(optimizer_with_extra_empty_group)
