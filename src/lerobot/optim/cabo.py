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

"""Optimizer-side helpers for Counterfactual Action-Budget Optimization (CABO)."""

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

CABO_VLM_GROUP = "vlm"
CABO_ACTION_GROUP = "action"
CABO_GROUP_NAME = "name"


def unwrap_optimizer(optimizer: Optimizer) -> Optimizer:
    """Return the torch optimizer hidden by wrappers such as AcceleratedOptimizer."""
    unwrapped = optimizer
    seen: set[int] = set()
    # AcceleratedOptimizer itself subclasses Optimizer, so the presence of a nested optimizer—not
    # isinstance—is the wrapper signal.
    while hasattr(unwrapped, "optimizer"):
        if id(unwrapped) in seen:
            break
        seen.add(id(unwrapped))
        nested = unwrapped.optimizer
        if nested is unwrapped:
            break
        unwrapped = nested
    if not isinstance(unwrapped, Optimizer):
        raise TypeError(f"CABO requires a torch Optimizer, got {type(unwrapped).__name__}")
    return unwrapped


def require_adamw(optimizer: Optimizer) -> AdamW:
    """Unwrap and validate the optimizer supported by the CABO update projector."""
    unwrapped = unwrap_optimizer(optimizer)
    if not isinstance(unwrapped, AdamW):
        raise TypeError(f"CABO currently supports AdamW only, got {type(unwrapped).__name__}")
    return unwrapped


def get_named_param_group(optimizer: Optimizer, name: str) -> dict[str, Any]:
    """Find exactly one optimizer parameter group by its stable CABO name."""
    unwrapped = unwrap_optimizer(optimizer)
    matches = [group for group in unwrapped.param_groups if group.get(CABO_GROUP_NAME) == name]
    if len(matches) != 1:
        raise ValueError(f"CABO expected one optimizer group named {name!r}, found {len(matches)}")
    return matches[0]


@torch.no_grad()
def adamw_candidate_parameter_delta(
    parameter: nn.Parameter,
    param_group: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Tensor | None:
    """Compute the next AdamW parameter delta without mutating parameters or optimizer state.

    The result includes decoupled weight decay and is evaluated at the group's current, unscaled
    learning rate. Scaling that learning rate by ``s`` therefore scales this complete delta by ``s``.
    """
    gradient = parameter.grad
    if gradient is None:
        return None
    if gradient.is_sparse:
        raise RuntimeError("CABO does not support sparse AdamW gradients")
    if torch.is_complex(parameter) or torch.is_complex(gradient):
        raise RuntimeError("CABO does not support complex AdamW parameters")
    if param_group.get("differentiable", False):
        raise RuntimeError("CABO does not support differentiable AdamW steps")

    beta1, beta2 = param_group["betas"]
    eps = float(param_group["eps"])
    learning_rate = float(param_group["lr"])
    weight_decay = float(param_group["weight_decay"])
    maximize = bool(param_group.get("maximize", False))
    amsgrad = bool(param_group.get("amsgrad", False))

    # Accumulate the estimator in float32 for fp16/bf16 parameters. Keeping float64 parameters in
    # float64 makes the helper precise enough for optimizer-equivalence tests.
    compute_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
    grad = gradient.detach().to(dtype=compute_dtype)
    if maximize:
        grad = -grad

    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    exp_avg_value = torch.zeros_like(grad) if exp_avg is None else exp_avg.detach().to(dtype=compute_dtype)
    exp_avg_sq_value = (
        torch.zeros_like(grad) if exp_avg_sq is None else exp_avg_sq.detach().to(dtype=compute_dtype)
    )

    next_exp_avg = exp_avg_value.mul(beta1).add(grad, alpha=1.0 - beta1)
    next_exp_avg_sq = exp_avg_sq_value.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)

    raw_step = state.get("step", 0)
    current_step = int(raw_step.item()) if isinstance(raw_step, Tensor) else int(raw_step)
    next_step = current_step + 1
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step

    variance = next_exp_avg_sq
    if amsgrad:
        max_exp_avg_sq = state.get("max_exp_avg_sq")
        if max_exp_avg_sq is not None:
            variance = torch.maximum(max_exp_avg_sq.detach().to(dtype=compute_dtype), variance)

    denominator = variance.sqrt().div(math.sqrt(bias_correction2)).add(eps)
    adaptive_update = next_exp_avg.div(bias_correction1).div(denominator)
    delta = adaptive_update.add(parameter.detach().to(dtype=compute_dtype), alpha=weight_decay)
    return delta.mul(-learning_rate)


def candidate_update_projection(
    probe_gradient: Tensor,
    parameter: nn.Parameter,
    param_group: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Tensor:
    """Return ``<probe_gradient, candidate AdamW delta>`` for one parameter."""
    delta = adamw_candidate_parameter_delta(parameter, param_group, state)
    if delta is None:
        return torch.zeros((), dtype=torch.float32, device=probe_gradient.device)
    return torch.sum(probe_gradient.detach().to(dtype=delta.dtype) * delta)


def update_cabo_budget(
    action_group: dict[str, Any],
    *,
    vlm_drift: float,
    action_drift: float,
    action_drift_ratio: float,
    probe_interval: int,
    ema_decay: float,
    budget_decay: float,
    budget_cap_windows: float,
) -> tuple[float, dict[str, float]]:
    """Update CABO's leaky functional-drift budget and return the next action scale."""
    finite = math.isfinite(vlm_drift) and math.isfinite(action_drift)
    if not finite or vlm_drift < 0.0 or action_drift < 0.0:
        action_group["cabo_action_scale"] = 0.0
        return 0.0, {
            "cabo/vlm_drift": vlm_drift,
            "cabo/action_drift": action_drift,
            "cabo/action_scale": 0.0,
            "cabo/budget": float(action_group.get("cabo_budget", 0.0)),
            "cabo/probe_nonfinite": 1.0,
        }

    initialized = bool(action_group.get("cabo_ema_initialized", False))
    if initialized:
        vlm_ema = ema_decay * float(action_group["cabo_vlm_drift_ema"]) + (1.0 - ema_decay) * vlm_drift
        action_ema = (
            ema_decay * float(action_group["cabo_action_drift_ema"]) + (1.0 - ema_decay) * action_drift
        )
    else:
        vlm_ema = vlm_drift
        action_ema = action_drift

    interval = float(probe_interval)
    credit = interval * action_drift_ratio**2 * vlm_ema
    old_budget = float(action_group.get("cabo_budget", 0.0))
    budget = budget_decay * old_budget + credit
    # Express the cap in windows of current EMA credit. A tiny floor keeps the state finite when
    # both modules are locally stationary without granting a meaningful extra action update.
    credit_floor = torch.finfo(torch.float64).tiny
    budget_cap = budget_cap_windows * max(credit, credit_floor)
    budget = min(budget, budget_cap)

    full_action_cost = interval * action_ema
    if full_action_cost <= 0.0:
        action_scale = 1.0
        spent = 0.0
    else:
        action_scale = min(1.0, math.sqrt(max(budget, 0.0) / full_action_cost))
        spent = action_scale**2 * full_action_cost
    budget = max(0.0, budget - spent)

    action_group.update(
        {
            "cabo_ema_initialized": True,
            "cabo_vlm_drift_ema": vlm_ema,
            "cabo_action_drift_ema": action_ema,
            "cabo_budget": budget,
            "cabo_action_scale": action_scale,
        }
    )
    return action_scale, {
        "cabo/vlm_drift": vlm_drift,
        "cabo/action_drift": action_drift,
        "cabo/vlm_drift_ema": vlm_ema,
        "cabo/action_drift_ema": action_ema,
        "cabo/action_scale": action_scale,
        "cabo/budget": budget,
        "cabo/credit": credit,
        "cabo/planned_action_cost": spent,
        "cabo/probe_nonfinite": 0.0,
    }


@contextmanager
def temporary_optimizer_group_lr_scales(
    optimizer: Optimizer,
    group_scales: Mapping[str, float],
) -> Iterator[dict[str, float]]:
    """Temporarily scale named group learning rates for exactly one optimizer step."""
    if not group_scales:
        yield {}
        return

    unwrapped = unwrap_optimizer(optimizer)
    original_lrs: dict[str, float | Tensor] = {}
    effective_lrs: dict[str, float] = {}
    try:
        for name, scale in group_scales.items():
            if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError(f"CABO optimizer group scale must be in [0, 1], got {scale} for {name!r}")
            group = get_named_param_group(unwrapped, name)
            original_lrs[name] = group["lr"]
            group["lr"] = group["lr"] * scale
            effective_lrs[name] = float(group["lr"])
        yield effective_lrs
    finally:
        for name, learning_rate in original_lrs.items():
            get_named_param_group(unwrapped, name)["lr"] = learning_rate
