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

"""Optimizer-side helpers for counterfactual action-space update control (CABO)."""

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

CABO_VLM_GROUP = "vlm"
CABO_ACTION_GROUP = "action"
CABO_GROUP_NAME = "name"


@dataclass
class OptimizerStepControl:
    """Policy-provided controls for one optimizer step.

    ``skip_optimizer_step`` is deliberately separate from setting a learning rate to zero: AdamW
    still advances its moments and step counter when its learning rate is zero.
    """

    group_scales: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    skip_optimizer_step: bool = False


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


def validate_adamw_param_group(param_group: Mapping[str, Any]) -> None:
    """Reject AdamW execution modes that the native-dtype CABO simulator cannot reproduce."""
    unsupported_modes = [
        name
        for name in ("differentiable", "fused", "capturable", "foreach")
        if bool(param_group.get(name, False))
    ]
    if unsupported_modes:
        modes = ", ".join(unsupported_modes)
        raise RuntimeError(f"CABO does not support these AdamW modes: {modes}")


@torch.no_grad()
def adamw_candidate_parameter_delta(
    parameter: nn.Parameter,
    param_group: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Tensor | None:
    """Compute the next AdamW parameter delta without mutating parameters or optimizer state.

    The result includes decoupled weight decay and is evaluated at the group's current, unscaled
    learning rate. Optimizer-state math and the candidate parameter write use their native dtypes,
    so bf16/fp16 quantization is represented at scale 1. CABO still treats learning-rate scaling as
    linear after this point; parameter quantization means that approximation is not exact for scales
    other than 1.
    """
    gradient = parameter.grad
    if gradient is None:
        return None
    if gradient.is_sparse:
        raise RuntimeError("CABO does not support sparse AdamW gradients")
    if torch.is_complex(parameter) or torch.is_complex(gradient):
        raise RuntimeError("CABO does not support complex AdamW parameters")
    validate_adamw_param_group(param_group)

    beta1, beta2 = param_group["betas"]
    eps = float(param_group["eps"])
    learning_rate = float(param_group["lr"])
    weight_decay = float(param_group["weight_decay"])
    maximize = bool(param_group.get("maximize", False))
    amsgrad = bool(param_group.get("amsgrad", False))

    grad = gradient.detach()
    if maximize:
        grad = -grad

    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    next_exp_avg = torch.zeros_like(parameter) if exp_avg is None else exp_avg.detach().clone()
    next_exp_avg_sq = torch.zeros_like(parameter) if exp_avg_sq is None else exp_avg_sq.detach().clone()
    if grad.dtype != next_exp_avg.dtype:
        raise RuntimeError(
            "CABO requires AdamW gradients and optimizer moments to share a dtype, "
            f"got gradient={grad.dtype}, exp_avg={next_exp_avg.dtype}"
        )

    # Match torch AdamW's native-dtype single-tensor update, including its use of lerp for the first
    # moment. This matters for bf16/fp16 rounding.
    next_exp_avg.lerp_(grad, 1.0 - beta1)
    next_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    raw_step = state.get("step", 0)
    current_step = int(raw_step.item()) if isinstance(raw_step, Tensor) else int(raw_step)
    next_step = current_step + 1
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step

    variance = next_exp_avg_sq
    if amsgrad:
        max_exp_avg_sq = state.get("max_exp_avg_sq")
        previous_max = (
            torch.zeros_like(parameter) if max_exp_avg_sq is None else max_exp_avg_sq.detach().clone()
        )
        variance = torch.maximum(previous_max, variance)

    denominator = variance.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    candidate = parameter.detach().clone()
    candidate.mul_(1.0 - learning_rate * weight_decay)
    candidate.addcdiv_(next_exp_avg, denominator, value=-learning_rate / bias_correction1)

    # The candidate has already been quantized by the native-dtype parameter write. Subtract its
    # representable value from the old representable value in a wider dtype so the returned delta
    # does not undergo an additional low-precision rounding.
    result_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
    return candidate.to(dtype=result_dtype).sub(parameter.detach().to(dtype=result_dtype))


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
    cross_drift: float,
    action_drift_ratio: float,
    base_action_scale: float,
    negative_cross_discount: float,
    probe_interval: int,
    ema_decay: float,
    budget_decay: float,
    budget_cap_windows: float,
) -> tuple[float, dict[str, float]]:
    """Update CABO's leaky joint-drift budget and return the next action scale.

    Let ``dv`` and ``da`` be the first-order flow-velocity changes induced by the full VLM and
    action-side AdamW candidate updates. Scaling only the action update by ``s`` gives

        mean(||dv + s * da||^2) = Dv + 2 * s * Cva + s^2 * Da.

    With ``base_action_scale=0``, the relative allowance preserves the old linearized worst-case
    guarantee that total RMS drift is at most ``1 + action_drift_ratio`` times the VLM-only RMS
    drift, but uses the measured cross term instead of assuming that both updates are perfectly
    aligned. A positive ``base_action_scale`` deliberately relaxes that relative bound by granting
    an action-only allowance of ``base_action_scale^2 * Da`` so a stationary VLM cannot starve the
    action side completely.
    """
    finite = math.isfinite(vlm_drift) and math.isfinite(action_drift) and math.isfinite(cross_drift)
    if not finite or vlm_drift < 0.0 or action_drift < 0.0:
        # Do not mutate controller state. The caller must skip the complete optimizer step; setting
        # only the action learning rate to zero would still advance AdamW state and leave the VLM
        # update unprotected.
        previous_scale = float(action_group.get("cabo_action_scale", 1.0))
        return previous_scale, {
            "cabo/vlm_drift": vlm_drift,
            "cabo/action_drift": action_drift,
            "cabo/cross_drift": cross_drift,
            "cabo/action_scale": previous_scale,
            "cabo/budget": float(action_group.get("cabo_budget", 0.0)),
            "cabo/probe_nonfinite": 1.0,
        }

    initialized = bool(action_group.get("cabo_ema_initialized", False))
    if initialized:
        vlm_ema = ema_decay * float(action_group["cabo_vlm_drift_ema"]) + (1.0 - ema_decay) * vlm_drift
        action_ema = (
            ema_decay * float(action_group["cabo_action_drift_ema"]) + (1.0 - ema_decay) * action_drift
        )
        cross_ema = (
            ema_decay * float(action_group.get("cabo_cross_drift_ema", cross_drift))
            + (1.0 - ema_decay) * cross_drift
        )
    else:
        vlm_ema = vlm_drift
        action_ema = action_drift
        cross_ema = cross_drift

    # These three statistics form an EMA Gram matrix. Clamp only round-off violations of its
    # Cauchy-Schwarz bound; a materially invalid value would otherwise make the quadratic controller
    # grant a fictitious cancellation credit.
    cross_limit = math.sqrt(max(vlm_ema * action_ema, 0.0))
    cross_ema = min(max(cross_ema, -cross_limit), cross_limit)
    effective_cross = cross_ema if cross_ema >= 0.0 else negative_cross_discount * cross_ema

    interval = float(probe_interval)
    # If the old marginal constraint s * sqrt(Da) <= rho * sqrt(Dv) is interpreted as a
    # worst-case bound, it guarantees total RMS drift <= (1 + rho) * sqrt(Dv). The exact joint
    # quadratic therefore has an incremental squared-drift allowance of (2*rho + rho^2) * Dv.
    relative_allowance = (2.0 * action_drift_ratio + action_drift_ratio**2) * vlm_ema
    base_allowance = base_action_scale**2 * action_ema
    credit = interval * (relative_allowance + base_allowance)
    old_budget = float(action_group.get("cabo_budget", 0.0))
    budget = budget_decay * old_budget + credit
    # Express the cap in windows of current EMA credit. A tiny floor keeps the state finite when
    # both modules are locally stationary without granting a meaningful extra action update.
    credit_floor = torch.finfo(torch.float64).tiny
    budget_cap = budget_cap_windows * max(credit, credit_floor)
    budget = min(budget, budget_cap)

    allowance = max(budget, 0.0) / interval
    if action_ema <= 0.0:
        action_scale = 1.0
    else:
        discriminant = effective_cross**2 + action_ema * allowance
        action_scale = min(1.0, (-effective_cross + math.sqrt(max(discriminant, 0.0))) / action_ema)

    # Negative incremental drift means the action update cancels part of the VLM update. It may
    # spend no budget, but it must never mint new budget from that cancellation.
    incremental_cost = 2.0 * action_scale * effective_cross + action_scale**2 * action_ema
    spent = interval * max(0.0, incremental_cost)
    budget = max(0.0, budget - spent)

    total_drift_ema = vlm_ema + 2.0 * action_scale * cross_ema + action_scale**2 * action_ema
    cross_correlation = cross_ema / cross_limit if cross_limit > 0.0 else 0.0

    action_group.update(
        {
            "cabo_ema_initialized": True,
            "cabo_vlm_drift_ema": vlm_ema,
            "cabo_action_drift_ema": action_ema,
            "cabo_cross_drift_ema": cross_ema,
            "cabo_budget": budget,
            "cabo_action_scale": action_scale,
        }
    )
    return action_scale, {
        "cabo/vlm_drift": vlm_drift,
        "cabo/action_drift": action_drift,
        "cabo/cross_drift": cross_drift,
        "cabo/vlm_drift_ema": vlm_ema,
        "cabo/action_drift_ema": action_ema,
        "cabo/cross_drift_ema": cross_ema,
        "cabo/effective_cross_drift_ema": effective_cross,
        "cabo/cross_correlation_ema": cross_correlation,
        "cabo/total_drift_ema": total_drift_ema,
        "cabo/action_scale": action_scale,
        "cabo/budget": budget,
        "cabo/credit": credit,
        "cabo/relative_allowance": relative_allowance,
        "cabo/base_action_allowance": base_allowance,
        "cabo/planned_action_cost": spent,
        "cabo/probe_nonfinite": 0.0,
    }


def update_cabo_residual_compensation(
    controller_group: dict[str, Any],
    *,
    vlm_drift: float,
    action_drift: float,
    cross_drift: float,
    residual_energy: float,
    residual_vlm_alignment: float,
    residual_action_alignment: float,
    ema_decay: float,
    regularization: float,
    max_vlm_scale: float,
) -> tuple[float, dict[str, float]]:
    """Scale the VLM update to reduce the residual left by a full action-side update.

    Let ``e = predicted_velocity - target_velocity`` and let ``dv`` and ``da`` be the
    linearized velocity changes induced by the next VLM and action-side AdamW updates. CABO keeps
    the action-side update at full scale and chooses a non-negative VLM scale ``s`` by minimizing

        mean(||e + da + s * dv||^2) + regularization * s^2 * mean(||dv||^2).

    The closed-form unconstrained minimizer is

        s = -mean(<e + da, dv>) / ((1 + regularization) * mean(||dv||^2)).

    Clamping to ``[0, max_vlm_scale]`` means the VLM is updated only when its candidate step is
    predicted to improve the action-side residual. Unlike magnitude balancing, a weak or large VLM
    influence receives no reward unless it points in a useful direction.
    """
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError(
            f"CABO residual regularization must be finite and non-negative, got {regularization}"
        )
    if not math.isfinite(max_vlm_scale) or max_vlm_scale < 0.0:
        raise ValueError(f"CABO residual max VLM scale must be finite and non-negative, got {max_vlm_scale}")

    statistics = (
        vlm_drift,
        action_drift,
        cross_drift,
        residual_energy,
        residual_vlm_alignment,
        residual_action_alignment,
    )
    finite = all(math.isfinite(value) for value in statistics)
    valid_nonnegative_moments = vlm_drift >= 0.0 and action_drift >= 0.0 and residual_energy >= 0.0
    previous_vlm_scale = float(controller_group.get("cabo_vlm_scale", 1.0))
    if not finite or not valid_nonnegative_moments:
        return previous_vlm_scale, {
            "cabo/vlm_drift": vlm_drift,
            "cabo/action_drift": action_drift,
            "cabo/cross_drift": cross_drift,
            "cabo/residual_energy": residual_energy,
            "cabo/residual_vlm_alignment": residual_vlm_alignment,
            "cabo/residual_action_alignment": residual_action_alignment,
            "cabo/vlm_scale": previous_vlm_scale,
            "cabo/action_scale": 1.0,
            "cabo/probe_nonfinite": 1.0,
        }

    # Use a mode-specific initialization bit. Legacy budget/balance checkpoints may already have
    # ``cabo_ema_initialized`` without the residual-alignment moments introduced here.
    initialized = bool(controller_group.get("cabo_residual_ema_initialized", False))

    def update_ema(key: str, value: float) -> float:
        if not initialized:
            return value
        return ema_decay * float(controller_group[key]) + (1.0 - ema_decay) * value

    vlm_ema = update_ema("cabo_vlm_drift_ema", vlm_drift)
    action_ema = update_ema("cabo_action_drift_ema", action_drift)
    cross_ema = update_ema("cabo_cross_drift_ema", cross_drift)
    residual_energy_ema = update_ema("cabo_residual_energy_ema", residual_energy)
    residual_vlm_alignment_ema = update_ema("cabo_residual_vlm_alignment_ema", residual_vlm_alignment)
    residual_action_alignment_ema = update_ema(
        "cabo_residual_action_alignment_ema", residual_action_alignment
    )

    # All six values are moments of the same projected vectors. Preserve their pairwise
    # Cauchy-Schwarz bounds after EMA in case floating-point reduction introduces tiny violations.
    cross_limit = math.sqrt(max(vlm_ema * action_ema, 0.0))
    cross_ema = min(max(cross_ema, -cross_limit), cross_limit)
    residual_vlm_limit = math.sqrt(max(residual_energy_ema * vlm_ema, 0.0))
    residual_vlm_alignment_ema = min(max(residual_vlm_alignment_ema, -residual_vlm_limit), residual_vlm_limit)
    residual_action_limit = math.sqrt(max(residual_energy_ema * action_ema, 0.0))
    residual_action_alignment_ema = min(
        max(residual_action_alignment_ema, -residual_action_limit), residual_action_limit
    )

    post_action_vlm_alignment = residual_vlm_alignment_ema + cross_ema
    raw_vlm_scale = 0.0 if vlm_ema <= 0.0 else -post_action_vlm_alignment / ((1.0 + regularization) * vlm_ema)
    vlm_scale = min(max(raw_vlm_scale, 0.0), max_vlm_scale)

    predicted_action_only_residual = max(
        residual_energy_ema + 2.0 * residual_action_alignment_ema + action_ema,
        0.0,
    )
    predicted_joint_residual = max(
        predicted_action_only_residual + 2.0 * vlm_scale * post_action_vlm_alignment + vlm_scale**2 * vlm_ema,
        0.0,
    )
    predicted_action_improvement = residual_energy_ema - predicted_action_only_residual
    predicted_vlm_improvement = predicted_action_only_residual - predicted_joint_residual
    cross_correlation = cross_ema / cross_limit if cross_limit > 0.0 else 0.0

    controller_group.update(
        {
            "cabo_ema_initialized": True,
            "cabo_residual_ema_initialized": True,
            "cabo_vlm_drift_ema": vlm_ema,
            "cabo_action_drift_ema": action_ema,
            "cabo_cross_drift_ema": cross_ema,
            "cabo_residual_energy_ema": residual_energy_ema,
            "cabo_residual_vlm_alignment_ema": residual_vlm_alignment_ema,
            "cabo_residual_action_alignment_ema": residual_action_alignment_ema,
            "cabo_vlm_scale": vlm_scale,
            # Residual mode always treats the action-side candidate as the primary, full update.
            "cabo_action_scale": 1.0,
        }
    )
    return vlm_scale, {
        "cabo/vlm_drift": vlm_drift,
        "cabo/action_drift": action_drift,
        "cabo/cross_drift": cross_drift,
        "cabo/vlm_drift_ema": vlm_ema,
        "cabo/action_drift_ema": action_ema,
        "cabo/cross_drift_ema": cross_ema,
        "cabo/cross_correlation_ema": cross_correlation,
        "cabo/residual_energy": residual_energy,
        "cabo/residual_vlm_alignment": residual_vlm_alignment,
        "cabo/residual_action_alignment": residual_action_alignment,
        "cabo/residual_energy_ema": residual_energy_ema,
        "cabo/residual_vlm_alignment_ema": residual_vlm_alignment_ema,
        "cabo/residual_action_alignment_ema": residual_action_alignment_ema,
        "cabo/post_action_vlm_alignment_ema": post_action_vlm_alignment,
        "cabo/predicted_action_only_residual_ema": predicted_action_only_residual,
        "cabo/predicted_joint_residual_ema": predicted_joint_residual,
        "cabo/predicted_action_improvement_ema": predicted_action_improvement,
        "cabo/predicted_vlm_improvement_ema": predicted_vlm_improvement,
        "cabo/vlm_scale": vlm_scale,
        "cabo/action_scale": 1.0,
        "cabo/residual_scale_clamped": float(vlm_scale != raw_vlm_scale),
        "cabo/probe_nonfinite": 0.0,
    }


def update_cabo_influence_balance(
    controller_group: dict[str, Any],
    *,
    vlm_drift: float,
    action_drift: float,
    cross_drift: float,
    ema_decay: float,
    max_scale: float,
) -> tuple[float, float, dict[str, float]]:
    """Balance direction-agnostic VLM and action influence in action space.

    ``vlm_drift`` and ``action_drift`` estimate the mean squared flow-velocity changes caused by
    the next full AdamW update for each parameter group. Let ``sv`` and ``sa`` scale the VLM and
    action updates. The log-symmetric scales

        sv = (Da / Dv) ** 1/4
        sa = (Dv / Da) ** 1/4

    satisfy ``sv**2 * Dv == sa**2 * Da`` while preserving ``sv * sa == 1``. Thus the weaker group
    is amplified by the reciprocal of the attenuation applied to the stronger group. The cross
    drift is deliberately excluded from the controller because this mode balances marginal
    magnitudes regardless of whether the two functional updates align or cancel; it is retained for
    diagnostics and total-drift reporting.

    ``max_scale`` bounds amplification and implies a reciprocal lower bound for attenuation. Exact
    equality is impossible with a finite scale when either marginal drift is zero, so the controller
    moves as far toward equality as this bound permits.
    """
    if not math.isfinite(max_scale) or max_scale < 1.0:
        raise ValueError(f"CABO balance max_scale must be finite and at least 1, got {max_scale}")

    finite = math.isfinite(vlm_drift) and math.isfinite(action_drift) and math.isfinite(cross_drift)
    previous_vlm_scale = float(controller_group.get("cabo_vlm_scale", 1.0))
    previous_action_scale = float(controller_group.get("cabo_action_scale", 1.0))
    if not finite or vlm_drift < 0.0 or action_drift < 0.0:
        return (
            previous_vlm_scale,
            previous_action_scale,
            {
                "cabo/vlm_drift": vlm_drift,
                "cabo/action_drift": action_drift,
                "cabo/cross_drift": cross_drift,
                "cabo/vlm_scale": previous_vlm_scale,
                "cabo/action_scale": previous_action_scale,
                "cabo/probe_nonfinite": 1.0,
            },
        )

    initialized = bool(controller_group.get("cabo_ema_initialized", False))
    if initialized:
        vlm_ema = ema_decay * float(controller_group["cabo_vlm_drift_ema"]) + (1.0 - ema_decay) * vlm_drift
        action_ema = (
            ema_decay * float(controller_group["cabo_action_drift_ema"]) + (1.0 - ema_decay) * action_drift
        )
        cross_ema = (
            ema_decay * float(controller_group.get("cabo_cross_drift_ema", cross_drift))
            + (1.0 - ema_decay) * cross_drift
        )
    else:
        vlm_ema = vlm_drift
        action_ema = action_drift
        cross_ema = cross_drift

    cross_limit = math.sqrt(max(vlm_ema * action_ema, 0.0))
    cross_ema = min(max(cross_ema, -cross_limit), cross_limit)

    max_log_scale = math.log(max_scale)
    if vlm_ema == 0.0 and action_ema == 0.0:
        raw_log_vlm_scale = 0.0
    elif vlm_ema == 0.0:
        raw_log_vlm_scale = math.inf
    elif action_ema == 0.0:
        raw_log_vlm_scale = -math.inf
    else:
        # One quarter appears because the measured drifts are squared influence magnitudes.
        raw_log_vlm_scale = 0.25 * (math.log(action_ema) - math.log(vlm_ema))

    log_vlm_scale = min(max(raw_log_vlm_scale, -max_log_scale), max_log_scale)
    vlm_scale = math.exp(log_vlm_scale)
    action_scale = math.exp(-log_vlm_scale)

    balanced_vlm_drift = vlm_scale**2 * vlm_ema
    balanced_action_drift = action_scale**2 * action_ema
    tiny = torch.finfo(torch.float64).tiny
    balanced_influence_ratio = math.sqrt((balanced_action_drift + tiny) / (balanced_vlm_drift + tiny))
    total_drift_ema = balanced_vlm_drift + 2.0 * vlm_scale * action_scale * cross_ema + balanced_action_drift
    cross_correlation = cross_ema / cross_limit if cross_limit > 0.0 else 0.0
    balance_clamped = abs(raw_log_vlm_scale) > max_log_scale

    controller_group.update(
        {
            "cabo_ema_initialized": True,
            "cabo_vlm_drift_ema": vlm_ema,
            "cabo_action_drift_ema": action_ema,
            "cabo_cross_drift_ema": cross_ema,
            "cabo_vlm_scale": vlm_scale,
            "cabo_action_scale": action_scale,
        }
    )
    return (
        vlm_scale,
        action_scale,
        {
            "cabo/vlm_drift": vlm_drift,
            "cabo/action_drift": action_drift,
            "cabo/cross_drift": cross_drift,
            "cabo/vlm_drift_ema": vlm_ema,
            "cabo/action_drift_ema": action_ema,
            "cabo/cross_drift_ema": cross_ema,
            "cabo/cross_correlation_ema": cross_correlation,
            "cabo/balanced_vlm_drift_ema": balanced_vlm_drift,
            "cabo/balanced_action_drift_ema": balanced_action_drift,
            "cabo/balanced_influence_ratio": balanced_influence_ratio,
            "cabo/total_drift_ema": total_drift_ema,
            "cabo/vlm_scale": vlm_scale,
            "cabo/action_scale": action_scale,
            "cabo/balance_clamped": float(balance_clamped),
            "cabo/probe_nonfinite": 0.0,
        },
    )


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
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    f"CABO optimizer group scale must be finite and non-negative, got {scale} for {name!r}"
                )
            group = get_named_param_group(unwrapped, name)
            original_lrs[name] = group["lr"]
            group["lr"] = group["lr"] * scale
            effective_lrs[name] = float(group["lr"])
        yield effective_lrs
    finally:
        for name, learning_rate in original_lrs.items():
            get_named_param_group(unwrapped, name)["lr"] = learning_rate
