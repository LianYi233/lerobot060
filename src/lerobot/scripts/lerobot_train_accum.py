#!/usr/bin/env python

"""Gradient-accumulation entry point for LeRobot training.

This wrapper keeps the public ``--steps`` semantics as *optimizer updates* while
accumulating several already-sharded DataLoader batches before each optimizer
step.  It is intentionally implemented as a thin wrapper around
``lerobot_train`` so PI0.5 Stage-I/Bridge/Joint scheduling, checkpoint cadence,
CABO/CAMB update control, and LR scheduling continue to operate in optimizer
step units.

Set ``LEROBOT_GRAD_ACCUM_STEPS`` in the environment (default: 1), then launch
this module with accelerate, e.g.::

    LEROBOT_GRAD_ACCUM_STEPS=8 accelerate launch --multi_gpu \
      --num_processes=2 --mixed_precision=bf16 \
      -m lerobot.scripts.lerobot_train_accum ...

For a DataLoader batch size B, world size W, and accumulation A, the effective
global batch is B * W * A.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import torch

from lerobot.scripts import lerobot_train as base
from lerobot.utils.utils import cycle as _base_cycle


_GRAD_ACCUM_STEPS = int(os.environ.get("LEROBOT_GRAD_ACCUM_STEPS", "1"))
if _GRAD_ACCUM_STEPS < 1:
    raise ValueError(
        f"LEROBOT_GRAD_ACCUM_STEPS must be >= 1, got {_GRAD_ACCUM_STEPS}"
    )


class _AccumulatingPreprocessor:
    """Apply the existing policy preprocessor independently to each micro-batch."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    @staticmethod
    def _convert_uint8_images(batch: dict[str, Any]) -> dict[str, Any]:
        # The stock trainer performs this conversion immediately before the
        # preprocessor.  Once ``cycle`` groups micro-batches into a list that
        # outer conversion is intentionally bypassed, so reproduce it here.
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and (key.startswith("observation.images.") or key.startswith("observation.image"))
            ):
                batch[key] = value.to(dtype=torch.float32) / 255.0
        return batch

    def __call__(self, batch):
        if isinstance(batch, list):
            return [
                self.wrapped(self._convert_uint8_images(micro_batch))
                for micro_batch in batch
            ]
        return self.wrapped(batch)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


_original_make_pre_post_processors = base.make_pre_post_processors


def _make_pre_post_processors_with_accum(*args, **kwargs):
    preprocessor, postprocessor = _original_make_pre_post_processors(*args, **kwargs)
    if _GRAD_ACCUM_STEPS == 1:
        return preprocessor, postprocessor
    return _AccumulatingPreprocessor(preprocessor), postprocessor


def _accumulating_cycle(iterable):
    """Yield one optimizer-step worth of DataLoader micro-batches at a time."""
    iterator = _base_cycle(iterable)
    if _GRAD_ACCUM_STEPS == 1:
        while True:
            yield next(iterator)
    else:
        while True:
            yield [next(iterator) for _ in range(_GRAD_ACCUM_STEPS)]


_original_compute_sampler_state = base.compute_sampler_state


def _compute_sampler_state_with_accum(
    step: int,
    num_frames: int,
    batch_size: int,
    num_processes: int,
) -> dict:
    """Resume at the sample offset consumed by ``step`` optimizer updates."""
    return _original_compute_sampler_state(
        step * _GRAD_ACCUM_STEPS,
        num_frames,
        batch_size,
        num_processes,
    )


def _update_policy_with_accum(
    train_metrics,
    policy,
    batch,
    optimizer,
    grad_clip_norm: float,
    accelerator,
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
):
    """Accumulate micro-batch gradients and perform exactly one optimizer step."""
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    micro_batches = batch if isinstance(batch, list) else [batch]
    n_micro = len(micro_batches)
    if n_micro != _GRAD_ACCUM_STEPS:
        raise RuntimeError(
            f"Expected {_GRAD_ACCUM_STEPS} micro-batches, received {n_micro}."
        )

    loss_sum = 0.0
    output_dict = None
    last_batch = micro_batches[-1]

    for micro_idx, micro_batch in enumerate(micro_batches):
        sample_weights = None
        weight_stats = None
        if sample_weighter is not None:
            sample_weights, weight_stats = sample_weighter.compute_batch_weights(micro_batch)

        # Suppress DDP gradient synchronization for all but the final
        # micro-batch. The final backward synchronizes the accumulated gradient.
        should_sync = micro_idx == n_micro - 1
        sync_context = (
            nullcontext()
            if should_sync or not base.has_method(accelerator, "no_sync")
            else accelerator.no_sync(policy)
        )

        with sync_context:
            with accelerator.autocast():
                if sample_weights is not None:
                    per_sample_loss, micro_output = policy.forward(
                        micro_batch, reduction="none"
                    )
                    epsilon = 1e-6
                    loss = (per_sample_loss * sample_weights).sum() / (
                        sample_weights.sum() + epsilon
                    )
                    if micro_output is None:
                        micro_output = {}
                    if weight_stats:
                        for key, value in weight_stats.items():
                            micro_output[f"sample_weight_{key}"] = value
                else:
                    loss, micro_output = policy.forward(micro_batch)

            if sample_weights is None:
                loss = base._normalize_policy_loss_for_distributed_training(
                    policy=policy,
                    batch=micro_batch,
                    loss=loss,
                    accelerator=accelerator,
                )

            # Average, rather than sum, the accumulated objective so its gradient
            # scale matches a single effective batch of B * W * A examples.
            accelerator.backward(loss / n_micro)

        loss_sum += float(loss.detach().item())
        if micro_output is not None:
            output_dict = micro_output

    # All gradient-dependent controls run once, on the fully accumulated and
    # synchronized gradient. This is essential for PI0.5 action/VLM clipping
    # and CABO/CAMB relative-update control.
    grad_norm, clip_metrics = base._clip_policy_gradients(
        policy=policy,
        optimizer=optimizer,
        grad_clip_norm=grad_clip_norm,
        accelerator=accelerator,
    )
    if clip_metrics:
        if output_dict is None:
            output_dict = {}
        output_dict.update(clip_metrics)

    optimizer_step_control = base._prepare_policy_optimizer_step_control(
        policy=policy,
        batch=last_batch,
        optimizer=optimizer,
        accelerator=accelerator,
        grad_norm=grad_norm,
    )
    if optimizer_step_control.metrics:
        if output_dict is None:
            output_dict = {}
        output_dict.update(optimizer_step_control.metrics)

    if not optimizer_step_control.skip_optimizer_step:
        with (
            lock if lock is not None else nullcontext(),
            base.temporary_optimizer_group_lr_scales(
                optimizer, optimizer_step_control.group_scales
            ),
        ):
            optimizer.step()
    else:
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None and not optimizer_step_control.skip_optimizer_step:
        lr_scheduler.step()

    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if (
        not optimizer_step_control.skip_optimizer_step
        and base.has_method(unwrapped_policy, "update")
    ):
        unwrapped_policy.update()

    train_metrics.loss = loss_sum / n_micro
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


def _install_accumulation_hooks() -> None:
    if _GRAD_ACCUM_STEPS == 1:
        return

    base.cycle = _accumulating_cycle
    base.make_pre_post_processors = _make_pre_post_processors_with_accum
    base.compute_sampler_state = _compute_sampler_state_with_accum
    base.update_policy = _update_policy_with_accum

    # Keep the formal PI0.5 checkpoint cadence expressed in optimizer updates.
    # The outer loop still advances exactly once per grouped effective batch,
    # so the existing 500-step setting remains correct and needs no scaling.
    print(
        "[gradient accumulation] "
        f"micro-batches/optimizer-step={_GRAD_ACCUM_STEPS}. "
        "CLI --steps, Stage-I/Bridge lengths, scheduler steps, CABO/CAMB warmup, "
        "and checkpoint cadence remain optimizer-step counts."
    )


def main() -> None:
    _install_accumulation_hooks()
    base.register_third_party_plugins()
    if base._remote_target_in_argv():
        import logging

        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    base.train()


if __name__ == "__main__":
    main()
