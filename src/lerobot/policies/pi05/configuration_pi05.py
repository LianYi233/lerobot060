#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field
from enum import StrEnum

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


class PI05TrainingStage(StrEnum):
    """Training objectives supported by PI0.5."""

    FLOW = "flow"
    NEXT_ACTION = "next_action"


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Training objective. ``next_action`` is kept as the public name for backwards compatibility,
    # but now performs action-only flow inpainting over the complete chunk. ``flow`` keeps the
    # standard observation-conditioned PI0.5 objective.
    # A string enum preserves the two-value contract while remaining decodable by draccus CLI/config loading.
    training_stage: PI05TrainingStage = PI05TrainingStage.FLOW
    # Number of valid temporal action tokens to hide and reconstruct per sample. The mask is sampled
    # uniformly without replacement; short padded chunks mask all available valid tokens.
    next_action_masked_steps: int = 40
    # Deprecated compatibility fields retained only so checkpoints produced by the former 25-to-25
    # MSE objective remain config-loadable. Flow inpainting does not use either split.
    next_action_context_steps: int = 25
    next_action_prediction_steps: int = 25
    # Stage 1 defaults to fixed-count inpainting: 40 hidden action tokens and 10 visible tokens for
    # a standard 50-step chunk. Complete-chunk masking remains available as an explicit override.
    next_action_full_mask_probability: float = 0.0
    # Number of action-only flow steps automatically run before a flow-training invocation. Set to 0
    # to start flow training immediately. This is orchestration metadata and does not alter either loss.
    next_action_pretrain_steps: int = 500

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings. Action and VLM parameters share this single AdamW learning rate.
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    # Global clipping is disabled for PI05. Its policy-specific hook clips only action-side gradients
    # against the VLM gradient RMS and deliberately leaves VLM gradients unchanged.
    optimizer_grad_clip_norm: float = 0.0

    # Limit action-side gradient spikes relative to the VLM. The comparison is
    # made using the RMS gradient over all elements in each parameter group:
    #     rms(g) = ||g||_2 / sqrt(number of gradient elements)
    # The default ratio of 10.0 enforces action_rms <= 10 * vlm_rms.
    clip_action_head_by_vlm: bool = True
    action_head_grad_clip_ratio: float = 10.0

    # Relative-update optimizer control (CABO). CABO deterministically computes the next AdamW
    # learning update for each parameter group, keeps the VLM update at full scale, and limits the
    # action expert/projection relative update rates against an EMA of the VLM relative update rate.
    # Weight decay is excluded from the measured learning rates so it is not mistaken for VLM signal.
    cabo_enabled: bool = True
    cabo_expert_update_ratio: float = 2.0
    cabo_projection_update_ratio: float = 5.0
    cabo_vlm_update_ema_decay: float = 0.95
    # Keep both action groups unrestricted while collecting a stable VLM reference.
    cabo_update_warmup_steps: int = 100
    # After warmup, retain this fraction of the warmup-average VLM rate as a reference floor.
    cabo_vlm_update_floor_ratio: float = 0.1

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-5

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        try:
            self.training_stage = PI05TrainingStage(self.training_stage)
        except ValueError as exc:
            raise ValueError(
                f"training_stage must be 'flow' or 'next_action', got {self.training_stage!r}"
            ) from exc
        if self.next_action_pretrain_steps < 0:
            raise ValueError(
                f"next_action_pretrain_steps must be non-negative, got {self.next_action_pretrain_steps}"
            )
        if not 0.0 <= self.next_action_full_mask_probability <= 1.0:
            raise ValueError(
                "next_action_full_mask_probability must be in [0, 1], "
                f"got {self.next_action_full_mask_probability}"
            )
        if self.training_stage == "next_action":
            if self.next_action_masked_steps <= 0:
                raise ValueError(
                    "next_action_masked_steps must be greater than 0 for next_action training, "
                    f"got {self.next_action_masked_steps}"
                )
            if (
                self.next_action_full_mask_probability < 1.0
                and self.next_action_masked_steps > self.chunk_size
            ):
                raise ValueError(
                    "next_action_masked_steps cannot exceed chunk_size for next_action training, "
                    f"got {self.next_action_masked_steps} > {self.chunk_size}"
                )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if not math.isfinite(self.action_head_grad_clip_ratio) or self.action_head_grad_clip_ratio <= 0.0:
            raise ValueError(
                f"action_head_grad_clip_ratio must be greater than 0, got {self.action_head_grad_clip_ratio}"
            )

        for name, value in (
            ("cabo_expert_update_ratio", self.cabo_expert_update_ratio),
            ("cabo_projection_update_ratio", self.cabo_projection_update_ratio),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than 0, got {value}")
        if not 0.0 <= self.cabo_vlm_update_ema_decay < 1.0:
            raise ValueError(
                f"cabo_vlm_update_ema_decay must be in [0, 1), got {self.cabo_vlm_update_ema_decay}"
            )
        if self.cabo_update_warmup_steps < 0:
            raise ValueError(
                f"cabo_update_warmup_steps must be non-negative, got {self.cabo_update_warmup_steps}"
            )
        if not 0.0 <= self.cabo_vlm_update_floor_ratio <= 1.0:
            raise ValueError(
                f"cabo_vlm_update_floor_ratio must be in [0, 1], got {self.cabo_vlm_update_floor_ratio}"
            )
        if self.cabo_active and self.train_expert_only:
            raise ValueError(
                "CABO requires trainable VLM parameters and is incompatible with train_expert_only=True"
            )

    @property
    def cabo_active(self) -> bool:
        """Whether CABO participates in the current training objective."""
        return self.cabo_enabled and self.training_stage == "flow"

    @property
    def next_action_pretraining_active(self) -> bool:
        """Whether a flow-training invocation should first run action-only flow pretraining."""
        return self.training_stage == "flow" and self.next_action_pretrain_steps > 0

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            # Next-action pretraining deliberately performs no gradient clipping, even if a config
            # loaded from another run carries a non-zero flow-training clipping value.
            grad_clip_norm=0.0 if self.training_stage == "next_action" else self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def drop_n_last_frames(self) -> int:
        """Keep padded tails: inpainting samples only from valid actions in each chunk."""
        return 0

    @property
    def reward_delta_indices(self) -> None:
        return None
