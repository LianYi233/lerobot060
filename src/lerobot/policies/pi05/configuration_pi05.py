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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

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
    optimizer_lr: float = 2.5e-4
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

    # Experimental Counterfactual Action-Budget Optimization (CABO). CABO estimates the joint
    # flow-velocity drift of the next VLM and action-side AdamW updates, including their cross term,
    # then scales only the action-side optimizer step to stay inside a leaky functional budget.
    cabo_enabled: bool = False
    cabo_action_drift_ratio: float = 0.1
    cabo_probe_interval: int = 8
    cabo_probe_batch_size: int = 1
    cabo_num_projections: int = 4
    # Dimensionless action-only allowance. When VLM drift is zero, this grants approximately this
    # fraction of the full candidate action step instead of starving the action side completely.
    cabo_base_action_scale: float = 0.1
    # Positive cross drift is charged fully. Negative (cancelling) cross drift receives only this
    # fraction of its measured credit to avoid trusting noisy cancellation estimates too strongly.
    cabo_negative_cross_discount: float = 0.5
    cabo_drift_ema_decay: float = 0.9
    cabo_budget_decay: float = 0.95
    cabo_budget_cap_windows: float = 4.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-5

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
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

        if not 0.0 < self.cabo_action_drift_ratio <= 1.0:
            raise ValueError(f"cabo_action_drift_ratio must be in (0, 1], got {self.cabo_action_drift_ratio}")
        if self.cabo_probe_interval <= 0:
            raise ValueError(f"cabo_probe_interval must be greater than 0, got {self.cabo_probe_interval}")
        if self.cabo_probe_batch_size <= 0:
            raise ValueError(
                f"cabo_probe_batch_size must be greater than 0, got {self.cabo_probe_batch_size}"
            )
        if self.cabo_num_projections < 2:
            raise ValueError(
                "cabo_num_projections must be at least 2 to estimate cross drift, "
                f"got {self.cabo_num_projections}"
            )
        if not 0.0 <= self.cabo_base_action_scale <= 1.0:
            raise ValueError(f"cabo_base_action_scale must be in [0, 1], got {self.cabo_base_action_scale}")
        if not 0.0 <= self.cabo_negative_cross_discount <= 1.0:
            raise ValueError(
                f"cabo_negative_cross_discount must be in [0, 1], got {self.cabo_negative_cross_discount}"
            )
        if not 0.0 <= self.cabo_drift_ema_decay < 1.0:
            raise ValueError(f"cabo_drift_ema_decay must be in [0, 1), got {self.cabo_drift_ema_decay}")
        if not 0.0 <= self.cabo_budget_decay < 1.0:
            raise ValueError(f"cabo_budget_decay must be in [0, 1), got {self.cabo_budget_decay}")
        if not math.isfinite(self.cabo_budget_cap_windows) or self.cabo_budget_cap_windows < 1.0:
            raise ValueError(
                f"cabo_budget_cap_windows must be finite and at least 1, got {self.cabo_budget_cap_windows}"
            )
        if self.cabo_enabled and self.train_expert_only:
            raise ValueError(
                "CABO requires trainable VLM parameters and is incompatible with train_expert_only=True"
            )

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
            grad_clip_norm=self.optimizer_grad_clip_norm,
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
    def reward_delta_indices(self) -> None:
        return None
