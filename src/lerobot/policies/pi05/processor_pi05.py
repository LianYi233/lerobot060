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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.piper_gripper_history import CONTEXT_DIM, CONTEXT_KEY, HISTORY_KEY, HISTORY_STEPS

# lerobot-train \
#     --dataset.repo_id=libero /media/wyn/data/10-EmbodiedAI/dataset/Gay-LIBERO_v3.0/libero_10_no_noops_1.0.0_lerobot_v3.0
#     --policy.type=pi05 --output_dir=/media/wyn/data/10-EmbodiedAI/2601-lerobot/train-libero
#     --job_name=pi05_training-baseline-327 \
#     --policy.repo_id=libero --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true --wandb.enable=true \
#     --policy.dtype=bfloat16 --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --steps=3000 \
#     --policy.device=cuda \
#     --batch_size=32


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        # state = transition.get(TransitionKey.OBSERVATION, {}).get(observation.state.arm.position)
        # state = transition.get(TransitionKey.OBSERVATION, {}).get(observation.robot_state.arm.position)
        # state = transition[TransitionKey.OBSERVATION]['state']['arm']['position']

        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi05_gripper_history_processor")
@dataclass
class Pi05GripperHistoryProcessorStep(NormalizerProcessorStep):
    """Pack normalized state + ACTION-normalized past openings + literal zero padding.

    Inherits the normalizer's checkpointed state_dict so inference uses the training
    split's action statistics, even when no labels or dataset are available.
    """

    history_steps: int = HISTORY_STEPS
    gripper_index: int = 6

    def __post_init__(self):
        super().__post_init__()
        if self.history_steps != HISTORY_STEPS or not 0 <= self.gripper_index < 7:
            raise ValueError("Expected 20 past gripper actions and a seven-coordinate action space")

    def get_config(self):
        return {
            **super().get_config(),
            "history_steps": self.history_steps,
            "gripper_index": self.gripper_index,
        }

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        state, history = observation.get(OBS_STATE), observation.get(HISTORY_KEY)
        if state is None or history is None:
            raise ValueError("History-enabled PI05 requires current state and causal past gripper actions")
        if history.ndim == 1:
            history = history.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != 7 or history.shape != (state.shape[0], HISTORY_STEPS):
            raise ValueError("Expected state [B,7] and past gripper actions [B,20]")
        if ACTION not in self._tensor_stats:
            raise ValueError("History processor is missing checkpointed training action statistics")
        history = history.to(device=state.device, dtype=state.dtype)
        # Use the same transform as target actions; other action coordinates are discarded.
        action_rows = state.new_zeros((*history.shape, 7))
        action_rows[..., self.gripper_index] = history
        normalized = self._apply_transform(action_rows, ACTION, FeatureType.ACTION)[..., self.gripper_index]
        context = torch.cat([state, normalized, state.new_zeros((state.shape[0], 5))], dim=-1)
        if context.shape[-1] != CONTEXT_DIM or not torch.isfinite(context).all():
            raise ValueError("Nonfinite Piper state/history context")
        observation[CONTEXT_KEY] = context
        return {**transition, TransitionKey.OBSERVATION: observation}


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = None
    if config.use_relative_actions:
        if not config.action_feature_names or not config.state_feature_names:
            raise ValueError("PI05 relative actions require saved state/action feature names")
        relative_step = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=["gripper.pos"],
            action_names=config.action_feature_names,
            state_names=config.state_feature_names,
        )

    history_steps = []
    if config.piper_gripper_history_steps:
        if not config.action_feature_names or not config.state_feature_names:
            raise ValueError("History conditioning requires saved Piper coordinate names")
        history_steps.append(
            Pi05GripperHistoryProcessorStep(
                features={ACTION: config.output_features[ACTION]},
                norm_map=config.normalization_mapping,
                stats={ACTION: dataset_stats[ACTION]} if dataset_stats and ACTION in dataset_stats else None,
                gripper_index=config.action_feature_names.index("gripper.pos"),
            )
        )

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        *([relative_step] if relative_step is not None else []),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        *history_steps,
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        *(
            [AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)]
            if relative_step is not None
            else []
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def validate_pi05_action_processors(config, preprocessor, postprocessor):
    """Fail before inference if checkpoint config and serialized action representation disagree."""
    history = [s for s in preprocessor.steps if isinstance(s, Pi05GripperHistoryProcessorStep)]
    if bool(history) != bool(config.piper_gripper_history_steps) or len(history) > 1:
        raise ValueError(
            "PI05 config and saved gripper history processor disagree; do not patch config alone"
        )
    if history:
        step = history[0]
        if not config.action_feature_names or step.gripper_index != config.action_feature_names.index(
            "gripper.pos"
        ):
            raise ValueError("History processor gripper coordinate differs from saved action order")
        normalizer = next(s for s in preprocessor.steps if type(s) is NormalizerProcessorStep)
        if preprocessor.steps.index(step) < preprocessor.steps.index(normalizer):
            raise ValueError("History context must be packed after state normalization")
        stats = step._tensor_stats.get(ACTION, {})
        reference = normalizer._tensor_stats.get(ACTION, {})
        if (
            not stats
            or stats.keys() != reference.keys()
            or step.norm_map != normalizer.norm_map
            or any(not torch.equal(value.cpu(), reference[key].cpu()) for key, value in stats.items())
        ):
            raise ValueError("History and target action normalization statistics must match")
    relative = [s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled]
    absolute = [s for s in postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled]
    if not config.use_relative_actions:
        if relative or absolute:
            raise ValueError("Absolute PI05 config cannot load relative action processors")
        return
    if len(relative) != 1 or len(absolute) != 1:
        raise ValueError("Relative PI05 checkpoint requires paired relative/absolute action processors")
    step = relative[0]
    if (
        not config.action_feature_names
        or not config.state_feature_names
        or step.action_names != config.action_feature_names
        or step.state_names != config.state_feature_names
        or step.exclude_joints != ["gripper.pos"]
        or absolute[0].relative_step is not step
    ):
        raise ValueError("Relative PI05 processors do not match saved coordinate order/gripper convention")
    normalizer = next(s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep))
    unnormalizer = next(s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep))
    if preprocessor.steps.index(step) > preprocessor.steps.index(normalizer) or postprocessor.steps.index(
        absolute[0]
    ) < postprocessor.steps.index(unnormalizer):
        raise ValueError("Relative actions must be converted in physical units outside normalization")
