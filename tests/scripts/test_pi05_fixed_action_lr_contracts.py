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

"""Dependency-free control-flow contracts extracted from the actual PI05/trainer source.

These stdlib tests execute the production hooks with minimal scalar/optimizer stand-ins;
they do NOT validate torch numerics, AdamW moments, model loading, or CUDA execution.
The numerical counterparts live in test_pi05_fixed_action_lr.py.

Run without importing pytest's ML-dependent conftest:
    python -m unittest discover -s tests/scripts -p test_pi05_fixed_action_lr_contracts.py -v
"""

import ast
import math
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE = ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"
POLICY_CONFIG_SOURCE = ROOT / "src/lerobot/policies/pi05/configuration_pi05.py"
TRAINER_SOURCE = ROOT / "src/lerobot/scripts/lerobot_train.py"
PIPELINE_CONFIG_SOURCE = ROOT / "src/lerobot/configs/train.py"


@dataclass
class _Control:
    group_scales: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    skip_optimizer_step: bool = False


class _Scalar:
    """Only the scalar API consumed by the trainer's finite-gradient gate."""

    def __init__(self, value):
        self.value = value

    def to(self, **_kwargs):
        return self

    def item(self):
        return self.value


class _Accelerator:
    def __init__(self, processes=1, finite_rank_count=None):
        self.num_processes = processes
        self.finite_rank_count = finite_rank_count
        self.reduce_calls = 0
        self.no_sync_calls = 0

    def unwrap_model(self, policy, **_kwargs):
        return policy

    def reduce(self, value, reduction="mean"):
        if reduction != "sum":
            raise AssertionError("Unexpected reduction")
        self.reduce_calls += 1
        count = value.item() * self.num_processes
        return _Scalar(count if self.finite_rank_count is None else self.finite_rank_count)

    def no_sync(self, _policy):
        self.no_sync_calls += 1
        return nullcontext()


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _class_methods(path, name, methods):
    original = next(node for node in _tree(path).body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    if {node.name for node in selected} != set(methods):
        raise AssertionError(f"Production methods missing from {name}: {methods}")
    return ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])


def _compile(nodes, namespace):
    # Keep original source annotations without importing torch/transformers/accelerate.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<extracted production control contracts>", "exec"), namespace)


def _named_group(optimizer, name):
    return next(group for group in optimizer.param_groups if group["name"] == name)


def _unexpected_candidate(*_args, **_kwargs):
    raise AssertionError("Fixed LR must not estimate the CPC candidate update")


class FixedActionLRContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {
            "math": math,
            "nullcontext": nullcontext,
            "OptimizerStepControl": _Control,
            "PI05TrainingStage": SimpleNamespace(NEXT_ACTION="next_action"),
            "CABO_VLM_PROMPT_GROUP": "vlm_prompt",
            "CABO_ACTION_PROMPT_GROUP": "action_prompt",
            "require_adamw": lambda optimizer: optimizer,
            "get_named_param_group": _named_group,
            "adamw_group_relative_update_moments": _unexpected_candidate,
            "has_method": lambda obj, name: callable(getattr(obj, name, None)),
            "torch": SimpleNamespace(isfinite=lambda value: _Scalar(int(math.isfinite(value))), int64="int64"),
        }
        preparer = next(
            node
            for node in _tree(TRAINER_SOURCE).body
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_policy_optimizer_step_control"
        )
        _compile(
            [
                _class_methods(
                    POLICY_CONFIG_SOURCE, "PI05Config", {"cabo_active", "optimizer_step_control_active"}
                ),
                _class_methods(
                    PIPELINE_CONFIG_SOURCE,
                    "TrainPipelineConfig",
                    {"is_reward_model_training", "trainable_config", "cabo_active", "optimizer_step_control_active"},
                ),
                _class_methods(
                    POLICY_SOURCE,
                    "PI05Policy",
                    {
                        "_is_cabo_active",
                        "_fixed_action_prompt_lr_scale",
                        "compute_optimizer_step_control",
                        "set_training_progress",
                    },
                ),
                preparer,
            ],
            namespace,
        )
        cls.config_type = namespace["PI05Config"]
        cls.policy_type = namespace["PI05Policy"]
        cls.pipeline_type = namespace["TrainPipelineConfig"]
        cls.prepare = staticmethod(namespace["_prepare_policy_optimizer_step_control"])

    def setUp(self):
        self.config = self.config_type()
        self.config.cabo_enabled = False
        self.config.action_prompt_lr_scale = 0.25
        self.config.training_stage = "flow"
        self.config.next_action_bridge_steps = 250
        self.policy = self.policy_type()
        self.policy.config = self.config
        self.policy._stage1_bridge_active = False
        self.optimizer = SimpleNamespace(
            param_groups=[dict(name="vlm_prompt", lr=0.1), dict(name="action_prompt", lr=0.1)]
        )

    def control(self):
        return self.policy.compute_optimizer_step_control({}, self.optimizer, _Accelerator())

    def test_bridge_and_flow_scale_only_action_lr(self):
        for stage, bridge, bypass in (
            ("flow", False, False), ("next_action", True, False), ("next_action", False, True)
        ):
            with self.subTest(stage=stage, bridge=bridge):
                self.config.training_stage = stage
                self.policy._stage1_bridge_active = bridge
                control = self.control()
                self.assertEqual(control.group_scales, {} if bypass else {"action_prompt": 0.25})
                self.assertEqual(control.metrics["fixed_action_lr/effective_action_prompt_lr"], 0.1 if bypass else 0.025)
                self.assertEqual(control.metrics["fixed_action_lr/effective_vlm_prompt_lr"], 0.1)
                self.assertEqual(control.metrics["cabo/active"], 0.0)
                self.assertFalse(control.skip_optimizer_step)
                self.assertEqual([group["lr"] for group in self.optimizer.param_groups], [0.1, 0.1])

    def test_actual_progress_hook_switches_after_750_sap_updates(self):
        self.config.training_stage = "next_action"
        for step, expected_scales in ((0, {}), (749, {}), (750, {"action_prompt": 0.25}), (999, {"action_prompt": 0.25})):
            with self.subTest(step=step):
                self.policy.set_training_progress(step=step, total_steps=1000)
                self.assertEqual(self.control().group_scales, expected_scales)

    def test_legacy_config_without_fixed_lr_field_remains_uncontrolled(self):
        self.policy.config = SimpleNamespace(cabo_enabled=False)
        self.assertEqual(self.control(), _Control())

    def test_invalid_runtime_scales_are_rejected(self):
        for scale in (0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(scale=scale):
                self.config.action_prompt_lr_scale = scale
                with self.assertRaisesRegex(ValueError, "action_prompt_lr_scale"):
                    self.control()

    def test_fixed_lr_and_cpc_are_mutually_exclusive(self):
        self.config.cabo_enabled = True
        with self.assertRaisesRegex(ValueError, "cabo_enabled=False"):
            self.control()

    def test_pipeline_and_policy_properties_include_fixed_control_but_not_cpc(self):
        pipeline = self.pipeline_type()
        pipeline.policy = self.config
        pipeline.reward_model = None
        for enabled, scale, expected in ((False, 0.25, True), (False, 1.0, False), (True, 1.0, True)):
            with self.subTest(enabled=enabled, scale=scale):
                self.config.cabo_enabled = enabled
                self.config.action_prompt_lr_scale = scale
                self.assertIs(self.config.cabo_active, enabled)
                self.assertIs(pipeline.cabo_active, enabled)
                self.assertIs(self.config.optimizer_step_control_active, expected)
                self.assertIs(pipeline.optimizer_step_control_active, expected)

    def test_actual_trainer_gate_dispatches_fixed_control_with_cpc_off(self):
        accelerator = _Accelerator()
        self.policy.compute_optimizer_step_control = Mock(wraps=self.policy.compute_optimizer_step_control)
        control = self.prepare(self.policy, {}, self.optimizer, accelerator, 1.0)
        self.assertEqual(control.group_scales, {"action_prompt": 0.25})
        self.policy.compute_optimizer_step_control.assert_called_once()
        self.assertEqual(accelerator.no_sync_calls, 1)

    def test_actual_trainer_gate_preserves_explicitly_disabled_legacy_control(self):
        self.policy.config = SimpleNamespace(cabo_active=False)
        self.policy.compute_optimizer_step_control = Mock(side_effect=AssertionError("disabled hook called"))
        self.assertEqual(self.prepare(self.policy, {}, self.optimizer, _Accelerator(), 1.0), _Control())
        self.policy.compute_optimizer_step_control.assert_not_called()

    def test_actual_trainer_gate_still_skips_nonfinite_gradients(self):
        for norm in (float("nan"), float("inf")):
            with self.subTest(norm=norm):
                policy = SimpleNamespace(
                    config=self.config,
                    compute_optimizer_step_control=Mock(side_effect=AssertionError("nonfinite hook called")),
                )
                control = self.prepare(policy, {}, self.optimizer, _Accelerator(), norm)
                self.assertTrue(control.skip_optimizer_step)
                self.assertEqual(control.metrics["optimizer_step/nonfinite_gradients"], 1.0)
                policy.compute_optimizer_step_control.assert_not_called()

    def test_actual_trainer_gate_synchronizes_only_finite_flag_for_fixed_control(self):
        accelerator = _Accelerator(processes=2)
        self.assertEqual(
            self.prepare(self.policy, {}, self.optimizer, accelerator, 1.0).group_scales,
            {"action_prompt": 0.25},
        )
        self.assertEqual(accelerator.reduce_calls, 1)
        remote_nonfinite = _Accelerator(processes=2, finite_rank_count=1)
        self.assertTrue(self.prepare(self.policy, {}, self.optimizer, remote_nonfinite, 1.0).skip_optimizer_step)

    def test_actual_trainer_gate_keeps_external_tuple_hook_compatibility(self):
        external_policy = SimpleNamespace(
            compute_optimizer_step_control=lambda **_kwargs: ({"external": 0.5}, {"legacy": 1.0})
        )
        control = self.prepare(external_policy, {}, self.optimizer, _Accelerator(), 1.0)
        self.assertEqual(control.group_scales, {"external": 0.5})
        self.assertEqual(control.metrics, {"legacy": 1.0})


if __name__ == "__main__":
    unittest.main()
