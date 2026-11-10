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

"""Dependency-free contracts executing the actual PI05 LoRA configuration and hooks.

Minimal parameter/PEFT stand-ins exercise control flow, not tensor numerics, LoRA
injection, autograd, checkpoint serialization, or GPU execution. Those real-model
tests live in tests/policies/pi0_pi05/test_pi05_lora_baseline.py.

Run without importing pytest's ML-dependent conftest:
    python -m unittest discover -s tests/scripts -p test_pi05_lora_core_contracts.py -v
"""

import ast
import datetime as dt
import math
import re
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = ROOT / "src/lerobot/policies/pi05/configuration_pi05.py"
MODEL_SOURCE = ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"
PRETRAINED_SOURCE = ROOT / "src/lerobot/policies/pretrained.py"
PIPELINE_SOURCE = ROOT / "src/lerobot/configs/train.py"
TRAINER_SOURCE = ROOT / "src/lerobot/scripts/lerobot_train.py"


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _compile(nodes, namespace):
    namespace.setdefault("__name__", __name__)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<extracted production LoRA contracts>", "exec"), namespace)


def _class_methods(path, class_name, method_names, *, base=None):
    original = next(
        node for node in _tree(path).body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {node.name for node in selected} != set(method_names):
        raise AssertionError(f"Missing production methods in {class_name}: {method_names}")
    return ast.ClassDef(
        name=class_name,
        bases=[] if base is None else [ast.Name(id=base, ctx=ast.Load())],
        keywords=[],
        body=selected,
        decorator_list=[],
    )


@dataclass
class _ConfigBase:
    """Only the unrelated parent/device plumbing needed to construct the real dataclass."""

    device: str = "cpu"
    pretrained_path: str | None = "local-base"
    use_peft: bool = False
    push_to_hub: bool = False

    def __post_init__(self):
        pass

    @classmethod
    def register_subclass(cls, _name):
        return lambda subclass: subclass


def _production_namespace():
    namespace = {
        "math": math,
        "re": re,
        "dataclass": dataclass,
        "field": field,
        "StrEnum": StrEnum,
        "PreTrainedConfig": _ConfigBase,
        "NormalizationMode": SimpleNamespace(IDENTITY="identity", QUANTILES="quantiles"),
        "ACTION": "action",
        "OBS_IMAGES": "observation.images",
        "OBS_STATE": "observation.state",
    }
    config_nodes = [
        node for node in _tree(CONFIG_SOURCE).body if isinstance(node, (ast.Assign, ast.ClassDef))
    ]
    gemma_nodes = [
        node
        for node in _tree(MODEL_SOURCE).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"GemmaConfig", "get_gemma_config"}
    ]
    _compile(
        [
            *config_nodes,
            *gemma_nodes,
            _class_methods(PRETRAINED_SOURCE, "PreTrainedPolicy", {"_validate_peft_config"}),
            _class_methods(MODEL_SOURCE, "PI05Pytorch", {"_freeze_backbones", "_freeze_vlm"}),
            _class_methods(
                MODEL_SOURCE,
                "PI05Policy",
                {"_get_default_peft_targets", "_validate_peft_config", "get_optim_params"},
                base="PreTrainedPolicy",
            ),
        ],
        namespace,
    )
    return namespace


class _Parameter:
    def __init__(self, shape=(2, 2), *, trainable=True):
        self.shape = shape
        self.requires_grad = trainable
        self.grad = object()

    def numel(self):
        return math.prod(self.shape)

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class _ParameterBank:
    def __init__(self, parameter):
        self.parameter = parameter

    def parameters(self):
        return [self.parameter]


class LoraCoreContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _production_namespace()
        cls.config_type = cls.namespace["PI05Config"]
        cls.policy_type = cls.namespace["PI05Policy"]
        cls.core_type = cls.namespace["PI05Pytorch"]
        cls.targets = cls.namespace["PI05_LORA_TARGET_MODULES"]

    def config(self, **changes):
        values = {
            "adaptation_mode": "lora",
            "num_vlm_prompt_tokens": 0,
            "num_prompt_tokens": 0,
            "next_action_pretrain_steps": 0,
            "next_action_bridge_steps": 0,
            "cabo_enabled": False,
        }
        return self.config_type(**(values | changes))

    def policy(self, **config_changes):
        policy = self.policy_type()
        policy.config = self.config(**config_changes)
        return policy

    def adapter(self, policy, **changes):
        return SimpleNamespace(
            **(
                policy._get_default_peft_targets()
                | {"peft_type": "LORA", "init_lora_weights": True}
                | changes
            )
        )

    def validate_adapter(self, policy, adapter):
        peft = ModuleType("peft")
        peft.PeftType = SimpleNamespace(LORA="LORA")
        with patch.dict(sys.modules, {"peft": peft}):
            policy._validate_peft_config(adapter)

    def core(self, *, module_name=None, enabled=True, trainable=True):
        parameters = {
            key: _Parameter(trainable=trainable)
            for key in ("base", "active_a", "active_b", "inactive_a", "inactive_b")
        }
        adapter = SimpleNamespace(
            active_adapters=["default"],
            disable_adapters=not enabled,
            lora_A={"default": _ParameterBank(parameters["active_a"]),
                    "other": _ParameterBank(parameters["inactive_a"])},
            lora_B={"default": _ParameterBank(parameters["active_b"]),
                    "other": _ParameterBank(parameters["inactive_b"])},
        )
        core = self.core_type()
        core.config = self.config()
        name = module_name or "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj"
        core.named_modules = lambda: [(name, adapter)]
        core.parameters = lambda: list(parameters.values())
        core.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(vision_tower=SimpleNamespace(eval=Mock())),
                                     eval=Mock()),
            gemma_expert=SimpleNamespace(eval=Mock()),
        )
        return core, adapter, parameters

    def test_default_prompt_configuration_is_preserved(self):
        config = self.config_type(train_expert_only=False, freeze_vision_encoder=False)
        self.assertEqual(config.adaptation_mode, "prompt")
        self.assertEqual((config.num_vlm_prompt_tokens, config.num_prompt_tokens), (16, 16))
        self.assertTrue(config.cabo_enabled and config.next_action_pretraining_active)
        self.assertTrue(config.train_expert_only and config.freeze_vision_encoder)

    def test_lora_disables_observation_detach_and_prompt_controls(self):
        config = self.config(separate_frozen_observations=True, train_expert_only=True)
        self.assertFalse(config.separate_frozen_observations)
        self.assertFalse(config.train_expert_only)
        self.assertTrue(config.freeze_vision_encoder)
        self.assertFalse(config.next_action_pretraining_active)
        self.assertFalse(config.optimizer_step_control_active)

    def test_lora_rejects_all_eight_incompatible_protocol_settings(self):
        for name, value in (
            ("training_stage", "next_action"),
            ("num_vlm_prompt_tokens", 1),
            ("num_prompt_tokens", 1),
            ("next_action_pretrain_steps", 1),
            ("next_action_bridge_steps", 1),
            ("cabo_enabled", True),
            ("action_prompt_lr_scale", 0.5),
            ("clip_action_head_by_vlm", True),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.config(**{name: value})

    def test_unknown_adaptation_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "adaptation_mode"):
            self.config(adaptation_mode="other")

    def test_exact_targets_include_all_seven_projections_in_both_transformers(self):
        for branch in ("paligemma.model.language_model", "gemma_expert.model"):
            for index in (0, 17):
                for tail in (
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                ):
                    name = f"model.paligemma_with_expert.{branch}.layers.{index}.{tail}"
                    with self.subTest(name=name):
                        self.assertIsNotNone(re.fullmatch(self.targets, name))

    def test_exact_targets_exclude_vision_embeddings_prompts_and_projections(self):
        for name in (
            "model.paligemma_with_expert.paligemma.model.vision_tower."
            "vision_model.encoder.layers.0.self_attn.q_proj",
            "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear",
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens",
            "model.paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense",
            "model.vlm_prompt_tokens", "model.prompt_tokens", "model.action_in_proj", "model.time_mlp_in",
        ):
            with self.subTest(name=name):
                self.assertIsNone(re.fullmatch(self.targets, name))

    def test_actual_default_dimensions_imply_26542080_lora_parameters(self):
        counts = {}
        target_count = 0
        for role, variant in (("vlm", "gemma_2b"), ("action", "gemma_300m")):
            config = self.namespace["get_gemma_config"](variant)
            query_width = config.num_heads * config.head_dim
            kv_width = config.num_kv_heads * config.head_dim
            attention = 2 * (config.width + query_width) + 2 * (config.width + kv_width)
            mlp = 3 * (config.width + config.mlp_dim)
            counts[role] = 16 * config.depth * (attention + mlp)
            target_count += 7 * config.depth
        self.assertEqual(counts, {"vlm": 19_611_648, "action": 6_930_432})
        self.assertEqual(sum(counts.values()), 26_542_080)
        self.assertEqual(target_count, 252)

    def test_default_peft_configuration_is_pure_r16_alpha16_lora(self):
        policy = self.policy()
        adapter = self.adapter(policy)
        self.validate_adapter(policy, adapter)
        self.assertEqual(adapter.target_modules, self.targets)
        self.assertEqual((adapter.r, adapter.lora_alpha), (16, 16))
        self.assertEqual((adapter.lora_dropout, adapter.bias, adapter.modules_to_save), (0.0, "none", []))

    def test_positive_rank_and_alpha_remain_explicitly_configurable(self):
        policy = self.policy()
        self.validate_adapter(policy, self.adapter(policy, r=8, lora_alpha=4))

    def test_bad_peft_configuration_is_rejected_in_thirteen_cases(self):
        policy = self.policy()
        for change in (
            {"peft_type": "IA3"}, {"target_modules": ["q_proj"]},
            {"modules_to_save": ["prompt_tokens"]}, {"r": 0}, {"r": True},
            {"lora_alpha": float("nan")}, {"lora_alpha": 0}, {"lora_dropout": 0.1},
            {"bias": "all"}, {"use_dora": True}, {"use_rslora": True},
            {"init_lora_weights": False}, {"rank_pattern": {"q_proj": 8}},
        ):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "PI05 LoRA"):
                self.validate_adapter(policy, self.adapter(policy, **change))

    def test_peft_requires_original_pretrained_path(self):
        policy = self.policy(pretrained_path=None)
        with self.assertRaisesRegex(ValueError, "pretrained_path"):
            self.validate_adapter(policy, self.adapter(policy))

    def test_prompt_peft_saving_behavior_remains_unchanged(self):
        policy = self.policy(adaptation_mode="prompt")
        adapter = SimpleNamespace(modules_to_save=["action_out_proj"])
        policy._validate_peft_config(adapter)
        self.assertEqual(
            adapter.modules_to_save,
            ["action_out_proj", "model.vlm_prompt_tokens", "model.prompt_tokens"],
        )
        self.assertEqual(policy._get_default_peft_targets()["target_modules"], "dummy-target-modules")

    def test_freeze_preserves_active_adapters_and_clears_frozen_gradients(self):
        core, _, parameters = self.core()
        core._freeze_vlm()
        self.assertEqual(
            {name for name, parameter in parameters.items() if parameter.requires_grad},
            {"active_a", "active_b"},
        )
        for name in ("base", "inactive_a", "inactive_b"):
            self.assertIsNone(parameters[name].grad)
        core.paligemma_with_expert.paligemma.model.vision_tower.eval.assert_called_once_with()
        core.paligemma_with_expert.paligemma.eval.assert_not_called()
        core.paligemma_with_expert.gemma_expert.eval.assert_not_called()

    def test_freeze_follows_changed_active_adapter_selection(self):
        core, adapter, parameters = self.core()
        adapter.active_adapters = ["other"]
        core._freeze_backbones()
        self.assertEqual(
            {name for name, parameter in parameters.items() if parameter.requires_grad},
            {"inactive_a", "inactive_b"},
        )

    def test_disabled_adapters_are_not_reenabled(self):
        core, _, parameters = self.core(enabled=False)
        core._freeze_backbones()
        self.assertFalse(any(parameter.requires_grad for parameter in parameters.values()))

    def test_inference_adapters_are_not_reenabled(self):
        core, _, parameters = self.core(trainable=False)
        core._freeze_backbones()
        self.assertFalse(any(parameter.requires_grad for parameter in parameters.values()))

    def test_wrong_target_adapters_remain_frozen(self):
        core, _, parameters = self.core(
            module_name="paligemma_with_expert.paligemma.model.vision_tower.layers.0.self_attn.q_proj"
        )
        core._freeze_backbones()
        self.assertFalse(any(parameter.requires_grad for parameter in parameters.values()))

    def test_prompt_freeze_contract_remains_unchanged(self):
        core, _, parameters = self.core()
        core.config.adaptation_mode = "prompt"
        core.vlm_prompt_tokens = _ParameterBank(parameters["active_a"])
        core.prompt_tokens = _ParameterBank(parameters["active_b"])
        core._freeze_backbones()
        self.assertEqual(
            {name for name, parameter in parameters.items() if parameter.requires_grad},
            {"active_a", "active_b"},
        )
        core.paligemma_with_expert.paligemma.eval.assert_called_once_with()
        core.paligemma_with_expert.gemma_expert.eval.assert_called_once_with()

    def test_optimizer_collects_exact_active_adapter_parameters(self):
        policy = self.policy()
        core, _, parameters = self.core()
        policy.model = core
        policy.parameters = core.parameters
        self.assertEqual(
            {id(parameter) for parameter in policy.get_optim_params()},
            {id(parameters["active_a"]), id(parameters["active_b"])},
        )

    def test_optimizer_rejects_an_empty_or_frozen_adapter_set(self):
        policy = self.policy()
        core, _, _ = self.core(trainable=False)
        policy.model = core
        policy.parameters = core.parameters
        with self.assertRaisesRegex(ValueError, "non-empty active LoRA"):
            policy.get_optim_params()


class LoraTrainingPipelineContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {"Path": Path, "dt": dt}
        _compile(
            [_class_methods(
                PIPELINE_SOURCE,
                "TrainPipelineConfig",
                {"validate", "is_reward_model_training", "trainable_config",
                 "cabo_active", "optimizer_step_control_active"},
            )],
            namespace,
        )
        cls.pipeline_type = namespace["TrainPipelineConfig"]

    def pipeline(self, *, mode="lora", peft=None, use_peft=False):
        pipeline = self.pipeline_type()
        pipeline.policy = SimpleNamespace(
            adaptation_mode=mode, use_peft=use_peft, pretrained_path="base", push_to_hub=False,
            get_optimizer_preset=Mock(return_value="optimizer"),
            get_scheduler_preset=Mock(return_value="scheduler"),
        )
        pipeline._resolve_pretrained_from_cli = Mock()
        pipeline.reward_model = None
        pipeline.peft = peft
        pipeline.rename_map = {}
        pipeline.job_name = "contract"
        pipeline.env = None
        pipeline.resume = False
        pipeline.output_dir = "uncreated-contract-output"
        pipeline.dataset = SimpleNamespace(repo_id="libero", eval_split=0.0)
        pipeline.use_policy_training_preset = True
        pipeline.eval_steps = 0
        pipeline.job = SimpleNamespace(is_remote=False)
        pipeline.save_checkpoint_to_hub = False
        return pipeline

    def test_missing_lora_adapter_fails_before_optimizer_setup(self):
        pipeline = self.pipeline()
        with self.assertRaisesRegex(ValueError, "requires a native PEFT adapter"):
            pipeline.validate()
        pipeline._resolve_pretrained_from_cli.assert_called_once_with()
        pipeline.policy.get_optimizer_preset.assert_not_called()

    def test_fresh_native_peft_training_passes_with_use_peft_false(self):
        pipeline = self.pipeline(peft=SimpleNamespace(method_type="LORA"))
        pipeline.validate()
        self.assertFalse(pipeline.policy.use_peft)
        self.assertEqual((pipeline.optimizer, pipeline.scheduler), ("optimizer", "scheduler"))

    def test_saved_adapter_loading_passes_without_new_peft_configuration(self):
        pipeline = self.pipeline(use_peft=True)
        pipeline.validate()
        self.assertIsNone(pipeline.peft)
        self.assertEqual(pipeline.optimizer, "optimizer")

    def test_prompt_training_does_not_require_a_peft_adapter(self):
        pipeline = self.pipeline(mode="prompt")
        pipeline.validate()
        self.assertEqual(pipeline.optimizer, "optimizer")


class LoraParameterReportContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        function = next(
            node for node in _tree(TRAINER_SOURCE).body
            if isinstance(node, ast.FunctionDef) and node.name == "_train_single_stage"
        )
        count_assignments = [
            node for node in function.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {"num_learnable_params", "num_total_params"}
                    for target in node.targets)
        ]
        reports = [
            node for node in function.body
            if isinstance(node, ast.If)
            and any(isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "_parameter_counts"
                            for target in statement.targets)
                    for statement in node.body)
        ]
        if len(count_assignments) != 2 or len(reports) != 1:
            raise AssertionError("Could not uniquely identify the production trainable parameter report")
        cls.nodes = [*count_assignments, reports[0]]

    def report(self, *, mode="lora", is_main=True, extra_parameters=()):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-lora-report-contract-")
        self.addCleanup(temporary.cleanup)
        prefix = "base_model.model.model.paligemma_with_expert."
        parameters = [
            (prefix + "paligemma.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
             _Parameter((16, 32))),
            (prefix + "gemma_expert.model.layers.0.self_attn.q_proj.lora_B.default.weight",
             _Parameter((16, 16))),
            (prefix + "paligemma.model.language_model.layers.0.self_attn.q_proj.base_layer.weight",
             _Parameter((32, 32), trainable=False)),
            (prefix + "paligemma.model.language_model.layers.0.self_attn.q_proj.lora_A.other.weight",
             _Parameter((16, 32), trainable=False)),
            *extra_parameters,
        ]
        policy = SimpleNamespace(
            parameters=lambda: [parameter for _, parameter in parameters],
            named_parameters=lambda: parameters,
            peft_config={"default": SimpleNamespace(base_model_name_or_path="original-local-base")},
        )
        cfg = SimpleNamespace(
            output_dir=Path(temporary.name), seed=7, steps=4000, batch_size=32,
            dataset=SimpleNamespace(root=Path("same-dataset")),
            policy=SimpleNamespace(adaptation_mode=mode, training_stage="flow", dtype="float32",
                                   tokenizer_name="same-tokenizer"),
        )
        namespace = {
            "Path": Path, "policy": policy, "cfg": cfg, "is_main_process": is_main,
            "accelerator": SimpleNamespace(num_processes=3, mixed_precision="no"),
            "logging": SimpleNamespace(info=Mock()), "write_json": Mock(),
        }
        _compile(self.nodes, namespace)
        return namespace

    def test_report_uses_actual_counts_names_shapes_and_base_initialization(self):
        namespace = self.report()
        writer = namespace["write_json"]
        writer.assert_called_once()
        payload, path = writer.call_args.args
        self.assertEqual(path.name, "trainable_parameters.json")
        self.assertEqual(payload["trainable_total"], 768)
        self.assertEqual(payload["all_parameters"], 2304)
        self.assertEqual(payload["counts"],
                         {"vlm_prompt": 0, "action_prompt": 0, "lora_vlm": 512,
                          "lora_action": 256, "other": 0})
        self.assertEqual(len(payload["parameters"]), 2)
        self.assertEqual([row["shape"] for row in payload["parameters"]], [[16, 32], [16, 16]])
        self.assertTrue(all(".default." in row["name"] for row in payload["parameters"]))
        self.assertEqual(payload["base_model_paths"], {"default": "original-local-base"})
        self.assertEqual((payload["training_seed"], payload["total_updates"]), (7, 4000))
        self.assertEqual((payload["batch_size_per_process"], payload["num_processes"],
                          payload["effective_batch_size"]), (32, 3, 96))
        self.assertEqual((payload["dtype"], payload["mixed_precision"]), ("float32", "no"))
        self.assertEqual((payload["dataset_root"], payload["tokenizer_name"]),
                         ("same-dataset", "same-tokenizer"))

    def test_report_exposes_unexpected_trainable_prompts_or_other_weights(self):
        namespace = self.report(extra_parameters=(
            ("model.vlm_prompt_tokens.weight", _Parameter((2, 32))),
            ("model.prompt_tokens.weight", _Parameter((2, 16))),
            ("model.unexpected.weight", _Parameter((3,))),
        ))
        payload = namespace["write_json"].call_args.args[0]
        self.assertEqual(payload["trainable_total"], 867)
        self.assertEqual(payload["counts"]["vlm_prompt"], 64)
        self.assertEqual(payload["counts"]["action_prompt"], 32)
        self.assertEqual(payload["counts"]["other"], 3)
        self.assertEqual(sum(payload["counts"].values()), payload["trainable_total"])

    def test_non_main_process_does_not_log_or_write_a_report(self):
        namespace = self.report(is_main=False)
        namespace["write_json"].assert_not_called()
        namespace["logging"].info.assert_not_called()

    def test_prompt_mode_does_not_write_lora_protocol_report(self):
        namespace = self.report(mode="prompt")
        namespace["write_json"].assert_not_called()
        namespace["logging"].info.assert_called()


if __name__ == "__main__":
    unittest.main()
