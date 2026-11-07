"""Exercise the production LR hooks with small CPU tensors, without VLM dependencies.

AST loading avoids importing optional torchvision/transformers/datasets packages;
the config validation, policy hooks, trainer dispatch and AdamW LR context are the
production implementations. These tests do not run the robot or a model forward.
Run directly with ``python tests/scripts/test_pi05_fixed_action_prompt_lr.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/lerobot/policies/pi05/configuration_pi05.py"
MODEL = ROOT / "src/lerobot/policies/pi05/modeling_pi05.py"
TRAINER = ROOT / "src/lerobot/scripts/lerobot_train.py"


@dataclass
class _BaseConfig:
    use_peft: bool = False

    def __post_init__(self):
        pass


def _compile(nodes, path, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@lru_cache(maxsize=1)
def _config_type():
    namespace = {
        "__name__": __name__,
        "PreTrainedConfig": _BaseConfig,
        "dataclass": dataclass,
        "field": field,
        "math": math,
        "StrEnum": StrEnum,
        "DEFAULT_IMAGE_SIZE": 224,
        "NormalizationMode": SimpleNamespace(IDENTITY="identity", QUANTILES="quantiles"),
    }
    nodes = []
    for node in ast.parse(CONFIG.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("PI05"):
            if node.name == "PI05Config":
                node.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
            nodes.append(node)
    _compile(nodes, CONFIG, namespace)
    return namespace["PI05Config"]


@lru_cache(maxsize=1)
def _production():
    name = "_pi05_fixed_lr_cabo_for_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/lerobot/optim/cabo.py")
    cabo = importlib.util.module_from_spec(spec)
    sys.modules[name] = cabo
    spec.loader.exec_module(cabo)
    namespace = dict(vars(cabo))
    namespace.update(
        torch=torch,
        nn=torch.nn,
        math=math,
        PI05TrainingStage=SimpleNamespace(NEXT_ACTION="next_action"),
        has_method=lambda obj, name: callable(getattr(obj, name, None)),
        nullcontext=nullcontext,
    )
    names = {
        "_is_cabo_active",
        "_cabo_parameter_groups",
        "get_optim_params",
        "validate_optimizer_step_control",
        "compute_optimizer_step_control",
    }
    selected = []
    for node in ast.parse(MODEL.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_unique_trainable_parameters":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "PI05Policy":
            node.bases, node.keywords, node.decorator_list = [], [], []
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
            selected.append(node)
    _compile(selected, MODEL, namespace)
    dispatch = [
        node
        for node in ast.parse(TRAINER.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_policy_optimizer_step_control"
    ]
    _compile(dispatch, TRAINER, namespace)
    return namespace, cabo


class _Accelerator:
    num_processes = 1

    def unwrap_model(self, policy, **_kwargs):
        return policy

    def no_sync(self, _policy):
        return nullcontext()


def _policy(scale=0.1, stage="flow", bridge=False):
    production, _ = _production()

    class TinyPolicy(torch.nn.Module, production["PI05Policy"]):
        def __init__(self):
            super().__init__()
            self.config = _config_type()(
                action_prompt_lr_scale=scale,
                cabo_enabled=False,
                training_stage=stage,
            )
            self._stage1_bridge_active = bridge
            self.model = torch.nn.Module()
            self.model.vlm_prompt_tokens = torch.nn.Embedding(2, 3, dtype=torch.float64)
            self.model.prompt_tokens = torch.nn.Embedding(2, 3, dtype=torch.float64)
            self.model.backbone = torch.nn.Linear(3, 3, dtype=torch.float64).requires_grad_(False)
            # The partition is already frozen here; the separate prompt-partition regression
            # exercises the actual backbone freezing methods.
            self.model._freeze_vlm = lambda: None

    return TinyPolicy()


class FixedActionPromptConfigTest(unittest.TestCase):
    def test_default_preserves_cpc_and_capacity_configurations(self):
        cfg = _config_type()()
        self.assertEqual(cfg.action_prompt_lr_scale, 1.0)
        self.assertTrue(cfg.cabo_active)
        cfg = _config_type()(cabo_enabled=False, train_action_head=True, train_action_projections=True)
        self.assertEqual(cfg.action_prompt_lr_scale, 1.0)

    def test_rejects_invalid_scales_and_mixed_parameter_budgets(self):
        for scale in (0, -1, 1.1, float("inf"), float("nan"), True):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "action_prompt_lr_scale"):
                _config_type()(cabo_enabled=False, action_prompt_lr_scale=scale)
        for extra in (
            {"cabo_enabled": True},
            {"num_vlm_prompt_tokens": 0},
            {"num_prompt_tokens": 0},
            {"use_peft": True},
            {"train_action_projections": True},
            {"train_action_projections": True, "train_action_expert_last_n_layers": 1},
            {"train_action_projections": True, "train_action_head": True},
        ):
            kwargs = {"cabo_enabled": False, "action_prompt_lr_scale": 0.1, **extra}
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "action_prompt_lr_scale"):
                _config_type()(**kwargs)


@unittest.skipIf(torch is None, "requires PyTorch CPU; no GPU or Transformers needed")
class FixedActionPromptStepTest(unittest.TestCase):
    def test_joint_steps_match_independent_adamw_and_preserve_scheduler(self):
        _, cabo = _production()
        for stage, bridge in (("flow", False), ("next_action", True)):
            for scale in (0.5, 0.1, 0.03, 0.01):
                for decay in (0.0, 0.01):
                    with self.subTest(stage=stage, scale=scale, decay=decay):
                        policy = _policy(scale, stage, bridge)
                        optimizer = torch.optim.AdamW(
                            policy.get_optim_params(), lr=0.1, weight_decay=decay, foreach=False
                        )
                        policy.validate_optimizer_step_control(optimizer)
                        original = [policy.model.vlm_prompt_tokens.weight, policy.model.prompt_tokens.weight]
                        reference = [torch.nn.Parameter(p.detach().clone()) for p in original]
                        ref_optimizer = torch.optim.AdamW(
                            [
                                {"params": [reference[0]], "lr": 0.1},
                                {"params": [reference[1]], "lr": 0.1 * scale},
                            ],
                            weight_decay=decay,
                            foreach=False,
                        )
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.7**step)
                        ref_scheduler = torch.optim.lr_scheduler.LambdaLR(
                            ref_optimizer, lambda step: 0.7**step
                        )
                        for step in range(3):
                            for p, ref in zip(original, reference, strict=True):
                                p.grad = torch.full_like(p, 0.4 - step)
                                ref.grad = p.grad.clone()
                            lrs = [group["lr"] for group in optimizer.param_groups]
                            control = policy.compute_optimizer_step_control({}, optimizer, _Accelerator())
                            self.assertEqual(control.group_scales, {"action_prompt": scale})
                            with cabo.temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
                                optimizer.step()
                            ref_optimizer.step()
                            self.assertEqual([group["lr"] for group in optimizer.param_groups], lrs)
                            for p, ref in zip(original, reference, strict=True):
                                # Independent scheduler/group scaling can change floating-point
                                # multiplication order; compare at double-precision tolerance.
                                torch.testing.assert_close(p, ref, rtol=1e-12, atol=1e-14)
                            scheduler.step()
                            ref_scheduler.step()

    def test_stage_one_keeps_action_lr_and_bridge_reuses_moments(self):
        _, cabo = _production()
        policy = _policy(stage="next_action")
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.1, weight_decay=0.01, foreach=False)
        vlm = policy.model.vlm_prompt_tokens.weight
        action = policy.model.prompt_tokens.weight
        initial_vlm = vlm.detach().clone()
        action.grad = torch.ones_like(action)
        control = policy.compute_optimizer_step_control({}, optimizer, _Accelerator())
        self.assertEqual(control.group_scales, {})
        self.assertEqual(control.metrics["fixed_lr/action_only_bypass"], 1.0)
        with cabo.temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            optimizer.step()
        torch.testing.assert_close(vlm, initial_vlm, rtol=0, atol=0)
        self.assertNotIn(vlm, optimizer.state)
        self.assertEqual(optimizer.state[action]["step"].item(), 1)
        moment_before = optimizer.state[action]["exp_avg"].clone()
        policy._stage1_bridge_active = True
        vlm.grad = torch.ones_like(vlm)
        control = policy.compute_optimizer_step_control({}, optimizer, _Accelerator())
        torch.testing.assert_close(optimizer.state[action]["exp_avg"], moment_before, rtol=0, atol=0)
        self.assertEqual(control.group_scales, {"action_prompt": 0.1})
        with cabo.temporary_optimizer_group_lr_scales(optimizer, control.group_scales):
            optimizer.step()
        self.assertEqual(optimizer.state[action]["step"].item(), 2)
        self.assertEqual(optimizer.state[vlm]["step"].item(), 1)

    def test_trainer_calls_fixed_control_when_cpc_is_disabled(self):
        production, _ = _production()
        policy = _policy()
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.1)
        result = production["_prepare_policy_optimizer_step_control"](
            policy, {}, optimizer, _Accelerator(), torch.tensor(1.0)
        )
        self.assertEqual(result.group_scales, {"action_prompt": 0.1})
        self.assertEqual(result.metrics["cabo/active"], 0.0)
        result = production["_prepare_policy_optimizer_step_control"](
            policy, {}, optimizer, _Accelerator(), torch.tensor(float("nan"))
        )
        self.assertTrue(result.skip_optimizer_step)

    def test_default_disabled_control_preserves_unnamed_groups_and_dispatch(self):
        production, _ = _production()
        policy = _policy(scale=1.0)
        parameters = policy.get_optim_params()
        self.assertEqual(len(parameters), 2)
        self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in parameters))
        policy.compute_optimizer_step_control = lambda **_kwargs: self.fail("inactive hook was invoked")
        optimizer = torch.optim.AdamW(parameters, lr=0.1)
        result = production["_prepare_policy_optimizer_step_control"](
            policy, {}, optimizer, _Accelerator(), torch.tensor(1.0)
        )
        self.assertEqual(result.group_scales, {})

    def test_fixed_control_validates_parameter_identity(self):
        policy = _policy()
        groups = policy.get_optim_params()
        groups[1]["params"] = [torch.nn.Parameter(torch.zeros(2, 3))]
        optimizer = torch.optim.AdamW(groups, lr=0.1)
        with self.assertRaisesRegex(ValueError, "parameter partition"):
            policy.validate_optimizer_step_control(optimizer)


if __name__ == "__main__":
    unittest.main()
