"""Check the long-run LR recipe against production functions without GPU dependencies.

AST loading retains the real scheduler, PI05 presets and preparation-config builder.
Only dependency registration, the optimizer config and PyTorch's LambdaLR wrapper
are replaced; the LR function and stage configuration are not reimplemented here.
Run with ``uv run --no-project --offline python tests/scripts/test_piper_long_scheduler.py``.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import logging
import math
import sys
import unittest
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCHEDULERS = ROOT / "src/lerobot/optim/schedulers.py"
POLICY_CONFIG = ROOT / "src/lerobot/policies/pi05/configuration_pi05.py"
TRAINER = ROOT / "src/lerobot/scripts/lerobot_train.py"


@dataclasses.dataclass
class _BasePolicyConfig:
    use_peft: bool = False
    push_to_hub: bool = False
    type = "pi05"

    def __post_init__(self):
        pass


@dataclasses.dataclass
class _WandbConfig:
    run_id: str | None = "formal-flow-run"


@dataclasses.dataclass
class _JobConfig:
    target: str = "local"


def _compile(nodes, source, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)


@lru_cache(maxsize=1)
def _production():
    namespace = {
        "__name__": __name__,
        "copy": copy,
        "dataclasses": dataclasses,
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "logging": logging,
        "math": math,
        "StrEnum": StrEnum,
        "PreTrainedConfig": _BasePolicyConfig,
        "DEFAULT_IMAGE_SIZE": 224,
        "NormalizationMode": SimpleNamespace(IDENTITY="identity", QUANTILES="quantiles"),
        "AdamWConfig": SimpleNamespace,
        # Capture the actual production callback instead of duplicating its LR formula.
        "LambdaLR": lambda optimizer, lr_lambda, last_epoch: lr_lambda,
    }
    scheduler = next(
        node
        for node in ast.parse(SCHEDULERS.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "CosineDecayWithWarmupSchedulerConfig"
    )
    scheduler.bases, scheduler.keywords = [], []
    scheduler.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
    _compile([scheduler], SCHEDULERS, namespace)

    policy_nodes = []
    for node in ast.parse(POLICY_CONFIG.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("PI05"):
            if node.name == "PI05Config":
                node.decorator_list = [ast.Name(id="dataclass", ctx=ast.Load())]
            policy_nodes.append(node)
    _compile(policy_nodes, POLICY_CONFIG, namespace)

    trainer_nodes = []
    for node in ast.parse(TRAINER.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_pi05_next_action_pretraining_active",
            "_make_pi05_next_action_pretraining_config",
        }:
            trainer_nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_PI05_NEXT_ACTION_PRETRAIN_DIR"
            for target in node.targets
        ):
            trainer_nodes.append(node)
    _compile(trainer_nodes, TRAINER, namespace)
    return namespace


def _long_flow_config():
    policy = _production()["PI05Config"](
        optimizer_lr=2.5e-5,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=49000,
        scheduler_decay_lr=1e-6,
        next_action_pretrain_steps=1000,
        next_action_bridge_steps=250,
    )
    return SimpleNamespace(
        policy=policy,
        steps=49000,
        resume=False,
        is_reward_model_training=False,
        peft=None,
        output_dir=Path("outputs/piper-long-scheduler-test"),
        job_name="piper-long-scheduler-test",
        save_steps=[2000, 11000, 24000, 49000],
        wandb=_WandbConfig(),
        job=_JobConfig(),
    )


def _preparation_config(formal_config):
    production = _production()
    name = "lerobot.policies.pi05.configuration_pi05"
    module = ModuleType(name)
    module.PI05Config = production["PI05Config"]
    module.PI05TrainingStage = production["PI05TrainingStage"]
    # The production builder's local import must use the isolated config type.
    # Restore sys.modules immediately so other tests retain their real imports.
    with patch.dict(sys.modules, {name: module}):
        return production["_make_pi05_next_action_pretraining_config"](formal_config)


def _learning_rate_at(scheduler_config, training_steps):
    callback = scheduler_config.build(optimizer=None, num_training_steps=training_steps)
    return lambda step: scheduler_config.peak_lr * callback(step)


class PiperLongSchedulerTest(unittest.TestCase):
    def test_long_flow_decays_until_49000_instead_of_plateauing_at_30000(self):
        formal = _long_flow_config()
        lr = _learning_rate_at(formal.policy.get_scheduler_preset(), formal.steps)
        self.assertGreater(lr(0), 0)
        self.assertLess(lr(0), 1e-6)
        # The existing warmup-to-cosine boundary is near, but not exactly at, peak LR.
        self.assertGreater(lr(1000), 0.99 * 2.5e-5)
        self.assertLessEqual(lr(1000), 2.5e-5)
        decay = [lr(step) for step in (1000, 10000, 30000, 48000, 49000)]
        self.assertTrue(all(left > right for left, right in zip(decay, decay[1:])))
        self.assertGreater(lr(30000), 1e-6)
        self.assertAlmostEqual(lr(49000), 1e-6, delta=1e-14)
        self.assertAlmostEqual(lr(50000), 1e-6, delta=1e-14)

    def test_preparation_builder_keeps_fixed_recipe_and_does_not_mutate_formal_flow(self):
        formal = _long_flow_config()
        original = copy.deepcopy(vars(formal))
        preparation = _preparation_config(formal)
        self.assertEqual(vars(formal), original)
        self.assertIsNot(preparation.policy, formal.policy)
        self.assertEqual(preparation.steps, 1000)
        self.assertEqual(preparation.policy.next_action_bridge_steps, 250)
        self.assertEqual(preparation.policy.training_stage, "next_action")
        self.assertFalse(preparation.policy.next_action_pretraining_active)
        self.assertEqual(preparation.scheduler.num_warmup_steps, 1000)
        self.assertEqual(preparation.scheduler.num_decay_steps, 30000)
        self.assertEqual(preparation.scheduler.peak_lr, 2.5e-5)
        self.assertEqual(preparation.scheduler.decay_lr, 1e-5)

    def test_1000_step_preparation_preserves_existing_short_run_scaling(self):
        preparation = _preparation_config(_long_flow_config())
        lr = _learning_rate_at(preparation.scheduler, preparation.steps)
        # The fixed 30k preset compresses warmup to 33 and decay to the 1000-step
        # preparation budget. A leaked 49k horizon or 1e-6 floor breaks these checks.
        self.assertLess(lr(0), lr(1))
        self.assertLess(lr(1), lr(32))
        self.assertGreater(lr(32), 0.95 * 2.5e-5)
        self.assertGreater(lr(33), 0.99 * 2.5e-5)
        self.assertLess(lr(33), 2.5e-5)
        self.assertLess(lr(34), lr(33))
        self.assertGreater(lr(750), 1e-5)
        self.assertLess(lr(750), lr(34))
        self.assertAlmostEqual(lr(1000), 1e-5, delta=1e-14)
        self.assertAlmostEqual(lr(1001), 1e-5, delta=1e-14)


if __name__ == "__main__":
    unittest.main()
