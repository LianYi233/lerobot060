"""CPU-only snapshot and launch-protocol checks; no model weights or GPU dependencies.

Load the small production functions from their AST so these checks can also run
with stdlib unittest in an environment without the optional training packages.
"""

import ast
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def _load_functions(relative_path, names, namespace):
    source = ast.parse((ROOT / relative_path).read_text())
    selected = [
        node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace


class ExactCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.gathers = [], []
        self.functions = _load_functions(
            "src/lerobot/scripts/lerobot_train.py",
            {"_is_scheduled_checkpoint_step", "_save_pi05_ntk_initial_snapshot"},
            {
                "TrainPipelineConfig": SimpleNamespace,
                "logging": logging,
                "get_step_checkpoint_dir": lambda root, total, step: root / f"checkpoints/{step:06d}",
                "save_checkpoint": lambda **kwargs: self.calls.append(kwargs),
                "gather_fsdp_state_dicts": self.gather,
            },
        )
        self.cfg = SimpleNamespace(
            steps=4000,
            save_steps=[0, 750, 1000, 4000],
            save_freq=4000,
            save_checkpoint=True,
            resume=False,
            policy=SimpleNamespace(ntk_save_stage_snapshots=False),
            output_dir=Path("run"),
            batch_size=32,
        )

    def gather(self, policy, optimizer):
        self.gathers.append((policy, optimizer))
        return {"prompts": "initialized"}, {"state": "empty"}

    def test_exact_completed_update_schedule_and_legacy_defaults(self):
        scheduled = self.functions["_is_scheduled_checkpoint_step"]
        self.assertEqual([s for s in range(4001) if scheduled(self.cfg, s)], [0, 750, 1000, 4000])
        # Resuming after 750 does not trigger a second initialization snapshot or repeat 750.
        self.assertEqual([s for s in range(751, 4001) if scheduled(self.cfg, s)], [1000, 4000])
        self.cfg.save_steps = []
        self.cfg.save_freq = 1000
        self.assertEqual([s for s in range(4001) if scheduled(self.cfg, s)], [1000, 2000, 3000, 4000])

    def test_actual_initialized_model_and_processors_saved_on_main_rank_only(self):
        fake_accelerate = SimpleNamespace(DistributedType=SimpleNamespace(FSDP="fsdp"))
        save_initial = self.functions["_save_pi05_ntk_initial_snapshot"]
        for fsdp in (False, True):
            for main_rank in (False, True):
                with self.subTest(fsdp=fsdp, main_rank=main_rank):
                    self.calls.clear()
                    self.gathers.clear()
                    barriers = []
                    accelerator = SimpleNamespace(
                        distributed_type="fsdp" if fsdp else "ddp",
                        is_main_process=main_rank,
                        num_processes=2,
                        unwrap_model=lambda model: model,
                        wait_for_everyone=lambda: barriers.append(True),
                    )
                    model, optimizer, scheduler, pre, post = [object() for _ in range(5)]
                    with patch.dict(sys.modules, {"accelerate": fake_accelerate}):
                        save_initial(self.cfg, model, optimizer, scheduler, pre, post, accelerator, 0)
                    self.assertEqual(len(self.gathers), int(fsdp))
                    self.assertEqual(len(self.calls), int(main_rank))
                    self.assertEqual(len(barriers), 1)
                    if main_rank:
                        saved = self.calls[0]
                        self.assertIs(saved["policy"], model)
                        self.assertIs(saved["optimizer"], optimizer)
                        self.assertIs(saved["scheduler"], scheduler)
                        self.assertIs(saved["preprocessor"], pre)
                        self.assertIs(saved["postprocessor"], post)
                        self.assertEqual(saved["checkpoint_dir"], Path("run/checkpoints/000000"))
                        self.assertEqual(saved["step"], 0)

    def test_initial_snapshot_skipped_on_resume_or_after_any_update(self):
        fake_accelerate = SimpleNamespace(DistributedType=SimpleNamespace(FSDP="fsdp"))
        save_initial = self.functions["_save_pi05_ntk_initial_snapshot"]
        # No accelerator attributes are accessed on a skipped snapshot.
        with patch.dict(sys.modules, {"accelerate": fake_accelerate}):
            self.cfg.resume = True
            save_initial(self.cfg, None, None, None, None, None, None, 0)
            self.cfg.resume = False
            save_initial(self.cfg, None, None, None, None, None, None, 750)
            self.cfg.save_checkpoint = False
            save_initial(self.cfg, None, None, None, None, None, None, 0)
            self.cfg.save_checkpoint = True
            self.cfg.save_steps = []
            save_initial(self.cfg, None, None, None, None, None, None, 0)
        self.assertEqual(self.calls, [])

    def test_configuration_rejects_invalid_steps_before_loading_a_model(self):
        validate = _load_functions("src/lerobot/configs/train.py", {"validate"}, {})["validate"]
        for values in ([-1], [4001], [1.5], [True], ["750"]):
            with self.subTest(values=values):
                self.cfg.save_steps = values
                with self.assertRaisesRegex(ValueError, "save_steps"):
                    validate(self.cfg)
        self.cfg.save_steps = [0, 750]
        self.cfg.save_checkpoint = False
        with self.assertRaisesRegex(ValueError, "save_checkpoint"):
            validate(self.cfg)


class ControlLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.env = dict(os.environ)
        for key in ("DATASET_ROOT", "PRETRAINED_PATH", "TOKENIZER_PATH"):
            path = root / key
            path.mkdir()
            self.env[key] = str(path)
        self.env.update(
            DRY_RUN="true",
            OUTPUT_ROOT=str(root / "out"),
            LOG_ROOT=str(root / "logs"),
            FLOW_STEPS="3000",
            SAVE_STEPS="[]",
            SAVE_FREQ="3000",
            NTK_SAVE_STAGE_SNAPSHOTS="true",
        )
        self.env.pop("RUN_NAME", None)

    def launch(self, *args):
        return subprocess.run(
            ["bash", str(ROOT / "examples/training/train_pi05_prompt_ntk_control.sh"), *args],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_plain_control_has_matched_budget_snapshots_and_isolated_output(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout.splitlines()[-1])
        for argument in (
            "--steps=4000",
            "--save_steps=[0,750,1000,4000]",
            "--save_freq=4000",
            "--policy.training_stage=flow",
            "--policy.next_action_pretrain_steps=0",
            "--policy.next_action_bridge_steps=0",
            "--policy.cabo_enabled=false",
            "--policy.num_vlm_prompt_tokens=16",
            "--policy.num_prompt_tokens=16",
            "--policy.ntk_save_stage_snapshots=false",
            "--save_checkpoint=true",
            "--job_name=pi05-dual_prompt_only-ntk-control-seed0",
        ):
            self.assertIn(argument, args)

    def test_cpc_enabled_control_is_optional_and_nonprotocol_args_pass_through(self):
        result = self.launch("direct_dual", "7", "--num_workers=2")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout.splitlines()[-1])
        self.assertIn("--policy.cabo_enabled=true", args)
        self.assertIn("--seed=7", args)
        self.assertIn("--num_workers=2", args)
        self.assertIn("--job_name=pi05-direct_dual-ntk-control-seed7", args)

    def test_protocol_overrides_are_rejected_in_both_cli_forms(self):
        for args in (
            ["--policy.next_action_pretrain_steps=1000"],
            ["--policy.next_action_pretrain_steps", "1000"],
            ["--policy.training_stage=next_action"],
            ["--steps=3000"],
            ["--policy.cabo_enabled=true"],
            ["--save_steps=[]"],
            ["--resume=true"],
            ["--config_path=old.json"],
            ["--policy.path=old_checkpoint"],
            ["--ste=3000"],
            ["--policy.cabo_enable=true"],
            ["--res=true"],
            ["--config_pat=old.json"],
        ):
            with self.subTest(args=args):
                result = self.launch("dual_prompt_only", "0", *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("fixed by the NTK control protocol", result.stderr)


if __name__ == "__main__":
    unittest.main()
