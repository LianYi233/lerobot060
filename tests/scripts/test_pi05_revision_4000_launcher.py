"""Regression tests for the 4,000-update PrimingVLA rebuttal launchers.

DRY_RUN exercises the actual shell dispatch without importing ML dependencies.
"""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_LAUNCHER = ROOT / "examples/training/train_pi05_revision_4000.sh"
EVAL_LAUNCHER = ROOT / "run_eval_pi05_revision_4000.sh"
MULTI_LAUNCHER = ROOT / "examples/training/launch_pi05_revision_4gpu.sh"


class Revision4000LauncherTest(unittest.TestCase):
    def test_default_tokenizer_path_is_on_autodl_fs(self):
        self.assertIn(
            'export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-fs/ckpt/google/paligemma-3b-pt-224}"',
            TRAIN_LAUNCHER.read_text(),
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pi05-revision-4000-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

        for dirname in ("dataset", "pretrained", "tokenizer"):
            (self.work / dirname).mkdir()

        self.env = dict(
            os.environ,
            DATASET_ROOT=str(self.work / "dataset"),
            PRETRAINED_PATH=str(self.work / "pretrained"),
            TOKENIZER_PATH=str(self.work / "tokenizer"),
            OUTPUT_ROOT=str(self.work / "checkpoints"),
            LOG_ROOT=str(self.work / "logs"),
            DRY_RUN="true",
            GPU_IDS="0",
            NUM_PROCESSES="1",
        )
        for variable in ("RUN_NAME", "SAVE_FREQ", "NTK_SAVE_STAGE_SNAPSHOTS"):
            self.env.pop(variable, None)

    def run_train(self, variant, seed="0", *args, **env_overrides):
        return subprocess.run(
            ["bash", str(TRAIN_LAUNCHER), variant, seed, *args],
            env=dict(self.env, **env_overrides),
            capture_output=True,
            text=True,
            check=False,
        )

    def dry_run_args(self, variant, seed="0", *args, **env_overrides):
        result = self.run_train(variant, seed, *args, **env_overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        command_lines = [line for line in result.stdout.splitlines() if line.startswith("env ")]
        self.assertEqual(len(command_lines), 1, result.stdout)
        return shlex.split(command_lines[0]), result.stdout

    def test_every_variant_has_exactly_4000_total_updates(self):
        cases = (
            ("direct_dual", 4000, 0, 0, "true", "0.0"),
            ("dual_prompt_only", 4000, 0, 0, "false", "0.0"),
            ("full_reference", 3000, 1000, 250, "true", "0.0"),
            ("action_flow_warmup", 3000, 1000, 250, "true", "1.0"),
        )
        for variant, flow, pretrain, bridge, cabo, full_mask_probability in cases:
            with self.subTest(variant=variant):
                cmd, stdout = self.dry_run_args(variant)
                for arg in (
                    f"--steps={flow}",
                    f"--policy.next_action_pretrain_steps={pretrain}",
                    f"--policy.next_action_bridge_steps={bridge}",
                    f"--policy.cabo_enabled={cabo}",
                    f"--policy.next_action_full_mask_probability={full_mask_probability}",
                    "--policy.num_prompt_tokens=16",
                    "--policy.num_vlm_prompt_tokens=16",
                    "--save_freq=1000",
                    "--seed=0",
                ):
                    self.assertIn(arg, cmd)
                self.assertIn("total=4000", stdout)
                self.assertIn(f"--output_dir={self.work / 'checkpoints' / f'pi05-revision-{variant}-seed0-total4000'}", cmd)
                self.assertFalse((self.work / "checkpoints").exists())
                self.assertFalse((self.work / "logs").exists())

    def test_external_flow_override_cannot_break_matched_budget(self):
        for variant, expected_steps in (("direct_dual", 4000), ("full_reference", 3000)):
            with self.subTest(variant=variant):
                cmd, _ = self.dry_run_args(variant, "2", FLOW_STEPS="9999")
                self.assertIn(f"--steps={expected_steps}", cmd)
                self.assertIn("--seed=2", cmd)

    def test_protocol_defining_cli_overrides_are_rejected(self):
        for arg in (
            "--steps=3000",
            "--policy.next_action_pretrain_steps=1000",
            "--policy.next_action_bridge_steps=250",
            "--policy.next_action_full_mask_probability=1.0",
            "--policy.cabo_enabled=false",
            "--policy.num_prompt_tokens=32",
            "--seed=999",
        ):
            with self.subTest(arg=arg):
                result = self.run_train("direct_dual", "0", arg)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Protocol-defining", result.stderr)

    def test_non_protocol_extra_args_are_forwarded(self):
        cmd, _ = self.dry_run_args("direct_dual", "1", "--wandb.notes=revision matched 4000")
        self.assertIn("--wandb.notes=revision matched 4000", cmd)

    def test_existing_output_is_not_overwritten(self):
        (self.work / "checkpoints" / "pi05-revision-direct_dual-seed0-total4000").mkdir(parents=True)
        result = self.run_train("direct_dual")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)

    def test_eval_launcher_help_describes_distinct_checkpoint_steps(self):
        result = subprocess.run(
            ["bash", str(EVAL_LAUNCHER), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("004000", result.stdout)
        self.assertIn("003000", result.stdout)

    def test_eval_resolves_correct_cumulative_checkpoint_without_starting_gpu(self):
        # Intentionally use missing checkpoint directories: the existing eval
        # script must print its selected path and exit before importing torch.
        for variant, checkpoint in (
            ("direct_dual", "004000"),
            ("full_reference", "003000"),
            ("action_flow_warmup", "003000"),
        ):
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(EVAL_LAUNCHER), variant, "libero_10", "1"],
                    env=dict(
                        self.env,
                        CKPT_ROOT=str(self.work / "checkpoints"),
                        BASE_OUTPUT=str(self.work / "eval"),
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                expected = (
                    self.work
                    / "checkpoints"
                    / f"pi05-revision-{variant}-seed1-total4000"
                    / "checkpoints"
                    / checkpoint
                    / "pretrained_model"
                )
                self.assertIn(str(expected), result.stdout)
                self.assertIn("ERROR: checkpoint not found", result.stdout)

    def test_four_gpu_launcher_schedules_only_remaining_three_by_default(self):
        result = subprocess.run(
            ["bash", str(MULTI_LAUNCHER), "others", "0"],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU 1: dual_prompt_only", result.stdout)
        self.assertIn("GPU 2: full_reference", result.stdout)
        self.assertIn("GPU 3: action_flow_warmup", result.stdout)
        self.assertNotIn("GPU 0: direct_dual", result.stdout)
        for gpu in (1, 2, 3):
            self.assertIn(f"CUDA_VISIBLE_DEVICES={gpu}", result.stdout)
        self.assertFalse((self.work / "checkpoints").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_four_gpu_preflight_refuses_existing_checkpoints(self):
        (self.work / "checkpoints" / "pi05-revision-full_reference-seed0-total4000").mkdir(parents=True)
        result = subprocess.run(
            ["bash", str(MULTI_LAUNCHER), "others", "0"],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
