"""Regression tests for the 4,000-update PrimingVLA rebuttal launchers.

DRY_RUN exercises the actual shell dispatch without importing ML dependencies.
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_LAUNCHER = ROOT / "examples/training/train_pi05_revision_4000.sh"
EVAL_LAUNCHER = ROOT / "run_eval_pi05_revision_4000.sh"
MULTI_LAUNCHER = ROOT / "examples/training/launch_pi05_revision_4gpu.sh"
SEQUENTIAL_LAUNCHER = ROOT / "run_eval_pi05_revision_4000_sequential.sh"


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
        self.prefix = self.work / "shared_prefix" / "checkpoints" / "001000"
        (self.prefix / "pretrained_model").mkdir(parents=True)
        (self.prefix / "training_state").mkdir()
        for relpath in (
            "pretrained_model/train_config.json",
            "pretrained_model/model.safetensors",
            "training_state/optimizer_state.safetensors",
            "training_state/optimizer_param_groups.json",
            "training_state/scheduler_state.json",
        ):
            (self.prefix / relpath).touch()

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
            PREFIX_CHECKPOINT_DIR=str(self.prefix),
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
            ("full_no_cpc", 3000, 1000, 250, "false", "0.0"),
            ("action_flow_warmup", 3000, 1000, 250, "true", "1.0"),
            ("fixed_action_lr_050", 3000, 1000, 250, "false", "0.0"),
            ("fixed_action_lr_025", 3000, 1000, 250, "false", "0.0"),
            ("fixed_action_lr_010", 3000, 1000, 250, "false", "0.0"),
            ("optim_carry_carry", 3000, 1000, 250, "true", "0.0"),
            ("optim_reset_carry", 3000, 1000, 250, "true", "0.0"),
            ("optim_carry_restart", 3000, 1000, 250, "true", "0.0"),
            ("optim_reset_restart", 3000, 1000, 250, "true", "0.0"),
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
                    f"--save_freq={flow}",
                    "--preserve_pi05_flow_save_freq=true",
                    "--seed=0",
                ):
                    self.assertIn(arg, cmd)
                self.assertIn("total=4000", stdout)
                self.assertIn(f"--output_dir={self.work / 'checkpoints' / f'pi05-revision-{variant}-seed0-total4000'}", cmd)
                self.assertFalse((self.work / "checkpoints").exists())
                self.assertFalse((self.work / "logs").exists())

    def test_staged_no_cpc_dispatches_to_existing_no_cabo_implementation(self):
        _, stdout = self.dry_run_args("full_no_cpc")
        self.assertIn("variant=no_cabo", stdout)
        self.assertIn("SAP=750 bridge=250 flow=3000 total=4000 CPC=false", stdout)

    def test_fixed_action_lr_variants_dispatch_to_staged_no_cpc_with_unique_names(self):
        for variant, scale in (
            ("fixed_action_lr_050", "0.5"),
            ("fixed_action_lr_025", "0.25"),
            ("fixed_action_lr_010", "0.1"),
        ):
            with self.subTest(variant=variant):
                cmd, stdout = self.dry_run_args(variant, "2", GPU_IDS="1")
                self.assertIn(f"--policy.action_prompt_lr_scale={scale}", cmd)
                self.assertIn("--policy.cabo_enabled=false", cmd)
                self.assertIn("variant=no_cabo", stdout)
                self.assertIn("SAP=750 bridge=250 flow=3000 total=4000 CPC=false", stdout)
                self.assertIn("CUDA_VISIBLE_DEVICES=1", cmd)
                self.assertIn(
                    f"--output_dir={self.work / 'checkpoints' / f'pi05-revision-{variant}-seed2-total4000'}",
                    cmd,
                )

    def test_non_fixed_variants_keep_default_action_lr_scale(self):
        for variant in ("full_reference", "full_no_cpc", "direct_dual", "dual_prompt_only", "action_flow_warmup"):
            with self.subTest(variant=variant):
                cmd, _ = self.dry_run_args(variant)
                self.assertIn("--policy.action_prompt_lr_scale=1.0", cmd)

    def test_optimizer_prefix_only_runs_stages_one_and_two(self):
        cmd, output = self.dry_run_args("optim_prefix", "3", GPU_IDS="0")
        self.assertIn("--pi05_stage1_only=true", cmd)
        self.assertIn("--policy.next_action_pretrain_steps=1000", cmd)
        self.assertIn("--policy.next_action_bridge_steps=250", cmd)
        self.assertIn("-prefix1000", output)

    def test_optimizer_transition_cells_use_same_prefix_and_distinct_states(self):
        for variant, moments, scheduler in (
            ("optim_carry_carry", "carry", "carry"),
            ("optim_reset_carry", "reset", "carry"),
            ("optim_carry_restart", "carry", "restart"),
            ("optim_reset_restart", "reset", "restart"),
        ):
            with self.subTest(variant=variant):
                cmd, output = self.dry_run_args(variant, "3", GPU_IDS="2")
                self.assertIn(f"--policy.stage3_adamw_moments={moments}", cmd)
                self.assertIn(f"--policy.stage3_lr_scheduler={scheduler}", cmd)
                self.assertIn(f"--pi05_stage3_transfer_checkpoint={self.prefix}", cmd)
                self.assertIn(f"--policy.pretrained_path={self.prefix / 'pretrained_model'}", cmd)
                self.assertIn("--steps=3000", cmd)
                self.assertIn("--seed=3", cmd)
                self.assertIn("CUDA_VISIBLE_DEVICES=2", cmd)
                self.assertIn("variant=full_reference", output)

    def test_transition_rejects_missing_source_state(self):
        result = self.run_train("optim_carry_carry", "3", PREFIX_CHECKPOINT_DIR="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("require PREFIX_CHECKPOINT_DIR", result.stderr)
        (self.prefix / "training_state/scheduler_state.json").unlink()
        result = self.run_train("optim_reset_restart", "3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing Stage-II state file", result.stderr)

    def test_transition_rejects_multiple_processes(self):
        result = self.run_train("optim_reset_carry", "3", NUM_PROCESSES="3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("NUM_PROCESSES=1", result.stderr)

    def test_transition_flags_cannot_be_overridden(self):
        for arg in ("--policy.stage3_adamw_moments=reset", "--policy.stage3_lr_scheduler=restart"):
            with self.subTest(arg=arg):
                result = self.run_train("optim_carry_carry", "3", arg)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Protocol-defining", result.stderr)

    def test_checkpoint_cadence_can_be_opted_back_into(self):
        cmd, _ = self.dry_run_args("direct_dual", SAVE_FREQ="1000")
        self.assertIn("--save_freq=1000", cmd)

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
            "--policy.action_prompt_lr_scale=0.1",
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
            ("full_no_cpc", "003000"),
            ("action_flow_warmup", "003000"),
            ("fixed_action_lr_050", "003000"),
            ("fixed_action_lr_025", "003000"),
            ("fixed_action_lr_010", "003000"),
            ("optim_carry_carry", "003000"),
            ("optim_reset_carry", "003000"),
            ("optim_carry_restart", "003000"),
            ("optim_reset_restart", "003000"),
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

    def test_fixed_action_lr_sequential_eval_reuses_complete_results_and_prints_only_suites(self):
        variants = ("fixed_action_lr_050", "fixed_action_lr_025", "fixed_action_lr_010")
        eval_root = self.work / "eval"
        suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        for variant in variants:
            out = eval_root / (
                f"libero060-all-pi05-revision-{variant}-seed0-total4000-003000-libero-all4-resume"
            )
            out.mkdir(parents=True)
            (out / "eval_info.json").write_text(json.dumps({
                "per_group": {suite: {"n_episodes": 100, "pc_success": 80.0} for suite in suites},
                "overall": {"n_episodes": 400, "pc_success": 80.0},
            }))
        result = subprocess.run(
            ["bash", str(SEQUENTIAL_LAUNCHER), "0", *variants],
            env=dict(
                self.env,
                BASE_OUTPUT=str(eval_root),
                EPISODES_PER_TASK="10",
                GPU_ID="3",
                PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("[RUN]", result.stdout)
        self.assertNotIn(".mp4", result.stdout)
        for variant in variants:
            self.assertIn(f"[SKIP] {variant}", result.stdout)
            self.assertTrue(any(line.split() == [variant, *(["80.00%"] * 5)] for line in result.stdout.splitlines()))

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
