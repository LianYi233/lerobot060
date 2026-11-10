"""Dependency-free executable launcher/provenance contracts for the LoRA baseline."""

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_pi05_revision_4000.sh"


class LoRABaselineLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-lora-launcher-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        for name in ("dataset", "base", "tokenizer"):
            (self.work / name).mkdir()
        self.env = dict(
            os.environ,
            DRY_RUN="true",
            DATASET_ROOT=str(self.work / "dataset"),
            DATASET_REPO_ID="libero",
            PRETRAINED_PATH=str(self.work / "base"),
            TOKENIZER_PATH=str(self.work / "tokenizer"),
            OUTPUT_ROOT=str(self.work / "outputs"),
            LOG_ROOT=str(self.work / "logs"),
            GPU_IDS="2",
            NUM_PROCESSES="1",
            BATCH_SIZE="32",
            DTYPE="float32",
            MIXED_PRECISION="no",
            VIDEO_BACKEND="pyav",
            COMPILE_MODEL="true",
            GRADIENT_CHECKPOINTING="true",
            PYTHON_BIN=sys.executable,
        )
        for key in ("SAVE_FREQ", "RUN_NAME", "REFERENCE_TRAIN_CONFIG", "REFERENCE_NUM_PROCESSES"):
            self.env.pop(key, None)

    def run_launcher(self, variant="lora_r16", seed="0", *extra, **env):
        return subprocess.run(
            ["bash", str(LAUNCHER), variant, seed, *extra],
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
            check=False,
        )

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("env ")]
        self.assertEqual(len(lines), 1, result.stdout)
        return shlex.split(lines[0])

    def make_reference(self, *, staged=False, processes=1):
        model_dir = self.work / "reference/checkpoints/003000/pretrained_model"
        model_dir.mkdir(parents=True)
        cfg = {
            "policy": {
                "type": "pi05",
                "training_stage": "flow",
                "pretrained_path": self.env["PRETRAINED_PATH"],
                "next_action_pretrain_steps": 0,
                "tokenizer_name": self.env["TOKENIZER_PATH"],
                "dtype": "float32",
            },
            "dataset": {"repo_id": "libero", "root": self.env["DATASET_ROOT"], "video_backend": "pyav"},
            "batch_size": 32,
            "steps": 4000,
        }
        if staged:
            transfer_dir = self.work / "reference_next_action_pretrain/checkpoints/001000/pretrained_model"
            transfer_dir.mkdir(parents=True)
            stage1 = copy.deepcopy(cfg)
            stage1["policy"]["training_stage"] = "next_action"
            stage1["steps"] = 1000
            (transfer_dir / "train_config.json").write_text(json.dumps(stage1))
            cfg["policy"]["pretrained_path"] = str(transfer_dir)
            cfg["policy"]["next_action_pretrain_steps"] = 1000
            cfg["steps"] = 3000
        path = model_dir / "train_config.json"
        path.write_text(json.dumps(cfg))
        if processes is not None:
            state_dir = model_dir.parent / "training_state"
            state_dir.mkdir()
            (state_dir / "training_step.json").write_text(
                json.dumps({"num_processes": processes, "batch_size": 32})
            )
        return path

    def test_external_lora_is_prompt_free_direct_4000_native_peft(self):
        result = self.run_launcher(seed="2", FLOW_STEPS="9999")
        cmd = self.command(result)
        for arg in (
            "--steps=4000", "--policy.adaptation_mode=lora", "--policy.training_stage=flow",
            "--policy.num_vlm_prompt_tokens=0", "--policy.num_prompt_tokens=0",
            "--policy.next_action_pretrain_steps=0", "--policy.next_action_bridge_steps=0",
            "--policy.cabo_enabled=false", "--policy.action_prompt_lr_scale=1.0",
            "--peft.method_type=LORA", "--peft.r=16", "--peft.lora_alpha=16",
            "--save_freq=1000", "--seed=2", "--num_processes=1", "CUDA_VISIBLE_DEVICES=2",
        ):
            self.assertIn(arg, cmd)
        self.assertFalse(any(arg.startswith("--policy.use_peft") for arg in cmd))
        self.assertIn("total=4000 CPC=false", result.stdout)
        self.assertIn("historical run parity is not verified", result.stdout)
        self.assertIn("pi05-revision-lora_r16-seed2-total4000", result.stdout)
        self.assertFalse((self.work / "outputs").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_common_data_base_batch_precision_flags_match_direct_dual(self):
        direct = self.command(self.run_launcher("direct_dual"))
        lora = self.command(self.run_launcher())
        prefixes = (
            "--dataset.", "--policy.pretrained_path=", "--policy.tokenizer_name=", "--policy.dtype=",
            "--policy.device=", "--policy.compile_model=", "--policy.gradient_checkpointing=",
            "--batch_size=", "--steps=", "--seed=", "--mixed_precision=", "--num_processes=",
        )
        self.assertEqual(
            [value for value in direct if value.startswith(prefixes)],
            [value for value in lora if value.startswith(prefixes)],
        )

    def test_protocol_bypasses_are_rejected(self):
        for arg in (
            "--policy.path=/tmp/primed", "--config_path=other.json", "--policy.use_peft=true",
            "--policy.adaptation_mode=prompt", "--peft.r=64", "--peft=null",
            "--dataset.root=/tmp/other", "--dataset.episodes=[0]", "--batch_size=1",
            "--optimizer.lr=0.001", "--use_policy_training_preset=false", "--steps=3000",
            "--policy.num_prompt_tokens=16", "--policy.cabo_enabled=true",
        ):
            with self.subTest(arg=arg):
                result = self.run_launcher("lora_r16", "0", arg)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("Protocol-defining", result.stderr)

    def test_explicit_policy_lr_tuning_and_log_options_are_forwarded(self):
        cmd = self.command(
            self.run_launcher(
                "lora_r16", "1", "--policy.optimizer_lr=5e-5",
                "--policy.scheduler_decay_lr=2e-5", "--log_freq=20",
            )
        )
        self.assertIn("--policy.optimizer_lr=5e-5", cmd)
        self.assertIn("--policy.scheduler_decay_lr=2e-5", cmd)
        self.assertIn("--log_freq=20", cmd)

    def test_existing_output_is_preserved(self):
        (self.work / "outputs/pi05-revision-lora_r16-seed0-total4000").mkdir(parents=True)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)

    def test_reference_parity_accepts_direct_run(self):
        reference = self.make_reference()
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference))
        self.command(result)
        self.assertIn("reference-check: PASS", result.stdout)
        self.assertIn("effective_batch=32", result.stdout)

    def test_reference_parity_follows_staged_transfer_to_original_base(self):
        reference = self.make_reference(staged=True)
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference))
        cmd = self.command(result)
        self.assertIn("recovered original initialization", result.stdout)
        self.assertIn(f"--policy.pretrained_path={self.work / 'base'}", cmd)
        self.assertIn("updates=4000", result.stdout)

    def test_reference_rejects_primed_initialization_and_missing_transfer_metadata(self):
        reference = self.make_reference(staged=True)
        transfer = json.loads(reference.read_text())["policy"]["pretrained_path"]
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference), PRETRAINED_PATH=transfer)
        self.assertEqual(result.returncode, 2)
        self.assertIn("original_base", result.stderr)
        (Path(transfer) / "train_config.json").unlink()
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot recover original initialization", result.stderr)

    def test_reference_rejects_changed_effective_batch(self):
        reference = self.make_reference()
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference), NUM_PROCESSES="3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("effective_batch", result.stderr)

    def test_reference_unknown_world_size_needs_explicit_verified_value(self):
        reference = self.make_reference(processes=None)
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference))
        self.assertEqual(result.returncode, 2)
        self.assertIn("REFERENCE_NUM_PROCESSES", result.stderr)
        self.command(self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference), REFERENCE_NUM_PROCESSES="1"))

    def test_reference_rejects_unmatched_3000_step_run(self):
        reference = self.make_reference()
        cfg = json.loads(reference.read_text())
        cfg["steps"] = 3000
        reference.write_text(json.dumps(cfg))
        result = self.run_launcher(REFERENCE_TRAIN_CONFIG=str(reference))
        self.assertEqual(result.returncode, 2)
        self.assertIn("total_updates", result.stderr)


if __name__ == "__main__":
    unittest.main()
