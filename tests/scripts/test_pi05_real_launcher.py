"""Exercise real shell dispatch using metadata fixtures, without loading ML dependencies."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_pi05_real.sh"
SPEC = importlib.util.spec_from_file_location("real_launcher", LAUNCHER.with_suffix(".py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RealLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pi05 real test ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.env = dict(
            os.environ,
            WORK_ROOT=str(self.work),
            DRY_RUN="true",
            PYTHON=sys.executable,
            GPU_IDS="0",
            NUM_PROCESSES="1",
        )
        for name in (
            "DATASET_BASE",
            "PRETRAINED_PATH",
            "TOKENIZER_PATH",
            "OUTPUT_ROOT",
            "LOG_ROOT",
            "FLOW_STEPS",
            "CHUNK_SIZE",
            "MASKED_STEPS",
            "N_ACTION_STEPS",
            "NORMALIZATION_MODE",
            "SAVE_FREQ",
            "BATCH_SIZE",
            "RUN_NAME",
        ):
            self.env.pop(name, None)
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            self.touch(self.work / "models/pi05_libero_base" / filename)
        for filename in ("tokenizer_config.json", "tokenizer.json"):
            self.touch(self.work / "models/google/paligemma-3b-pt-224" / filename)
        for folder in MODULE.TASKS.values():
            self.make_dataset(folder)

    @staticmethod
    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def make_dataset(self, folder):
        root = self.work / "datasets/May-pick-and-place" / folder
        for filename in (
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.wrist/chunk-000/file-000.mp4",
        ):
            self.touch(root / filename)
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_episodes": 2,
            "total_frames": 120,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [7]},
                "observation.images.wrist": {"dtype": "video", "shape": [240, 320, 3]},
            },
        }
        (root / "meta/info.json").write_text(json.dumps(info))
        stats = {
            key: {"q01": [0.0] * 7, "q99": [1.0] * 7, "mean": [0.0] * 7, "std": [1.0] * 7}
            for key in ("observation.state", "action")
        }
        (root / "meta/stats.json").write_text(json.dumps(stats))

    def run_launcher(self, *args, **env):
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            env=dict(self.env, **env),
            capture_output=True,
            text=True,
            check=False,
        )

    def command_args(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("env ")]

    def test_all_dispatches_four_isolated_runs_and_preserves_quoted_args(self):
        result = self.run_launcher("all", "full_reference", "2", "--wandb.notes=real robot run")
        commands = self.command_args(result)
        self.assertEqual(len(commands), 4)
        outputs = []
        for command, folder in zip(commands, MODULE.TASKS.values(), strict=True):
            self.assertIn(f"--dataset.root={self.work / 'datasets/May-pick-and-place' / folder}", command)
            self.assertIn(f"--dataset.repo_id=may-pick-and-place/{folder}", command)
            self.assertIn("--steps=3000", command)
            self.assertIn("--policy.next_action_pretrain_steps=1000", command)
            self.assertIn("--policy.next_action_bridge_steps=250", command)
            self.assertIn("--wandb.notes=real robot run", command)
            self.assertIn("--save_freq=500", command)
            outputs.extend(item for item in command if item.startswith("--output_dir="))
        self.assertEqual(len(set(outputs)), 4)
        self.assertFalse((self.work / "chkpt").exists())
        self.assertFalse((self.work / "logs").exists())

    def test_variants_share_existing_recipe(self):
        for variant, pretrain, bridge, cabo in (
            ("no_bridge", 1000, 0, "true"),
            ("direct_dual", 0, 0, "true"),
            ("dual_prompt_only", 0, 0, "false"),
            ("vlm_only", 0, 0, "false"),
            ("action_only", 0, 0, "false"),
            ("no_cabo", 1000, 250, "false"),
        ):
            with self.subTest(variant=variant):
                (command,) = self.command_args(self.run_launcher("1", variant, "0"))
                self.assertIn(f"--policy.next_action_pretrain_steps={pretrain}", command)
                self.assertIn(f"--policy.next_action_bridge_steps={bridge}", command)
                self.assertIn(f"--policy.cabo_enabled={cabo}", command)

    def test_short_chunks_multi_gpu_and_full_task_name(self):
        (command,) = self.command_args(
            self.run_launcher(
                MODULE.TASKS["3"], "full_reference", "0", CHUNK_SIZE="8", GPU_IDS="0,1", NUM_PROCESSES="2"
            )
        )
        for arg in (
            "--policy.chunk_size=8",
            "--policy.n_action_steps=8",
            "--policy.next_action_masked_steps=6",
            "--multi_gpu",
            "--num_processes=2",
            "CUDA_VISIBLE_DEVICES=0,1",
        ):
            self.assertIn(arg, command)

    def test_checks_all_datasets_before_first_launch(self):
        root = self.work / "datasets/May-pick-and-place" / MODULE.TASKS["4"]
        (root / "meta/tasks.parquet").unlink()
        result = self.run_launcher("all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("meta/tasks.parquet", result.stderr)
        self.assertNotIn("Launching:", result.stdout)

    def test_quantile_failure_and_explicit_mean_std(self):
        path = self.work / "datasets/May-pick-and-place" / MODULE.TASKS["1"] / "meta/stats.json"
        stats = json.loads(path.read_text())
        del stats["action"]["q01"]
        path.write_text(json.dumps(stats))
        result = self.run_launcher("1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing action.q01", result.stderr)
        (command,) = self.command_args(self.run_launcher("1", NORMALIZATION_MODE="MEAN_STD"))
        norm_arg = next(item for item in command if item.startswith("--policy.normalization_mapping="))
        self.assertEqual(json.loads(norm_arg.split("=", 1)[1])["ACTION"], "MEAN_STD")

    def test_existing_priming_output_and_invalid_gpu_configuration(self):
        result = self.run_launcher("1", GPU_IDS="0,1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NUM_PROCESSES", result.stderr)
        name = f"pi05-may-{MODULE.TASKS['1']}-full_reference-seed0_next_action_pretrain"
        (self.work / "chkpt/2601-lerobot/prompt-ablation-real" / name).mkdir(parents=True)
        result = self.run_launcher("1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)

    def test_feature_and_path_overrides_cannot_bypass_preflight(self):
        result = self.run_launcher("1", "full_reference", "0", "--policy.path=/wrong/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment variables", result.stderr)


if __name__ == "__main__":
    unittest.main()
