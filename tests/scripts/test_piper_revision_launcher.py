"""Exercise actual dispatch, budget accounting and safe export with local tiny fixtures."""

import importlib.util
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_piper_revision.sh"
SPEC = importlib.util.spec_from_file_location("piper_revision_launcher", LAUNCHER.with_suffix(".py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def model_file(path, changed=0):
    header = json.dumps({
        "backbone": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "learned": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1, changed))


class PiperRevisionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="piper revision ")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.base = self.work / "original base"
        self.base.mkdir()
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.base / name).write_text("{}")
        model_file(self.base / "model.safetensors")
        tokenizer = self.work / "tokenizer"
        tokenizer.mkdir()
        for name in ("tokenizer_config.json", "tokenizer.json"):
            (tokenizer / name).write_text("{}")
        names = [f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
        for folder in MODULE.TASKS.values():
            dataset = self.work / "data" / folder
            files = ["meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet", "data/chunk-000/file-000.parquet"]
            files += [f"videos/observation.images.{camera}/chunk-000/file-000.mp4" for camera in ("wrist", "front")]
            for name in files:
                path = dataset / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            info = {
                "codebase_version": "v3.0", "fps": 10, "total_tasks": 1,
                "total_episodes": 10, "total_frames": 100,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    **{key: {"dtype": "float32", "shape": [7], "names": names} for key in ("action", "observation.state")},
                    **{f"observation.images.{camera}": {"dtype": "video", "shape": [224, 224, 3]} for camera in ("wrist", "front")},
                },
            }
            (dataset / "meta/info.json").write_text(json.dumps(info))
            (dataset / "meta/stats.json").write_text(json.dumps({key: {"q01": [0] * 7, "q99": [1] * 7} for key in ("action", "observation.state")}))
        self.env = dict(os.environ)
        for key in ("OUTPUT_ROOT", "RUN_GROUP", "GPU_ID", "GPU_IDS", "TOTAL_UPDATES", "CUMULATIVE_CHECKPOINTS", "WORLD_SIZE", "EVAL_SPLIT", "NUM_PROCESSES", "BATCH_SIZE", "TASKS"):
            self.env.pop(key, None)
        self.env.update(WORK_ROOT=str(self.work), PRETRAINED_PATH=str(self.base), DATASET_BASE=str(self.work / "data"), TOKENIZER_PATH=str(tokenizer), DRY_RUN="true", PYTHON=sys.executable)

    def run_launcher(self, *args, **env):
        return subprocess.run(["bash", str(LAUNCHER), *args], env={**self.env, **env}, text=True, capture_output=True, check=False)

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [dict(arg[2:].split("=", 1) for arg in shlex.split(line) if arg.startswith("--")) for line in result.stdout.splitlines() if line.startswith("env ")]

    def test_actual_shell_core_has_equal_cumulative_budget_and_two_milestones(self):
        commands = self.commands(self.run_launcher("core", "0"))
        self.assertEqual(len(commands), 16)
        self.assertEqual(len({command["output_dir"] for command in commands}), 16)
        for command in commands:
            prep = int(command["policy.next_action_pretrain_steps"])
            self.assertEqual(int(command["steps"]) + prep, 4000)
            self.assertEqual([step + prep for step in json.loads(command["save_steps"])], [3000, 4000])
            self.assertEqual(command["policy.num_prompt_tokens"], "16")
            self.assertEqual(command["policy.num_vlm_prompt_tokens"], "16")
            self.assertEqual(command["policy.piper_gripper_history_steps"], "0")
            self.assertEqual(command["policy.train_action_head"], "false")
            self.assertEqual(command["dataset.eval_split"], "0.1")
        self.assertFalse((self.work / "runs").exists())

    def test_warmup_is_full_mask_unconditioned_and_no_bridge_reassigns_budget(self):
        commands = self.commands(self.run_launcher("flow_warmup,no_bridge", "0", "--tasks", "1"))
        warmup, no_bridge = commands
        self.assertEqual(warmup["policy.next_action_full_mask_probability"], "1.0")
        self.assertEqual(warmup["policy.next_action_masked_steps"], "50")
        self.assertEqual(warmup["policy.next_action_bridge_steps"], "250")
        self.assertEqual(no_bridge["policy.next_action_pretrain_steps"], "1000")
        self.assertEqual(no_bridge["policy.next_action_bridge_steps"], "0")

    def test_lr_sweep_and_seed_repetitions_keep_the_same_data_and_policy_contract(self):
        commands = self.commands(self.run_launcher("cpc_sweep", "0,1,2", "--tasks", "1", VLM_PROMPT_TOKENS="128", GRIPPER_HISTORY_STEPS="20", FLOW_STEPS="12000"))
        self.assertEqual(len(commands), 12)
        self.assertEqual({command["policy.action_prompt_lr_scale"] for command in commands}, {"0.5", "0.1", "0.03", "0.01"})
        for command in commands:
            self.assertEqual(command["policy.cabo_enabled"], "false")
            self.assertEqual(command["policy.num_vlm_prompt_tokens"], "16")
            self.assertEqual(command["policy.piper_gripper_history_steps"], "0")
            self.assertEqual(command["steps"], "3000")
            self.assertEqual(command["piper_eval.seed"], "20261107")

    def test_larger_budget_is_explicit_and_applies_to_every_method(self):
        commands = self.commands(self.run_launcher("core", "0", "--tasks", "1", TOTAL_UPDATES="12000", CUMULATIVE_CHECKPOINTS="[4000,12000]"))
        for command in commands:
            prep = int(command["policy.next_action_pretrain_steps"])
            self.assertEqual(int(command["steps"]) + prep, 12000)
            self.assertEqual([step + prep for step in json.loads(command["save_steps"])], [4000, 12000])

    def test_invalid_protocols_fail_before_any_training_output(self):
        for env in ({"GPU_IDS": "0,1"}, {"NUM_PROCESSES": "2"}, {"EVAL_SPLIT": "0"}, {"TOTAL_UPDATES": "1000"}, {"CUMULATIVE_CHECKPOINTS": "[1000,4000]"}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_launcher("core", "0", **env).returncode, 0)
                self.assertFalse((self.work / "runs").exists())

    def test_numerically_duplicate_seeds_are_rejected(self):
        for seeds in ("0,00", "1,+1"):
            result = self.run_launcher("core", seeds)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("distinct", result.stderr)
            self.assertFalse((self.work / "runs").exists())

    def test_new_method_cannot_change_a_locked_task_dataset(self):
        contract = self.work / "task_contract.json"
        MODULE.lock_json_contract(contract, {"data_hash": "first", "train_episodes": [0, 1]})
        MODULE.lock_json_contract(contract, {"data_hash": "first", "train_episodes": [0, 1]})
        with self.assertRaisesRegex(ValueError, "contract differs"):
            MODULE.lock_json_contract(contract, {"data_hash": "changed", "train_episodes": [0, 1, 2]})
        self.assertEqual(MODULE.read_json(contract)["data_hash"], "first")

    def test_adapted_head_base_is_rejected(self):
        (self.base / "config.json").write_text(json.dumps({"train_action_head": True}))
        result = self.run_launcher("core", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("original base", result.stderr)

    def test_missing_later_task_fails_before_first_run(self):
        (self.work / "data" / MODULE.TASKS["4"] / "meta/info.json").unlink()
        result = self.run_launcher("core", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.work / "runs").exists())

    def test_audit_uses_true_step_peak_and_rejects_skipped_optimizer_updates(self):
        out = self.work / "audit-run"
        out.mkdir()
        job = {"output_dir": str(out), "recipe": MODULE.recipe("direct_cpc")}
        MODULE.write_json(out / "trainable_parameters.json", {
            "trainable_total": 49152, "categories": {"vlm_prompt": 32768, "action_prompt": 16384},
        })
        MODULE.write_json(out / "dataset_split.json", {"train_episodes": [0, 1], "eval_episodes": [2]})
        MODULE.write_json(out / "piper_normalization.json", {"training_episodes": [0, 1]})
        MODULE.write_json(out / "checkpoints/004000/pretrained_model/config.json", {
            "training_stage": "flow", "num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16,
            "train_action_head": False, "train_action_projections": False,
            "train_action_expert_last_n_layers": 0, "piper_gripper_history_steps": 0,
            "piper_train_normalization": True, "use_relative_actions": False,
            "cabo_enabled": True, "action_prompt_lr_scale": 1.0, "chunk_size": 50,
        })
        record = {"train": {"gpu_mem_gb": 5.0}, "diagnostics_rank0": {
            "resources/step_peak_allocated_gpu_gib": {"mean": 5.0, "max": 9.0},
            "optimizer_step/skipped": {"mean": 0.0, "count": 50},
        }}
        diagnostic = out / "training_diagnostics.jsonl"
        diagnostic.write_text(json.dumps(record) + "\n")
        self.assertEqual(MODULE.audit_run(job)["peak_allocated_gpu_gib"], 9.0)
        record["diagnostics_rank0"]["optimizer_step/skipped"]["mean"] = 0.02
        diagnostic.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "skipped optimizer"):
            MODULE.audit_run(job)

    def test_compaction_roundtrip_preserves_shared_base_and_unrelated_files(self):
        paths = MODULE.paths_from_env(self.env)
        paths["output"].mkdir(parents=True)
        out = paths["output"] / "owned"
        job = {"output_dir": str(out), "task_id": "1", "recipe": MODULE.recipe("direct_cpc"), "run_name": "owned"}
        original_bytes = []
        for step in (3000, 4000):
            model = out / "checkpoints" / f"{step:06d}" / "pretrained_model"
            model_file(model / "model.safetensors", changed=step)
            original_bytes.append((model / "model.safetensors").read_bytes())
            for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
                (model / name).write_text("{}")
            state = model.parent / "training_state"
            state.mkdir()
            (state / "training_step.json").write_text(json.dumps({"step": step}))
        unrelated = paths["output"] / "unrelated_model.safetensors"
        unrelated.write_bytes(b"never remove me")
        common = {"cumulative_checkpoints": [3000, 4000], "base_sha256": MODULE.sha256_file(self.base / "model.safetensors"), "keep_full": False}
        bundles = MODULE.compact_run(job, paths, common, {"parameters": {"trainable_total": 49152}})
        from revision_checkpoint import materialize_checkpoint
        for item, original in zip(bundles, original_bytes, strict=True):
            restored = self.work / f"restored-{item['cumulative_updates']}"
            materialize_checkpoint(item["bundle"], paths["output"] / "storage_base", restored)
            self.assertEqual((restored / "model.safetensors").read_bytes(), original)
            self.assertTrue((restored / "dataset_info.json").is_file())
            self.assertFalse((Path(item["source_checkpoint"]) / "model.safetensors").exists())
            self.assertTrue((Path(item["source_checkpoint"]).parent / "training_state/training_step.json").is_file())
        self.assertTrue((paths["output"] / "storage_base/model.safetensors").is_file())
        self.assertEqual(unrelated.read_bytes(), b"never remove me")
        storage = paths["output"] / "storage_base"
        shared_sha = MODULE.sha256_file(storage / "model.safetensors")
        MODULE.verify_completed({"checkpoints": bundles}, storage, shared_sha)
        # A surviving directory is not enough to skip a model whose only archive is damaged.
        delta = Path(bundles[1]["bundle"]) / "tensors.delta.bin"
        delta.write_bytes(delta.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt checkpoint"):
            MODULE.verify_completed({"checkpoints": bundles}, storage, shared_sha)


if __name__ == "__main__":
    unittest.main()
