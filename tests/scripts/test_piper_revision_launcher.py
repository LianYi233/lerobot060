"""Exercise actual dispatch, budget accounting and safe export with local tiny fixtures."""

import importlib.util
import io
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/training/train_piper_revision.sh"
LONG_LAUNCHER = ROOT / "examples/training/train_piper_revision_long.sh"
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
        for key in ("OUTPUT_ROOT", "RUN_GROUP", "GPU_ID", "GPU_IDS", "TOTAL_UPDATES", "CUMULATIVE_CHECKPOINTS", "WORLD_SIZE", "EVAL_SPLIT", "NUM_PROCESSES", "BATCH_SIZE", "TASKS", "LOSS_TARGET", "LOSS_WINDOW_UPDATES"):
            self.env.pop(key, None)
        self.env.update(WORK_ROOT=str(self.work), PRETRAINED_PATH=str(self.base), DATASET_BASE=str(self.work / "data"), TOKENIZER_PATH=str(tokenizer), DRY_RUN="true", PYTHON=sys.executable)

    def run_launcher(self, *args, **env):
        return subprocess.run(["bash", str(LAUNCHER), *args], env={**self.env, **env}, text=True, capture_output=True, check=False)

    def run_long_launcher(self, *args, **env):
        return subprocess.run(["bash", str(LONG_LAUNCHER), *args], env={**self.env, **env}, text=True, capture_output=True, check=False)

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

    def test_long_preset_trains_four_independent_models_in_requested_order(self):
        result = self.run_long_launcher()
        commands = self.commands(result)
        self.assertEqual(len(commands), 4)
        self.assertEqual([Path(command["dataset.root"]).name for command in commands], [MODULE.TASKS[task] for task in ("2", "3", "4", "1")])
        self.assertEqual(len({command["output_dir"] for command in commands}), 4)
        for command in commands:
            self.assertEqual(command["policy.pretrained_path"], str(self.base))
            self.assertIn("-priming-seed0-total50000", command["job_name"])
            self.assertEqual(Path(command["output_dir"]).parent.name, "reality-long-v1")
            self.assertEqual(command["policy.next_action_pretrain_steps"], "1000")
            self.assertEqual(command["policy.next_action_bridge_steps"], "250")
            self.assertEqual(command["steps"], "49000")
            self.assertEqual(json.loads(command["save_steps"]), [9000, 29000, 49000])
            self.assertEqual(command["policy.scheduler_warmup_steps"], "1000")
            self.assertEqual(command["policy.scheduler_decay_steps"], "49000")
            self.assertEqual(float(command["policy.scheduler_decay_lr"]), 1e-6)
            self.assertEqual(command["policy.cabo_enabled"], "true")
        self.assertIn("Loss target: 1.000e-04", result.stdout)
        self.assertFalse((self.work / "runs").exists())

    def test_long_default_dataset_matches_this_server(self):
        args = types.SimpleNamespace(preset="reality-long", total_updates=None, checkpoints=None, run_group=None, group=None, tasks=None)
        env, group, tasks = MODULE.configure_preset(args, {})
        self.assertEqual(MODULE.paths_from_env(env)["dataset"], Path("/root/autodl-fs/datasets/reality"))
        self.assertEqual(group, "priming")
        self.assertEqual(tasks, "2,3,4,1")

    def test_long_preset_does_not_inherit_a_short_run_budget_or_old_task_order(self):
        result = self.run_long_launcher(TOTAL_UPDATES="4000", CUMULATIVE_CHECKPOINTS="[3000,4000]", TASKS="1", RUN_GROUP="review-v1")
        commands = self.commands(result)
        self.assertEqual([Path(command["dataset.root"]).name for command in commands], [MODULE.TASKS[task] for task in ("2", "3", "4", "1")])
        self.assertTrue(all(command["steps"] == "49000" for command in commands))
        self.assertTrue(all(Path(command["output_dir"]).parent.name == "reality-long-v1" for command in commands))
        self.assertIn("ignores exported", result.stdout)

    def test_long_cli_can_change_budget_and_run_only_second_pair(self):
        commands = self.commands(self.run_long_launcher("priming", "0", "--tasks", "4,1", "--total-updates", "80000", "--checkpoints", "[10000,40000,80000]", "--run-group", "reality-long-80k-v1"))
        self.assertEqual([Path(command["dataset.root"]).name for command in commands], [MODULE.TASKS[task] for task in ("4", "1")])
        for command in commands:
            self.assertEqual(command["steps"], "79000")
            self.assertEqual(command["policy.scheduler_decay_steps"], "79000")
            self.assertEqual(json.loads(command["save_steps"]), [9000, 39000, 79000])
            self.assertEqual(Path(command["output_dir"]).parent.name, "reality-long-80k-v1")

    def test_long_ablation_methods_keep_total_budget_and_own_schedule_horizon(self):
        commands = self.commands(self.run_long_launcher("core", "0", "--tasks", "2"))
        self.assertEqual(len(commands), 4)
        for command in commands:
            self.assertEqual(int(command["steps"]) + int(command["policy.next_action_pretrain_steps"]), 50000)
            self.assertEqual(command["policy.scheduler_decay_steps"], command["steps"])
            self.assertEqual(float(command["policy.scheduler_decay_lr"]), 1e-6)

    def test_loss_target_is_monitor_only_and_invalid_targets_fail(self):
        commands = self.commands(self.run_long_launcher("--tasks", "2", LOSS_TARGET="0.00001", LOSS_WINDOW_UPDATES="1000"))
        self.assertEqual(commands[0]["steps"], "49000")
        for env in ({"LOSS_TARGET": "nan"}, {"LOSS_TARGET": "inf"}, {"LOSS_TARGET": "0"}, {"LOSS_WINDOW_UPDATES": "0"}):
            with self.subTest(env=env):
                result = self.run_long_launcher("--tasks", "2", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.work / "runs").exists())

    def test_status_does_not_require_model_or_dataset_access(self):
        result = self.run_long_launcher("--status", PRETRAINED_PATH="/missing/base", DATASET_BASE="/missing/data", TOKENIZER_PATH="/missing/tokenizer")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reality-long-v1", result.stdout)

    def test_loss_progress_keeps_preparation_separate_and_preserves_small_values(self):
        out = self.work / "loss-progress"
        job = {"output_dir": str(out), "run_name": "task2-priming-seed0-total50000", "recipe": MODULE.recipe("priming", 50000)}
        prep = out.with_name(out.name + "_next_action_pretrain")
        prep.mkdir()
        (prep / "training_diagnostics.jsonl").write_text(json.dumps({"step": 1000, "window_updates": 1000, "train": {"loss": 1e-6}}) + "\n")
        self.assertFalse(MODULE.loss_report(job)["stable_target_met"])
        out.mkdir()
        record = {"step": 500, "window_updates": 500, "training_stage": "flow", "train": {"loss": 0.0001, "update_s": 1.0, "dataloading_s": 0.1}}
        (out / "training_diagnostics.jsonl").write_text(json.dumps(record) + "\n")
        report = MODULE.loss_report(job)
        self.assertTrue(report["stable_target_met"])
        self.assertEqual(report["first_target_cumulative_update"], 1500)
        self.assertEqual(report["last_cumulative_update"], 1500)
        terminal, log = io.StringIO(), io.BytesIO()
        with redirect_stdout(terminal):
            MODULE.progress(job, 600, 1, 4, log_stream=log)
        self.assertIn("1.000e-04", terminal.getvalue())
        self.assertIn("mean[500 updates]", terminal.getvalue())
        self.assertIn("cumulative 1500/50000", terminal.getvalue())
        self.assertEqual(log.getvalue().decode(), terminal.getvalue())

    def test_long_scheduler_is_audited_against_actual_saved_config(self):
        # A command-line schedule is not sufficient evidence if the saved model differs.
        out = self.work / "scheduler-audit"
        out.mkdir()
        job = {"output_dir": str(out), "recipe": MODULE.recipe("direct_cpc", 50000), "adaptation_scheduler": {"warmup_steps": 1000, "decay_steps": 50000, "decay_lr": 1e-6}}
        MODULE.write_json(out / "trainable_parameters.json", {"trainable_total": 49152, "categories": {"vlm_prompt": 32768, "action_prompt": 16384}})
        MODULE.write_json(out / "dataset_split.json", {"train_episodes": [0, 1], "eval_episodes": [2]})
        MODULE.write_json(out / "piper_normalization.json", {"training_episodes": [0, 1]})
        config = {
            "training_stage": "flow", "num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16,
            "train_action_head": False, "train_action_projections": False,
            "train_action_expert_last_n_layers": 0, "piper_gripper_history_steps": 0,
            "piper_train_normalization": True, "use_relative_actions": False,
            "cabo_enabled": True, "action_prompt_lr_scale": 1.0, "chunk_size": 50,
            "scheduler_warmup_steps": 1000, "scheduler_decay_steps": 50000, "scheduler_decay_lr": 1e-6,
        }
        config_path = out / "checkpoints/050000/pretrained_model/config.json"
        MODULE.write_json(config_path, config)
        (out / "training_diagnostics.jsonl").write_text(json.dumps({"diagnostics_rank0": {"optimizer_step/skipped": {"mean": 0, "count": 50}}}) + "\n")
        MODULE.audit_run(job)
        config["scheduler_decay_steps"] = 30000
        MODULE.write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "Saved policy config"):
            MODULE.audit_run(job)

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
