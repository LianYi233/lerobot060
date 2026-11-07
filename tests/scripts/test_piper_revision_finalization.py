"""Run real concurrent checkpoint finalization without torch, a GPU or network access.

Only training computation and the capacity preflight are replaced. Both workers
call the actual execute_job, audit, task-contract, compact and materialize paths.
Run with: uv run --no-project python tests/scripts/test_piper_revision_finalization.py
"""

import importlib.util
import io
import json
import struct
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "piper_revision_finalization_launcher", ROOT / "examples/training/train_piper_revision.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "piper_revision_finalization_checkpoint", ROOT / "examples/piper/revision_checkpoint.py"
)
CHECKPOINT = importlib.util.module_from_spec(CHECKPOINT_SPEC)
CHECKPOINT_SPEC.loader.exec_module(CHECKPOINT)


def write_model(path, value, identity):
    """Two tiny tensors with a shared backbone and distinct checkpoint metadata."""
    header = {
        "__metadata__": {"format": "pt", "identity": identity},
        "backbone": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "prompt": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<fff", 1, 2, value))


class ContendedArtifactLock:
    """Observe ownership while forcing both workers to contend for one real lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.accounting_lock = threading.Lock()
        self.both_attempted = threading.Event()
        self.attempts = 0
        self.owner = None
        self.entries = []

    def __enter__(self):
        with self.accounting_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.both_attempted.set()
        if not self.lock.acquire(timeout=10):
            raise AssertionError("Timed out acquiring the artifact lock")
        self.owner = threading.get_ident()
        self.entries.append(self.owner)
        if not self.both_attempted.wait(timeout=10):
            self.owner = None
            self.lock.release()
            raise AssertionError("Both workers must finish training and attempt finalization")
        return self

    def __exit__(self, *exception):
        self.owner = None
        self.lock.release()


class PiperRevisionFinalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="piper concurrent finalization ")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.paths = {
            "output": self.work / "runs/shared-group",
            "dataset": self.work / "datasets",
            "base": self.work / "original-base",
            "tokenizer": self.work / "tokenizer",
        }
        self.paths["output"].mkdir(parents=True)
        write_model(self.paths["base"] / "model.safetensors", 0, "original initialization")
        self.original_base = (self.paths["base"] / "model.safetensors").read_bytes()
        self.common = {
            "cumulative_checkpoints": [10000, 30000, 50000],
            "base_sha256": MODULE.sha256_file(self.paths["base"] / "model.safetensors"),
            "keep_full": False,
            "reserve_gib": 30,
            "eval_split": 0.1,
            "n_action_steps": 8,
            "batch_size": 8,
            "num_workers": 0,
            "action_eval_freq": 500,
            "action_eval_samples": 32,
            "loss_monitor": {"target": 1e-4, "window_updates": 500, "early_stopping": False},
        }
        self.jobs = []
        for task in ("2", "3"):
            name = f"task{task}-priming-seed0-total50000"
            self.jobs.append({
                "run_name": name,
                "task_id": task,
                "method": "priming",
                "train_seed": 0,
                "recipe": MODULE.recipe("priming", 50000),
                "output_dir": str(self.paths["output"] / name),
                "adaptation_scheduler": {"warmup_steps": 1000, "decay_steps": 49000, "decay_lr": 1e-6},
            })
            MODULE.write_json(
                self.paths["dataset"] / MODULE.TASKS[task] / "meta/info.json",
                {"task_id": task, "fixture": "task-specific dataset metadata"},
            )
        self.unrelated = self.paths["output"] / "unrelated/model.safetensors"
        write_model(self.unrelated, -99, "unrelated checkpoint must survive")
        self.unrelated_bytes = self.unrelated.read_bytes()
        self.expected_models = {}
        self.expected_states = {}

    def create_training_outputs(self, job):
        """Create the artifacts the real audit and compact functions consume."""
        task = job["task_id"]
        output = Path(job["output_dir"])
        preparation = output.with_name(output.name + "_next_action_pretrain")
        split = {"train_episodes": [0, 1], "eval_episodes": [2]}
        normalization = {"task_id": task, "training_episodes": [0, 1]}
        MODULE.write_json(output / "trainable_parameters.json", {
            "trainable_total": 49152,
            "categories": {"vlm_prompt": 32768, "action_prompt": 16384},
        })
        MODULE.write_json(output / "piper_normalization.json", normalization)
        for directory, step, loss, stage in (
            (output, 49000, 1e-5, "flow"),
            (preparation, 1000, 0.1, "stage1"),
        ):
            MODULE.write_json(directory / "dataset_split.json", split)
            record = {
                "step": step,
                "window_updates": 500,
                "training_stage": stage,
                "train": {"loss": loss},
                "diagnostics_rank0": {
                    "optimizer_step/skipped": {"mean": 0.0, "count": 500, "nonfinite": 0},
                    "resources/step_peak_allocated_gpu_gib": {"max": float(task)},
                },
            }
            (directory / "training_diagnostics.jsonl").write_text(json.dumps(record) + "\n")
        config = {
            "training_stage": "flow", "num_vlm_prompt_tokens": 16, "num_prompt_tokens": 16,
            "train_action_head": False, "train_action_projections": False,
            "train_action_expert_last_n_layers": 0, "piper_gripper_history_steps": 0,
            "piper_train_normalization": True, "use_relative_actions": False,
            "cabo_enabled": True, "action_prompt_lr_scale": 1.0, "chunk_size": 50,
            **{f"scheduler_{key}": value for key, value in job["adaptation_scheduler"].items()},
        }
        checkpoints = [
            (output, total - 1000, total) for total in self.common["cumulative_checkpoints"]
        ] + [(preparation, 1000, 1000)]
        snapshots = {}
        states = {}
        for directory, step, total in checkpoints:
            model = directory / "checkpoints" / f"{step:06d}" / "pretrained_model"
            identity = f"task{task}-total{total}"
            write_model(model / "model.safetensors", int(task) * 100000 + total, identity)
            saved_config = {**config, "fixture_identity": identity}
            if directory == preparation:
                saved_config["training_stage"] = "stage1"
            MODULE.write_json(model / "config.json", saved_config)
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                MODULE.write_json(model / name, {"task_id": task, "identity": identity, "file": name})
            statistics = model / "normalizers/state.safetensors"
            write_model(statistics, int(task), f"statistics for {identity}")
            state = model.parent / "training_state"
            MODULE.write_json(state / "training_step.json", {"step": step})
            (state / "optimizer_state.pt").write_bytes(f"optimizer state for {identity}".encode())
            states[str(model)] = {path.name: path.read_bytes() for path in state.iterdir()}
            snapshots[str(model)] = {
                path.relative_to(model).as_posix(): path.read_bytes()
                for path in model.rglob("*") if path.is_file()
            }
            snapshots[str(model)]["dataset_info.json"] = (
                self.paths["dataset"] / MODULE.TASKS[task] / "meta/info.json"
            ).read_bytes()
        return snapshots, states

    def test_overlapping_training_serializes_real_finalization_and_restores_both_tasks(self):
        artifact_lock = ContendedArtifactLock()
        stop_event = threading.Event()
        entered_training = threading.Barrier(2, timeout=10)
        finished_outputs = threading.Barrier(2, timeout=10)
        accounting_lock = threading.Lock()
        active_training = 0
        peak_training = 0
        training_threads = set()

        def train(command, env, log, job, common, index, count, cancel):
            nonlocal active_training, peak_training
            self.assertIs(cancel, stop_event)
            self.assertIsNone(artifact_lock.owner, "Training must run outside the artifact lock")
            with accounting_lock:
                active_training += 1
                peak_training = max(peak_training, active_training)
                training_threads.add(threading.get_ident())
            entered_training.wait()
            snapshots, states = self.create_training_outputs(job)
            Path(log).write_text(f"Completed fixture training for {job['run_name']}\n")
            with accounting_lock:
                self.expected_models.update(snapshots)
                self.expected_states.update(states)
            finished_outputs.wait()
            with accounting_lock:
                active_training -= 1
            return 1.0

        def require_owned_lock(function):
            def checked(*args, **kwargs):
                self.assertEqual(
                    artifact_lock.owner, threading.get_ident(),
                    f"{function.__name__} must execute under the shared artifact lock",
                )
                return function(*args, **kwargs)
            return checked

        # Spies enforce the lock boundary but execute the real filesystem operations.
        # Capacity admission has its own tests; this fixture does not require 30 GiB free.
        with (
            redirect_stdout(io.StringIO()),
            patch.object(MODULE, "space_check", return_value={"fixture_capacity_preflight": True}),
            patch.object(MODULE, "run_training", side_effect=train) as training,
            patch.object(MODULE, "audit_run", side_effect=require_owned_lock(MODULE.audit_run)) as audit,
            patch.object(
                MODULE, "lock_json_contract", side_effect=require_owned_lock(MODULE.lock_json_contract)
            ) as contracts,
            patch.object(MODULE, "compact_run", side_effect=require_owned_lock(MODULE.compact_run)) as compact,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(
                    MODULE.execute_job, job, index, 2,
                    paths=self.paths, common=self.common, env={}, hardware={"fixture": "CPU only"},
                    artifact_lock=artifact_lock, stop_event=stop_event,
                )
                for index, job in enumerate(self.jobs, 1)
            ]
            results = [future.result(timeout=30) for future in futures]

        self.assertEqual(peak_training, 2, "Both training callbacks must overlap before finalization")
        self.assertEqual(active_training, 0)
        self.assertEqual(len(training_threads), 2)
        self.assertEqual(artifact_lock.attempts, 2)
        self.assertEqual(set(artifact_lock.entries), training_threads)
        self.assertEqual((training.call_count, audit.call_count, contracts.call_count, compact.call_count), (2,) * 4)
        self.assertFalse(stop_event.is_set())

        storage = self.paths["output"] / "storage_base"
        identity = MODULE.read_json(storage / "identity.json")
        base_hash = MODULE.sha256_file(storage / "model.safetensors")
        self.assertEqual(identity["model_sha256"], base_hash)
        self.assertEqual(identity["training_base_sha256"], self.common["base_sha256"])
        self.assertEqual(list(self.paths["output"].rglob("storage_base/model.safetensors")), [storage / "model.safetensors"])
        self.assertFalse((self.paths["output"] / "storage_base.tmp").exists())
        first_model = self.paths["output"] / identity["source_run"] / "checkpoints/009000/pretrained_model"
        self.assertEqual((storage / "model.safetensors").read_bytes(), self.expected_models[str(first_model)]["model.safetensors"])
        self.assertEqual((self.paths["base"] / "model.safetensors").read_bytes(), self.original_base)
        self.assertEqual(self.unrelated.read_bytes(), self.unrelated_bytes)
        self.assertEqual(len(list((self.paths["output"] / "records").glob("*.json"))), 2)

        restored_hashes = set()
        for job in results:
            output = Path(job["output_dir"])
            record = MODULE.read_json(self.paths["output"] / "records" / f"{job['run_name']}.json")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record, MODULE.read_json(output / "revision_manifest.json"))
            self.assertEqual(len(record["checkpoints"]), 4)
            self.assertTrue(record["loss_summary"]["stable_target_met"])
            self.assertEqual(record["audit"]["parameters"]["trainable_total"], 49152)
            self.assertEqual(record["audit"]["peak_allocated_gpu_gib"], float(job["task_id"]))
            MODULE.verify_completed(record, storage, base_hash)
            contract = MODULE.read_json(
                self.paths["output"] / "task_protocols" / f"task{job['task_id']}-actual.json"
            )
            self.assertEqual(contract, {
                "split": record["audit"]["split"], "normalization": record["audit"]["normalization"],
            })
            self.assertTrue((self.paths["output"] / "logs" / f"{job['run_name']}.log").is_file())
            for checkpoint in record["checkpoints"]:
                model = Path(checkpoint["source_checkpoint"])
                bundle = Path(checkpoint["bundle"])
                self.assertEqual(checkpoint["storage_manifest"]["base_model_sha256"], base_hash)
                self.assertTrue(checkpoint["storage_manifest"]["reconstruction_verified"])
                self.assertFalse((model / "model.safetensors").exists())
                marker = MODULE.read_json(model / "COMPACT_CHECKPOINT.json")
                self.assertEqual(marker["storage_base"], str(storage))
                self.assertEqual(marker["bundle"], str(bundle))
                state = model.parent / "training_state"
                self.assertEqual(
                    {path.name: path.read_bytes() for path in state.iterdir()}, self.expected_states[str(model)]
                )
                for name, content in self.expected_models[str(model)].items():
                    if name != "model.safetensors":
                        self.assertEqual((model / name).read_bytes(), content)
                restored = self.work / "restored" / job["run_name"] / bundle.name
                CHECKPOINT.materialize_checkpoint(bundle, storage, restored)
                self.assertEqual({
                    path.relative_to(restored).as_posix(): path.read_bytes()
                    for path in restored.rglob("*") if path.is_file()
                }, self.expected_models[str(model)])
                restored_hashes.add(MODULE.sha256_file(restored / "model.safetensors"))
        self.assertEqual(len(restored_hashes), 8, "Every task and milestone must retain its own weights")


if __name__ == "__main__":
    unittest.main(verbosity=2)
