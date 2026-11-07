"""Validate paired trial accounting without ML packages or robot access."""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/piper/revision_trials.py"
SPEC = importlib.util.spec_from_file_location("revision_trials", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RevisionTrialsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="piper trial test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "trials.jsonl"
        self.weights = self.root / "model.safetensors"
        self.weights.write_bytes(b"fixture only: not real model weights")
        self.manifest = self.root / "revision_manifest.json"
        self.manifest.write_text('{"fixture": true, "cumulative_updates": 4000}', encoding="utf-8")
        self.model = MODULE._model_identity({
            "checkpoint": str(self.weights), "model_manifest": str(self.manifest),
            "cumulative_updates": 4000,
        })
        self.plan = MODULE.make_plan(["priming"], ["T1"], [0], trials_per_task=2)

    def record(self, row=0, outcome="success", plan=None, **kwargs):
        plan = plan or self.plan
        return MODULE.record_trial(plan, self.log, plan["trials"][row]["trial_id"], outcome, 12.5,
                                   model=kwargs.pop("model", self.model), **kwargs)

    def report(self, plan=None):
        plan = plan or self.plan
        return MODULE.summarize(plan, MODULE.read_records(plan, self.log))

    def test_plan_is_paired_and_reproducible(self):
        args = (["priming", "direct_cpc", "flow_warmup"], ["T1", "T3"], [0, 9])
        first = MODULE.make_plan(*args, trials_per_task=3, policy_seed=21, order_seed=87)
        second = MODULE.make_plan(*args, trials_per_task=3, policy_seed=21, order_seed=87)
        changed = MODULE.make_plan(*args, trials_per_task=3, policy_seed=21, order_seed=88)
        self.assertEqual(first, second)
        self.assertNotEqual([row["trial_id"] for row in first["trials"]],
                            [row["trial_id"] for row in changed["trials"]])
        self.assertEqual(len(first["trials"]), 36)
        by_layout = {}
        for row in first["trials"]:
            by_layout.setdefault(row["layout_id"], []).append(row)
        for rows in by_layout.values():
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["policy_seed"] for row in rows}), 1)
            self.assertEqual({row["train_seed"] for row in rows}, {0, 9})
            self.assertEqual({(row["method"], row["train_seed"]) for row in rows},
                             {(method, seed) for method in args[0] for seed in args[2]})
        original_seeds = {row["trial_id"]: row["policy_seed"] for row in first["trials"]}
        self.assertEqual(original_seeds, {row["trial_id"]: row["policy_seed"] for row in changed["trials"]})
        subset = MODULE.make_plan(args[0], ["T3"], args[2], trials_per_task=3, policy_seed=21)
        self.assertTrue(all(row["policy_seed"] == original_seeds[row["trial_id"]] for row in subset["trials"]))

    def test_default_allocation_is_twenty(self):
        plan = MODULE.make_plan(MODULE.DEFAULT_METHODS, list(MODULE.TASKS), [0])
        self.assertEqual(len(plan["trials"]), 4 * 4 * 20)

    def test_plan_file_refuses_replacement_and_tampering(self):
        path = self.root / "plan.json"
        MODULE._write_json(path, self.plan, exclusive=True)
        self.assertEqual(MODULE.load_plan(path), self.plan)
        with self.assertRaises(FileExistsError):
            MODULE._write_json(path, self.plan, exclusive=True)
        altered = json.loads(path.read_text())
        altered["trials"][0]["policy_seed"] += 1
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "校验"):
            MODULE.load_plan(path)

    def test_no_records_are_pending_not_failures_or_successes(self):
        report = self.report()
        self.assertEqual(report["status"]["planned"], 2)
        self.assertEqual(report["status"]["completed"], 0)
        self.assertEqual(report["status"]["pending"], 2)
        self.assertEqual(report["status"]["failures"], 0)
        self.assertIsNone(report["per_task_seed"][0]["wilson95_pct"])
        self.assertIsNone(report["per_method"][0]["pooled_success_rate_pct"])
        self.assertIsNone(report["per_method"][0]["macro_success_rate_pct"])

    def test_first_record_requires_model_identity_and_valid_budget(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.record(model={})
        with self.assertRaisesRegex(ValueError, "累计更新"):
            self.record(model=dict(self.model, cumulative_updates=3000))
        self.assertEqual(self.report()["status"]["completed"], 0)

    def test_model_index_binds_each_trial_and_record_reuses_it(self):
        plan = MODULE.make_plan(["priming"], [1], [0], trials_per_task=2,
                                models=[dict(self.model, method="priming", task_id=1, train_seed=0)])
        path = self.root / "bound_plan.json"
        MODULE._write_json(path, plan, exclusive=True)
        self.assertEqual(MODULE.load_plan(path), plan)
        event = self.record(plan=plan, model={})
        self.assertEqual(event["model"], self.model)

    def test_later_trial_can_reuse_first_record_identity(self):
        self.record()
        event = self.record(row=1, model={})
        self.assertEqual(event["model"], self.model)

    def test_duplicates_and_conflicting_models_are_refused(self):
        first = self.record()
        with self.assertRaisesRegex(ValueError, "重试"):
            self.record()
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.record(row=1, model=dict(self.model, checkpoint_sha256="f" * 64))
        with self.log.open("a") as stream:
            stream.write(json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "重复"):
            self.report()

    def test_hardware_invalid_retry_is_linked_and_not_double_counted(self):
        first = self.record(outcome="invalid_hardware", reason="相机采集服务崩溃，未收到策略动作")
        report = self.report()["status"]
        self.assertEqual((report["completed"], report["pending"], report["retry_pending"]), (0, 2, 1))
        with self.assertRaisesRegex(ValueError, "重试"):
            self.record()
        with self.assertRaisesRegex(ValueError, "重试"):
            self.record(retry_of="unrelated.a1")
        second = self.record(retry_of=first["attempt_id"])
        self.assertEqual(second["attempt_id"], first["trial_id"] + ".a2")
        self.assertEqual(second["retry_of"], first["attempt_id"])
        report = self.report()["status"]
        self.assertEqual((report["attempts"], report["completed"], report["successes"]), (2, 1, 1))
        self.assertEqual(report["invalid_hardware_attempts"], 1)
        self.assertEqual(report["retry_pending"], 0)
        self.assertEqual(report["not_started"], 1)

    def test_protection_stop_is_a_failure_and_cannot_be_retried(self):
        first = self.record(outcome="policy_stop", reason="策略动作触发关节限制保护")
        self.assertFalse(first["success"])
        self.assertEqual(first["failure_category"], "policy_safety_stop")
        with self.assertRaisesRegex(ValueError, "重试"):
            self.record(retry_of=first["attempt_id"])
        report = self.report()["status"]
        self.assertEqual((report["successes"], report["failures"], report["policy_stops"]), (0, 1, 1))

    def test_outcomes_durations_and_trial_ids_are_validated(self):
        for duration in (-1, math.nan, math.inf, 0, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                MODULE.record_trial(self.plan, self.log, self.plan["trials"][0]["trial_id"],
                                    "success", duration, model=self.model)
        for kwargs in ({"outcome": "failure"}, {"outcome": "invalid_hardware"},
                       {"outcome": "policy_stop"}, {"failure_category": "timeout"},
                       {"outcome": "failure", "failure_category": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.record(**kwargs)
        with self.assertRaisesRegex(ValueError, "不在计划"):
            MODULE.record_trial(self.plan, self.log, "unknown", "success", 1, model=self.model)
        self.assertEqual(self.report()["status"]["completed"], 0)

    def test_partial_results_have_observed_rates_but_no_completed_macro(self):
        self.record()
        report = self.report()
        row = report["per_task_seed"][0]
        self.assertEqual((row["successes"], row["completed"], row["planned"]), (1, 1, 2))
        self.assertEqual(row["pooled_success_rate_pct"], 100)
        self.assertIsNone(report["per_method"][0]["macro_success_rate_pct"])
        self.record(row=1, outcome="failure", failure_category="timeout")
        report = self.report()
        self.assertEqual(report["per_method"][0]["macro_success_rate_pct"], 50)
        self.assertIsNone(report["per_method"][0]["training_seed_std_pct"])
        interval = report["per_task_seed"][0]["wilson95_pct"]
        self.assertAlmostEqual(interval[0], 9.453, places=2)
        self.assertAlmostEqual(interval[1], 90.547, places=2)

    def test_training_seed_statistics_wait_for_all_planned_tasks_and_seeds(self):
        plan = MODULE.make_plan(["priming"], ["T1", "T2"], [0, 1], trials_per_task=2)
        for index, row in enumerate(plan["trials"]):
            failure = row["train_seed"] == 0 and row["task_id"] == "T2"
            self.record(row=index, plan=plan, outcome="failure" if failure else "success",
                        failure_category="timeout" if failure else None)
            if index < len(plan["trials"]) - 1:
                self.assertIsNone(self.report(plan)["per_method"][0]["training_seed_mean_pct"])
        report = self.report(plan)
        result = report["per_method"][0]
        self.assertEqual(result["macro_success_rate_pct"], 75)
        self.assertEqual(result["pooled_success_rate_pct"], 75)
        self.assertEqual(result["training_seed_mean_pct"], 75)
        self.assertAlmostEqual(result["training_seed_std_pct"], math.sqrt(1250))
        self.assertEqual(result["completed_train_seeds"], [0, 1])

    def test_tampered_success_and_truncated_jsonl_fail_closed(self):
        event = self.record()
        event["success"] = False
        self.log.write_text(json.dumps(event) + "\n")
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.report()
        self.log.write_text(json.dumps(event))
        with self.assertRaisesRegex(ValueError, "未完整"):
            self.report()

    def test_nonrecord_json_rows_are_refused(self):
        self.log.write_text("false\n")
        with self.assertRaisesRegex(ValueError, "记录对象"):
            self.report()

    def test_concurrent_duplicate_appends_only_accept_one_record(self):
        path = self.root / "concurrent_plan.json"
        MODULE._write_json(path, self.plan, exclusive=True)
        command = [sys.executable, str(SCRIPT), "record", "--plan", str(path), "--log", str(self.log),
                   "--trial-id", self.plan["trials"][0]["trial_id"], "--outcome", "success",
                   "--duration-seconds", "10", "--checkpoint", str(self.weights),
                   "--model-manifest", str(self.manifest), "--cumulative-updates", "4000"]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        for process in processes:
            process.communicate(timeout=15)
        self.assertEqual(sorted(process.returncode for process in processes), [0, 2])
        records = MODULE.read_records(self.plan, self.log)
        self.assertEqual(len(records), 1)
        self.assertEqual(self.report()["status"]["completed"], 1)

    def test_duplicate_plan_groups_and_mismatched_model_budget_are_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.make_plan(["priming", "priming"], ["T1"], [0])
        with self.assertRaises(ValueError):
            MODULE.make_plan(["priming"], ["T1"], [0, 0])
        with self.assertRaisesRegex(ValueError, "累计更新"):
            MODULE.make_plan(["priming"], ["T1"], [0], models=[
                dict(self.model, cumulative_updates=3000, method="priming", task_id="T1", train_seed=0)])

    def test_cli_plan_record_summary_roundtrip(self):
        path = self.root / "cli_plan.json"
        result_path = self.root / "summary.json"

        def run(*args):
            return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)

        result = run("plan", "--output", path, "--methods", "priming", "--tasks", "T1", "--trials-per-task", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = MODULE.load_plan(path)
        result = run("record", "--plan", path, "--log", self.log, "--trial-id", plan["trials"][0]["trial_id"],
                     "--outcome", "success", "--duration-seconds", 13.2,
                     "--checkpoint", self.weights, "--model-manifest", self.manifest, "--cumulative-updates", 4000)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run("summary", "--plan", path, "--log", self.log, "--output", result_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result_path.read_text())
        self.assertEqual(report["status"]["successes"], 1)
        self.assertIn("已完成 1", result.stdout)
        result = run("summary", "--plan", path, "--log", self.log, "--output", self.log)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(MODULE.read_records(plan, self.log)), 1)


if __name__ == "__main__":
    unittest.main()
