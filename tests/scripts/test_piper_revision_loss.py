"""Dependency-free checks: uv run --no-project python tests/scripts/test_piper_revision_loss.py."""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "examples/piper/revision_loss.py"
SPEC = importlib.util.spec_from_file_location("piper_revision_loss", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WRITER_SPEC = importlib.util.spec_from_file_location(
    "training_diagnostics", ROOT / "src/lerobot/utils/training_diagnostics.py"
)
WRITER = importlib.util.module_from_spec(WRITER_SPEC)
WRITER_SPEC.loader.exec_module(WRITER)


def record(step, loss=1e-5, updates=50, **extra):
    return {"step": step, "window_updates": updates, "train": {"loss": loss}, **extra}


class LossSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "training_diagnostics.jsonl"

    def write(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def summarize(self, **kwargs):
        result = MODULE.summarize_loss(self.directory, **kwargs)
        # Summaries are saved by the launcher as strict JSON, including malformed/NaN inputs.
        json.dumps(result, allow_nan=False)
        return result

    def test_actual_writer_format_and_weighted_means(self):
        writer = WRITER.TrainingDiagnostics(self.path, {"training_stage": "flow"})
        for size, end, loss in ((3, 3, 2.0), (2, 5, 8.0), (1, 6, 0.0)):
            for _ in range(size):
                writer.update({"loss": loss})
            writer.write(end, {"loss": loss})
        summary = self.summarize(target=4.5, window_updates=5)
        self.assertAlmostEqual(summary["trailing_loss"], (3 * 2 + 2 * 8) / 6)
        self.assertEqual(summary["actual_window_updates"], 6)
        self.assertEqual(summary["latest_window_updates"], 1)
        self.assertEqual(summary["first_target_step"], 5)
        self.assertAlmostEqual(summary["first_target_trailing_loss"], 4.4)
        self.assertEqual(summary["first_target_actual_window_updates"], 5)
        self.assertTrue(summary["stable_target_met"])

    def test_shortest_whole_window_suffix_does_not_guess_partial_mean(self):
        self.write([record(300, 9e-4, 300), record(600, 1e-5, 300), record(650, 0.0, 50)])
        summary = self.summarize()
        self.assertEqual(summary["actual_window_updates"], 650)
        self.assertEqual(summary["trailing_start_step"], 1)
        self.assertEqual(summary["trailing_end_step"], 650)
        self.assertEqual(summary["trailing_log_windows"], 3)
        self.assertAlmostEqual(summary["trailing_loss"], (300 * 9e-4 + 300 * 1e-5) / 650)
        self.assertFalse(summary["stable_target_met"])
        # Once the oldest whole window can be removed, it is removed entirely.
        with self.path.open("a") as stream:
            stream.write(json.dumps(record(800, 0.0, 150)) + "\n")
        summary = self.summarize()
        self.assertEqual(summary["actual_window_updates"], 500)
        self.assertEqual(summary["trailing_start_step"], 301)
        self.assertAlmostEqual(summary["trailing_loss"], 300 * 1e-5 / 500)
        self.assertEqual(summary["first_target_step"], 800)

    def test_final_partial_logging_interval_reports_525_actual_updates(self):
        self.write([record(step) for step in range(50, 551, 50)] + [record(575, updates=25)])
        summary = self.summarize()
        self.assertEqual(summary["actual_window_updates"], 525)
        self.assertEqual(summary["trailing_start_step"], 51)
        self.assertEqual(summary["trailing_end_step"], 575)
        self.assertEqual(summary["first_target_step"], 500)
        self.assertTrue(summary["stable_target_met"])

    def test_low_loss_without_500_updates_does_not_meet_target(self):
        self.write([record(step) for step in range(50, 451, 50)])
        summary = self.summarize()
        self.assertAlmostEqual(summary["trailing_loss"], 1e-5)
        self.assertEqual(summary["actual_window_updates"], 450)
        self.assertFalse(summary["coverage_complete"])
        self.assertTrue(summary["window_all_finite"])
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_one_low_log_window_does_not_hide_previous_high_loss(self):
        self.write([record(step, 1e-2) for step in range(50, 451, 50)] + [record(500, 1e-6)])
        summary = self.summarize()
        self.assertEqual(summary["last_logged_step"], 500)
        self.assertEqual(summary["latest_logged_loss"], 1e-6)
        self.assertGreater(summary["trailing_loss"], summary["target"])
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_first_achievement_is_sampled_at_log_boundary_and_survives_later_spike(self):
        rows = [record(50, 2e-3)] + [record(step) for step in range(100, 551, 50)]
        self.write(rows)
        summary = self.summarize()
        self.assertEqual(summary["first_target_step"], 550)
        self.assertTrue(summary["stable_target_met"])
        self.write(rows + [record(600, 1e-2)])
        summary = self.summarize()
        self.assertEqual(summary["first_target_step"], 550)
        self.assertFalse(summary["stable_target_met"])

    def test_missing_main_log_does_not_search_preparation_or_other_runs(self):
        preparation = self.directory / "preparation"
        preparation.mkdir()
        (preparation / self.path.name).write_text(json.dumps(record(500, updates=500)) + "\n")
        summary = self.summarize()
        self.assertEqual(summary["status"], "missing_log")
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["last_logged_step"])
        self.write([record(450, updates=450)])
        self.assertEqual(self.summarize()["actual_window_updates"], 450)
        self.assertFalse(self.summarize()["stable_target_met"])

    def test_empty_log_returns_safe_empty_summary(self):
        self.path.write_text("\n  \n")
        summary = self.summarize()
        self.assertEqual(summary["status"], "empty_log")
        self.assertEqual(summary["actual_window_updates"], 0)
        self.assertIsNone(summary["trailing_loss"])
        self.assertFalse(summary["stable_target_met"])

    def test_partial_final_json_or_utf8_retains_last_complete_window(self):
        for fragment in (b'{"step": 550, "train":', b'{"note": "\xe4\xb8'):
            with self.subTest(fragment=fragment):
                self.write([record(500, updates=500)])
                with self.path.open("ab") as stream:
                    stream.write(fragment)
                summary = self.summarize()
                self.assertEqual(summary["last_logged_step"], 500)
                self.assertTrue(summary["ignored_incomplete_last_line"])
                self.assertTrue(summary["stable_target_met"])
                self.assertEqual(summary["malformed_records"], 0)
                self.assertTrue(summary["warnings"])

    def test_complete_final_json_without_newline_is_accepted(self):
        self.path.write_text(json.dumps(record(500, updates=500)))
        summary = self.summarize()
        self.assertTrue(summary["stable_target_met"])
        self.assertFalse(summary["ignored_incomplete_last_line"])

    def test_malformed_complete_line_breaks_continuity(self):
        self.path.write_text(
            json.dumps(record(250, updates=250)) + "\nnot json\n"
            + json.dumps(record(500, updates=250)) + "\n"
        )
        summary = self.summarize()
        self.assertEqual(summary["malformed_records"], 1)
        self.assertEqual(summary["actual_window_updates"], 250)
        self.assertEqual(summary["current_segment_start_step"], 251)
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_malformed_final_complete_record_invalidates_current_coverage(self):
        self.write([record(500, updates=500), {"step": 550, "window_updates": 0}])
        summary = self.summarize()
        self.assertEqual(summary["malformed_records"], 1)
        self.assertEqual(summary["actual_window_updates"], 0)
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_nonfinite_missing_negative_and_non_numeric_losses_never_meet_target(self):
        for loss in (None, math.nan, math.inf, -math.inf, -1e-5, True, "0.00001"):
            with self.subTest(loss=loss):
                self.write([record(500, loss, 500)])
                summary = self.summarize()
                self.assertEqual(summary["last_logged_step"], 500)
                self.assertTrue(summary["coverage_complete"])
                self.assertFalse(summary["window_all_finite"])
                self.assertIsNone(summary["latest_logged_loss"])
                self.assertIsNone(summary["trailing_loss"])
                self.assertFalse(summary["stable_target_met"])
                self.assertIsNone(summary["first_target_step"])
                self.assertEqual(summary["invalid_loss_windows"], 1)

    def test_policy_nonfinite_counter_invalidates_otherwise_finite_window(self):
        self.write([record(500, updates=500, diagnostics_rank0={"loss": {"mean": 1e-5, "nonfinite": 1}})])
        summary = self.summarize()
        self.assertEqual(summary["latest_logged_loss"], 1e-5)
        self.assertIsNone(summary["trailing_loss"])
        self.assertFalse(summary["stable_target_met"])

    def test_nonfinite_window_can_only_age_out_after_sufficient_finite_coverage(self):
        rows = [record(50, None)] + [record(step) for step in range(100, 501, 50)]
        self.write(rows)
        self.assertFalse(self.summarize()["stable_target_met"])
        self.write(rows + [record(550)])
        summary = self.summarize()
        self.assertTrue(summary["stable_target_met"])
        self.assertEqual(summary["actual_window_updates"], 500)
        self.assertEqual(summary["first_target_step"], 550)

    def test_adjacent_duplicates_never_inflate_update_count(self):
        self.write([record(50)] * 10)
        summary = self.summarize()
        self.assertEqual(summary["accepted_log_windows"], 1)
        self.assertEqual(summary["duplicate_records"], 9)
        self.assertEqual(summary["actual_window_updates"], 50)
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_conflicting_same_step_record_restarts_coverage(self):
        self.write([record(500, updates=500), record(500, 2e-5)])
        summary = self.summarize()
        self.assertEqual(summary["restart_or_overlap_records"], 1)
        self.assertEqual(summary["actual_window_updates"], 50)
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_overlapping_increasing_steps_are_not_counted_twice(self):
        self.write([record(step) for step in range(50, 451, 50)] + [record(500, updates=100)])
        summary = self.summarize()
        self.assertEqual(summary["restart_or_overlap_records"], 1)
        self.assertEqual(summary["actual_window_updates"], 100)
        self.assertEqual(summary["current_segment_start_step"], 401)
        self.assertFalse(summary["stable_target_met"])

    def test_step_rewind_does_not_inherit_previous_target_achievement(self):
        self.write([record(500, updates=500)] + [record(step) for step in range(50, 451, 50)])
        summary = self.summarize()
        self.assertEqual(summary["segments"], 2)
        self.assertEqual(summary["last_logged_step"], 450)
        self.assertEqual(summary["actual_window_updates"], 450)
        self.assertFalse(summary["stable_target_met"])
        self.assertIsNone(summary["first_target_step"])

    def test_gaps_reset_but_contiguous_resume_retains_coverage(self):
        self.write([record(250, updates=250), record(750, updates=250)])
        summary = self.summarize()
        self.assertEqual(summary["gap_records"], 1)
        self.assertEqual(summary["actual_window_updates"], 250)
        self.assertFalse(summary["stable_target_met"])
        self.write([record(250, updates=250), record(500, updates=250)])
        summary = self.summarize()
        self.assertEqual(summary["segments"], 1)
        self.assertEqual(summary["first_target_step"], 500)
        self.assertTrue(summary["stable_target_met"])

    def test_absolute_resume_step_is_not_assumed_to_be_observed_update_count(self):
        self.write([record(10000)])
        summary = self.summarize()
        self.assertEqual(summary["actual_window_updates"], 50)
        self.assertEqual(summary["trailing_start_step"], 9951)
        self.assertFalse(summary["stable_target_met"])

    def test_stage_change_cannot_combine_preparation_with_adaptation(self):
        self.write([
            record(250, updates=250, training_stage="stage1"),
            record(500, updates=250, training_stage="flow"),
        ])
        summary = self.summarize()
        self.assertEqual(summary["stage_changes"], 1)
        self.assertEqual(summary["actual_window_updates"], 250)
        self.assertFalse(summary["stable_target_met"])

    def test_trainer_loss_takes_precedence_over_rank_local_policy_loss(self):
        self.write([record(500, 1e-2, 500, diagnostics_rank0={"loss": {"mean": 1e-6, "nonfinite": 0}})])
        summary = self.summarize()
        self.assertEqual(summary["trailing_loss"], 1e-2)
        self.assertFalse(summary["stable_target_met"])

    def test_large_finite_loss_does_not_overflow_weighted_sum(self):
        self.write([record(500, 1e308, 500)])
        summary = self.summarize()
        self.assertEqual(summary["trailing_loss"], 1e308)
        self.assertTrue(summary["window_all_finite"])
        self.assertFalse(summary["stable_target_met"])

    def test_invalid_configuration_is_rejected_even_without_log(self):
        for target in (0, -1, math.nan, math.inf, -math.inf, True, "1e-4", 10**400):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.summarize(target=target)
        for size in (0, -1, 500.0, True, None, "500"):
            with self.subTest(window_updates=size), self.assertRaises(ValueError):
                self.summarize(window_updates=size)

    def test_invalid_step_or_update_count_cannot_fabricate_coverage(self):
        for step, updates in ((50, 500), (500, True), (500.0, 500), (500, -1), (True, 1)):
            with self.subTest(step=step, updates=updates):
                self.write([record(step, updates=updates)])
                summary = self.summarize()
                self.assertEqual(summary["status"], "no_valid_windows")
                self.assertEqual(summary["malformed_records"], 1)
                self.assertFalse(summary["stable_target_met"])

    def test_cli_returns_summary_and_rejects_nonfinite_target(self):
        self.write([record(500, updates=500)])
        process = subprocess.run(
            [sys.executable, str(FILE), str(self.directory), "--target", "1e-4", "--window-updates", "500"],
            capture_output=True, text=True, check=True,
        )
        self.assertTrue(json.loads(process.stdout)["stable_target_met"])
        process = subprocess.run(
            [sys.executable, str(FILE), str(self.directory), "--target", "nan"],
            capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("finite positive", process.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
