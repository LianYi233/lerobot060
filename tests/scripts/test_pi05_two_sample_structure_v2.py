"""CPU-only tests for the complete-ODE two-sample structure metrics."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from two_sample_sap_structure import parse_episode_offsets_many, select_chunk_starts  # noqa: E402
from two_sample_sap_structure_v2 import (  # noqa: E402
    _aggregate,
    action_transitions,
    evaluate_transition_recovery,
    permute_visible_context,
    reconstruction_metrics,
    second_order_levy_area,
    select_eval_mask,
)


class StructureEvaluationV2Test(unittest.TestCase):
    def test_generic_episode_selectors_and_overlap_guard(self):
        self.assertEqual(parse_episode_offsets_many("2:10,3:20,4:30"), ((2, 10), (3, 20), (4, 30)))
        for value in ("", "0:1,0:1", "1:2,bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_episode_offsets_many(value)
        records = {3: {"dataset_from_index": 100, "dataset_to_index": 200}}
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_chunk_starts(records, ((3, 0), (3, 10), (3, 50)), 50)

    def test_mask_schemes_are_fixed_size_and_event_agnostic(self):
        rng = np.random.default_rng(12)
        for mode in ("random", "long_gap"):
            mask = select_eval_mask(3, 50, 40, mode, rng)
            np.testing.assert_array_equal(mask.sum(axis=1), [40, 40, 40])
        np.testing.assert_array_equal(
            select_eval_mask(1, 50, 40, "long_gap", np.random.default_rng(1))[0, 5:45],
            np.ones(40, dtype=bool),
        )
        with self.assertRaises(ValueError):
            select_eval_mask(1, 50, 49, "random", rng)

    def test_visible_context_permutation_preserves_action_multiset(self):
        values = np.arange(30, dtype=np.float32).reshape(1, 10, 3)
        mask = select_eval_mask(1, 10, 6, "long_gap", np.random.default_rng(0))
        observed = permute_visible_context(values, mask, np.random.default_rng(20))
        np.testing.assert_array_equal(observed[mask], values[mask])
        np.testing.assert_array_equal(
            np.sort(observed[~mask], axis=0), np.sort(values[~mask], axis=0)
        )

    def test_levy_area_distinguishes_motion_before_grip(self):
        move_then_grip = np.array([[0., 0.], [1., 0.], [1., 1.]])
        grip_then_move = np.array([[0., 0.], [0., 1.], [1., 1.]])
        first = second_order_levy_area(move_then_grip)
        second = second_order_levy_area(grip_then_move)
        self.assertAlmostEqual(first[0, 1], 0.5)
        self.assertAlmostEqual(second[0, 1], -0.5)
        self.assertAlmostEqual(first[0, 1], -first[1, 0])

    def test_time_augmented_signature_detects_event_timing(self):
        early = np.array([[0.], [1.], [1.], [1.]])
        late = np.array([[0.], [0.], [1.], [1.]])
        self.assertAlmostEqual(second_order_levy_area(early, include_time=False)[0, 0], 0.)
        self.assertNotAlmostEqual(second_order_levy_area(early)[0, 1],
                                  second_order_levy_area(late)[0, 1])

    def test_transition_recovery_skips_visible_events_and_penalizes_misses(self):
        truth = np.array([[-1., 0.], [-1., 0.], [1., 0.], [1., 0.], [1., 0.]])
        mask = np.array([False, True, True, True, False])
        exact = evaluate_transition_recovery(truth, truth, mask, channel=0, tolerance=0)
        self.assertEqual(exact["events"], 1)
        self.assertEqual(exact["correct_events"], 1)
        self.assertEqual(exact["transition_timing_mae_matched"], 0.)
        never = truth.copy()
        never[:, 0] = -1
        missed = evaluate_transition_recovery(never, truth, mask, channel=0)
        self.assertEqual(missed["events"], 1)
        self.assertEqual(missed["correct_events"], 0)
        self.assertIsNone(missed["transition_timing_mae_matched"])
        visible_event = np.array([False, False, True, True, True])
        self.assertEqual(evaluate_transition_recovery(truth, truth, visible_event, channel=0)["events"], 0)
        self.assertEqual(action_transitions(truth, 0), [(2, 1)])

    def test_reconstruction_metrics_report_both_structural_quantities(self):
        truth = np.array([[0., -1.], [1., -1.], [1., 1.], [1., 1.]], dtype=np.float32)
        mask = np.array([False, True, True, False])
        values = reconstruction_metrics(truth, truth, mask, event_channel=1, tolerance=1)
        self.assertEqual(values["masked_mse"], 0.)
        self.assertEqual(values["signature_area_l2"], 0.)
        self.assertEqual(values["signature_area_relative_error"], 0.)
        self.assertEqual(values["motion_derivative_mse"], 0.)
        self.assertEqual(values["correct_events"], 1)

    def test_aggregation_respects_distinct_chunk_unit(self):
        prototype = {"masked_mse": 1., "permuted_context_mse": 2., "context_order_gain": 1.,
                     "motion_derivative_mse": 1., "signature_area_l2": 1.,
                     "signature_area_relative_error": 1., "events": 1, "correct_events": 1,
                     "timing_error_sum": 1, "false_transitions": 0}
        rows = [dict(prototype, chunk="1:0", masked_mse=1.),
                dict(prototype, chunk="1:0", masked_mse=3.),
                dict(prototype, chunk="2:0", masked_mse=8.)]
        result = _aggregate(rows)
        self.assertEqual(result["chunks"], 2)
        self.assertAlmostEqual(result["masked_mse"], 5.)
        self.assertAlmostEqual(result["transition_recall_at_tolerance"], 1.)

    def test_cli_help_available_without_torch(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples/analysis/two_sample_sap_structure_v2.py"), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--ode-steps", proc.stdout)
        self.assertIn("--eval-chunks", proc.stdout)


if __name__ == "__main__":
    unittest.main()
