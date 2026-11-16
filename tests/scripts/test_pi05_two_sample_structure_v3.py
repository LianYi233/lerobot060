"""Numerical and protocol tests for the theory-driven, evaluation-only SAP v3."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from two_sample_sap_structure_v3 import (  # noqa: E402
    ALL_NAMES,
    MODEL_NAMES,
    _draw_figures,
    _aggregate,
    delay_hankel,
    flow_alignment_scores,
    hankel_structure_scores,
    lag_correlation_scores,
    lagged_covariance,
    parse_analyses,
    temporal_gain_scores,
)


class TwoSampleStructureV3Test(unittest.TestCase):
    def test_analysis_selector_is_explicit(self):
        self.assertEqual(len(parse_analyses("all")), 4)
        self.assertEqual(parse_analyses("flow_field,hankel"), ("flow_field", "hankel"))
        for invalid in ("", "all,hankel", "hankel,hankel", "made_up"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_analyses(invalid)

    def test_temporal_gain_uses_matched_hidden_positions(self):
        target = np.array([[[2., 1.], [4., 2.], [6., 3.], [8., 4.]]])
        mask = np.array([[False, True, True, False]])
        correct = target.copy()
        corrupted = target.copy()
        corrupted[:, 1:3, :] += 2
        scores = temporal_gain_scores(correct, corrupted, target, mask, [0])
        self.assertEqual(scores["masked_mse"], 0.)
        self.assertEqual(scores["permuted_mse"], 4.)
        self.assertEqual(scores["temporal_predictive_gain"], 4.)
        self.assertEqual(scores["motion_temporal_predictive_gain"], 4.)

    def test_hankel_delay_columns_preserve_order(self):
        x = np.arange(12).reshape(6, 2)
        value = delay_hankel(x, 3)
        self.assertEqual(value.shape, (6, 4))
        np.testing.assert_array_equal(value[:, 0], np.array([0, 1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(value[:, -1], np.array([6, 7, 8, 9, 10, 11]))
        with self.assertRaises(ValueError):
            delay_hankel(x, 7)

    def test_hankel_identical_paths_have_zero_distance(self):
        t = np.linspace(0, 2 * np.pi, 30)
        truth = np.stack([np.sin(t), np.cos(2 * t)], axis=1)
        same = hankel_structure_scores(truth, truth, window=5, rank=3)
        self.assertLess(same["hankel_subspace_distance"], 1e-6)
        self.assertAlmostEqual(same["hankel_spectrum_distance"], 0., places=9)
        reordered = hankel_structure_scores(truth[::-1], truth, window=5, rank=3)
        self.assertGreater(reordered["hankel_subspace_distance"], 1e-4)

    def test_hankel_static_reference_is_not_falsely_informative(self):
        target = np.ones((20, 2))
        results = hankel_structure_scores(target * 3, target, window=5)
        self.assertIsNone(results["hankel_subspace_distance"])
        self.assertIsNone(results["hankel_spectrum_distance"])

    def test_lag_correlation_zero_for_exact_reconstruction(self):
        t = np.linspace(-1, 1, 28)
        truth = np.stack([t ** 2, np.sin(5 * t), np.sin(3 * t + .7)], axis=1)
        identical = lag_correlation_scores(truth, truth, max_lag=5)
        self.assertAlmostEqual(identical["lag_correlation_relative_error"], 0.)
        reordered = lag_correlation_scores(truth[::-1], truth, max_lag=5)
        self.assertGreater(reordered["lag_correlation_relative_error"], 0.01)

    def test_lag_cross_joint_covariance_is_order_sensitive(self):
        t = np.arange(12, dtype=float)
        x = np.stack([t ** 2, (t - 4) ** 3], axis=1)
        a = lagged_covariance(x, 2)
        b = lagged_covariance(x[::-1], 2)
        self.assertEqual(a.shape, (3, 2, 2))
        self.assertGreater(float(np.linalg.norm(a - b)), .001)
        with self.assertRaises(ValueError):
            lagged_covariance(x, 11)

    def test_flow_alignment_exact_reversed_and_zero(self):
        target = np.array([[1., 2.], [-2., 3.]])
        exact = flow_alignment_scores(target, target)
        self.assertAlmostEqual(exact["flow_cosine"], 1.)
        self.assertAlmostEqual(exact["flow_relative_mse"], 0.)
        reverse = flow_alignment_scores(-target, target)
        self.assertAlmostEqual(reverse["flow_cosine"], -1.)
        self.assertAlmostEqual(reverse["flow_relative_mse"], 4.)
        self.assertIsNone(flow_alignment_scores(np.zeros_like(target), target)["flow_cosine"])

    def test_aggregate_averages_by_distinct_chunk(self):
        rows = [{"chunk": "a", "m": 2.}, {"chunk": "a", "m": 4.},
                {"chunk": "b", "m": 10.}]
        summary = _aggregate(rows, ("m",))
        self.assertEqual(summary["chunks"], 2)
        self.assertAlmostEqual(summary["m"], 6.5)

    def test_each_analysis_figure_can_be_generated(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        times = (.1, .5, .9)
        summary = {
            "temporal_gain": {
                mode: {name: {"temporal_predictive_gain": .01} for name in ALL_NAMES}
                for mode in ("random", "long_gap")
            },
            "hankel": {"long_gap": {
                name: {"hankel_subspace_distance": .1,
                       "hankel_spectrum_distance": .2}
                for name in ALL_NAMES
            }},
            "flow_field": {
                mode: {name: {"by_t": {str(t): {"flow_cosine": .6} for t in times}}
                       for name in MODEL_NAMES}
                for mode in ("random", "long_gap")
            },
            "lag_correlation": {"long_gap": {
                name: {"lag_correlation_relative_error": .3} for name in ALL_NAMES
            }},
        }
        examples = {
            "hankel_spectra": {name: [4., 2., 1.] for name in (*ALL_NAMES, "ground_truth")},
            "lag_norms": {name: [2., 1., .5] for name in (*ALL_NAMES, "ground_truth")},
        }
        with tempfile.TemporaryDirectory() as directory:
            _draw_figures(Path(directory), parse_analyses("all"), summary, examples, times)
            for name in ("temporal_gain", "hankel", "flow_field", "lag_correlation"):
                path = Path(directory) / f"{name}_v3.png"
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 1000)

    def test_cli_help_does_not_import_model_or_require_gpu(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples/analysis/two_sample_sap_structure_v3.py"), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--analyses", proc.stdout)
        self.assertIn("--eval-chunks", proc.stdout)


if __name__ == "__main__":
    unittest.main()
