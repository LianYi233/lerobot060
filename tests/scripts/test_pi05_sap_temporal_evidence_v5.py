"""CPU-only regression guards for the order-grounded SAP V5 experiments."""

from __future__ import annotations

from argparse import Namespace
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from sap_temporal_v5_utils import (  # noqa: E402
    MASK_MODES,
    ar2_parameter_error,
    conditional_signature_mmd,
    ordered_history,
    paired_interval,
    ridge_forecast,
    signature_features,
    swap_visible_order,
    synthetic_ar2_chunks,
    temporal_mask,
    unordered_history,
)
from sap_temporal_evidence_v5 import _summarize_blocked  # noqa: E402
from plot_sap_temporal_v5 import plot  # noqa: E402


class SapTemporalEvidenceV5Test(unittest.TestCase):
    def test_forecast_hides_all_future_even_if_scoring_only_five(self):
        for mode in MASK_MODES:
            with self.subTest(mode=mode):
                hidden, score = temporal_mask(3, 50, mode, np.random.default_rng(1))
                self.assertEqual(hidden.shape, (3, 50))
                self.assertFalse(np.any(score & ~hidden))
                self.assertTrue((~hidden).any(axis=1).all())
        mask, score = temporal_mask(2, 50, "future5", np.random.default_rng(0))
        self.assertTrue(mask[:, 20:].all())
        self.assertFalse(mask[:, :20].any())
        self.assertTrue(score[:, 20:25].all())
        self.assertFalse(score[:, 25:].any())
        mask, score = temporal_mask(1, 50, "middle20", np.random.default_rng(0))
        self.assertTrue(mask[0, 15:35].all())
        self.assertEqual(int(score.sum()), 20)

    def test_permutation_preserves_visible_whole_action_vectors(self):
        actions = np.arange(2 * 50 * 7, dtype=np.float32).reshape(2, 50, 7)
        hidden, _ = temporal_mask(2, 50, "middle20", np.random.default_rng(2))
        altered = swap_visible_order(actions, hidden, np.random.default_rng(7))
        for index in range(2):
            np.testing.assert_array_equal(altered[index, hidden[index]], actions[index, hidden[index]])
            original = {tuple(row) for row in actions[index, ~hidden[index]]}
            permuted = {tuple(row) for row in altered[index, ~hidden[index]]}
            self.assertEqual(original, permuted)
        self.assertFalse(np.array_equal(altered, actions))

    def test_bag_features_preserve_vectors_and_match_feature_dimension(self):
        chunks = np.arange(2 * 50 * 7, dtype=np.float32).reshape(2, 50, 7)
        ordered = ordered_history(chunks)
        unordered = unordered_history(chunks)
        self.assertEqual(ordered.shape, unordered.shape)
        for index in range(2):
            original = sorted(map(tuple, chunks[index, :20]))
            scrambled = sorted(map(tuple, unordered[index].reshape(20, 7)))
            self.assertEqual(original, scrambled)
        repeated_norms = np.zeros((1, 50, 3), dtype=np.float32)
        repeated_norms[0, 0] = [1, 0, 0]
        repeated_norms[0, 1] = [-1, 0, 0]  # Equal norm, different vectors.
        altered = repeated_norms.copy()
        altered[0, :20] = repeated_norms[0, np.random.default_rng(3).permutation(20)]
        np.testing.assert_array_equal(unordered_history(repeated_norms), unordered_history(altered))

    def test_ridge_preflight_separates_known_ar2_order_from_bag(self):
        train, _ = synthetic_ar2_chunks(180, seed=11)
        test, _ = synthetic_ar2_chunks(50, seed=13, first_episode=100000)
        result = ridge_forecast(train, test, context=20, horizon=5, seed=3)
        self.assertTrue(np.isfinite(result["ordered"]["relative_mse"]))
        self.assertTrue(np.isfinite(result["unordered"]["relative_mse"]))
        # This tests a valid ordering problem, not any guarantee for real LIBERO.
        self.assertLess(result["ordered"]["relative_mse"], result["unordered"]["relative_mse"])

    def test_synthetic_generator_preserves_known_dynamics(self):
        samples, meta = synthetic_ar2_chunks(6, seed=5)
        self.assertEqual(samples.shape, (6, 50, 7))
        self.assertEqual(len({m["episode_index"] for m in meta}), 6)
        errors = []
        for path, row in zip(samples, meta, strict=True):
            truth = row["true_ar2"]
            result = ar2_parameter_error(path, beta1=truth["beta1"],
                                         beta2=truth["beta2"])
            errors.append(result["coefficient_l2"])
            self.assertIsNotNone(result["r_hat"])
            self.assertIsNotNone(result["omega_hat"])
        self.assertTrue(all(value is not None for value in errors))
        self.assertLess(float(np.mean(errors)), .3)

    def test_signature_is_time_order_sensitive_and_equal_paths_have_zero_mmd(self):
        base = np.zeros((6, 50, 2), dtype=np.float64)
        for i in range(6):
            base[i, :, 0] = np.sin(np.linspace(0., 3., 50) + .1 * i)
            base[i, :, 1] = np.cos(np.linspace(0., 3., 50) + .1 * i)
        reversed_path = base[:, ::-1].copy()
        self.assertGreater(float(np.linalg.norm(signature_features(base) -
                                                signature_features(reversed_path))), 0.)
        context = base[:, :15]
        self.assertAlmostEqual(conditional_signature_mmd(base, base, context), 0., places=10)
        self.assertGreater(conditional_signature_mmd(base, reversed_path, context), 0.)

    def test_paired_bootstrap_and_difference_in_differences(self):
        gap = paired_interval(np.array([.2, .3, .4]), np.array([.4, .5, .6]), seed=1)
        self.assertAlmostEqual(gap["mean"], .2)
        self.assertTrue(all(abs(v - .2) < 1e-10 for v in gap["ci95"]))
        rows = []
        for episode in (1, 2, 3):
            for label, ordered, permuted in (("init", .8, .85), ("sap", .6, .8),
                                             ("shuffled", .65, .7)):
                for condition, mse in (("ordered", ordered), ("permuted", permuted)):
                    rows.append({"episode_index": episode, "model": label, "mode": "future10",
                                 "condition": condition, "flow_time": .99, "flow_mse": mse})
        summary = _summarize_blocked(rows, [], modes=("future10",), high_noise=.99, seed=0)
        comparison = summary["future10"]["flow_comparison"]
        self.assertAlmostEqual(comparison["init_minus_sap"]["mean"], .2)
        self.assertAlmostEqual(comparison["shuffled_minus_sap"]["mean"], .05)
        self.assertAlmostEqual(comparison["extra_order_gain_sap_vs_init"]["mean"], .15)

    def test_help_is_available_without_torch(self):
        script = ROOT / "examples" / "analysis" / "sap_temporal_evidence_v5.py"
        for name in ("data-check", "blocked-eval", "synthetic-prepare",
                     "synthetic-score", "probe", "signature-mmd"):
            with self.subTest(command=name):
                check = subprocess.run([sys.executable, str(script), name, "--help"],
                                       capture_output=True, text=True, check=False)
                self.assertEqual(check.returncode, 0, check.stderr)

    def test_appendix_figure_reads_all_stages_without_cherry_picking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_json = {
                "P0-data-check/data_order_preflight_v5.json": {
                    "horizons": {str(h): {"ordered": {"relative_mse": .9},
                                         "unordered": {"relative_mse": 1.1}}
                                 for h in (5, 10, 20)}},
                "P1-blocked/metrics_temporal_v5.json": {
                    "results": {mode: {"models": {name: {
                        "high_noise_flow_mse": .3 + .1 * i,
                        "full_ode_masked_action_mse": .4 + .1 * i,
                    } for i, name in enumerate(("init", "sap", "shuffled"))}}
                                for mode in ("future10", "middle20", "random20")}},
                "P2-synthetic/recovery/ar2_recovery_v5.json": {
                    "results": {name: {"generated": {"mean_coefficient_l2": .1 + .1 * i},
                                       "oracle_real": {"mean_coefficient_l2": .05}}
                                for i, name in enumerate(("init", "sap", "shuffled"))}},
                "P3-probe/future_probe_v5.json": {"results": {
                    name: {str(h): {"R2_relative_to_intercept": -.2 + .1 * i}
                           for h in (1, 5, 10)}
                    for i, name in enumerate(("init", "sap", "shuffled"))}},
                "P4-signature/signature_mmd_v5.json": {
                    "results": {name: {"biased_product_mmd_squared": .3 + .1 * i}
                                for i, name in enumerate(("init", "sap", "shuffled"))}},
            }
            for relative_path, content in source_json.items():
                source = root / relative_path
                source.parent.mkdir(parents=True)
                source.write_text(json.dumps(content), encoding="utf-8")
            plot(Namespace(root=root, output=root / "figures"))
            self.assertTrue((root / "figures" / "sap_temporal_evidence_v5.png").is_file())
            self.assertTrue((root / "figures" / "sap_temporal_evidence_v5.pdf").is_file())
            figure_manifest = json.loads((root / "figures" / "figure_manifest_v5.json").read_text())
            self.assertEqual(set(figure_manifest["inputs"]), {"P0", "P1", "P2", "P3", "P4"})


if __name__ == "__main__":
    unittest.main()
