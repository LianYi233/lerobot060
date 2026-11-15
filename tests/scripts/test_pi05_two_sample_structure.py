"""Pure-CPU contract tests for the two-sample PrimingVLA mechanism experiment."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from two_sample_sap_structure import (  # noqa: E402
    interpolate_visible,
    masked_mse,
    parse_episode_offsets,
    quantile_normalize,
    select_chunk_starts,
)


class TwoSampleStructureTest(unittest.TestCase):
    def test_two_distinct_episode_offset_selectors_required(self):
        self.assertEqual(parse_episode_offsets("0:10, 1:11"), ((0, 10), (1, 11)))
        for spec in ("0:10", "0:10,0:10", "0:-1,1:0", "foo:1,1:0", "0:1,1:2,2:3"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_episode_offsets(spec)

    def test_selects_full_non_overlapping_chunks(self):
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 120},
            {"dataset_from_index": 120, "dataset_to_index": 240},
        ]
        self.assertEqual(select_chunk_starts(episodes, ((0, 10), (1, 10)), 50), [10, 130])
        self.assertEqual(select_chunk_starts(episodes, ((0, 0), (0, 50)), 50), [0, 50])
        for selections in (((0, 10), (0, 20)), ((1, 80), (0, 0)), ((9, 0), (0, 0))):
            with self.subTest(selections=selections), self.assertRaises(ValueError):
                select_chunk_starts(episodes, selections, 50)

    def test_pi05_quantile_normalization(self):
        raw = np.array([[[0.0, 3.0], [5.0, 3.0], [10.0, 3.0]]], dtype=np.float32)
        q01 = np.array([0.0, 3.0], dtype=np.float32)
        q99 = np.array([10.0, 3.0], dtype=np.float32)
        normalized = quantile_normalize(raw, q01, q99)
        np.testing.assert_allclose(normalized[0, :, 0], [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(normalized[0, :, 1], [-1.0] * 3)
        self.assertEqual(normalized.dtype, np.float32)

    def test_interpolation_uses_only_visible_values(self):
        actions = np.stack([np.arange(5), np.arange(5) * -2], axis=-1)[None].astype(np.float32)
        mask = np.array([[False, True, True, True, False]])
        reconstructed = interpolate_visible(actions, mask)
        np.testing.assert_allclose(reconstructed, actions)
        self.assertAlmostEqual(masked_mse(reconstructed, actions, mask), 0.0)
        altered = reconstructed.copy()
        altered[mask] += 2.0
        self.assertAlmostEqual(masked_mse(altered, actions, mask), 4.0)

    def test_interpolation_holds_endpoints_and_handles_no_context(self):
        actions = np.array([[[0.0], [2.0], [4.0], [6.0]]], dtype=np.float32)
        sparse_mask = np.array([[True, False, True, True]])
        np.testing.assert_allclose(interpolate_visible(actions, sparse_mask)[0, :, 0], [2, 2, 2, 2])
        fully_masked = np.ones((1, 4), dtype=bool)
        np.testing.assert_allclose(interpolate_visible(actions, fully_masked)[0, :, 0], [3, 3, 3, 3])

    def test_cli_help_does_not_need_gpu_or_torch_import(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples/analysis/two_sample_sap_structure.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("train", proc.stdout)
        self.assertIn("analyze", proc.stdout)


if __name__ == "__main__":
    unittest.main()
