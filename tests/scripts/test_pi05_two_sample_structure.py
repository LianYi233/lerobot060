"""Pure-CPU contract tests for the two-sample PrimingVLA mechanism experiment."""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from two_sample_sap_structure import (  # noqa: E402
    _load_real_action_chunks,
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

    def test_selects_from_sparse_episode_metadata(self):
        episodes = {
            8: {"dataset_from_index": 1000, "dataset_to_index": 1100},
            9: {"dataset_from_index": 1100, "dataset_to_index": 1200},
        }
        self.assertEqual(select_chunk_starts(episodes, ((8, 1), (9, 2)), 50), [1001, 1102])
        with self.assertRaisesRegex(ValueError, "not found"):
            select_chunk_starts(episodes, ((7, 1), (8, 2)), 50)

    def test_direct_parquet_reader_touches_only_two_action_windows(self):
        # A tiny PyArrow API stand-in exercises the no-HF-cache code path even
        # in lightweight CI environments without torch or pyarrow installed.
        class Expr:
            def __init__(self, fn):
                self.fn = fn

            def __and__(self, other):
                return Expr(lambda row: self.fn(row) and other.fn(row))

        class Field:
            def __init__(self, key):
                self.key = key

            def __eq__(self, other):
                return Expr(lambda row: row[self.key] == other)

            def __ge__(self, other):
                return Expr(lambda row: row[self.key] >= other)

            def __lt__(self, other):
                return Expr(lambda row: row[self.key] < other)

            def isin(self, values):
                return Expr(lambda row: row[self.key] in values)

        class Table:
            def __init__(self, rows):
                self.rows = rows

            def to_pylist(self):
                return self.rows

            def to_pydict(self):
                return {key: [row[key] for row in self.rows] for key in self.rows[0]}

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            def to_table(self, *, columns, filter):
                selected = [{k: row[k] for k in columns} for row in self.rows if filter.fn(row)]
                return Table(selected)

        episodes = [
            {"episode_index": ep, "dataset_from_index": ep * 10, "dataset_to_index": (ep + 1) * 10,
             "data/chunk_index": 0, "data/file_index": 0}
            for ep in range(2)
        ]
        actions = [
            {"episode_index": i // 10, "index": i, "action": [float(i), float(i + 1)]}
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "meta/episodes").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000/file-000.parquet").touch()
            (root / "meta/info.json").write_text(
                json.dumps({"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"})
            )
            (root / "meta/stats.json").write_text(json.dumps({"action": {"q01": [0, 0], "q99": [20, 20]}}))
            fake_dataset = types.ModuleType("pyarrow.dataset")
            fake_dataset.field = Field
            fake_dataset.dataset = lambda path, format: FakeDataset(
                episodes if Path(path).name == "episodes" else actions
            )
            fake_arrow = types.ModuleType("pyarrow")
            fake_arrow.__path__ = []
            fake_arrow.dataset = fake_dataset
            with patch.dict(sys.modules, {"pyarrow": fake_arrow, "pyarrow.dataset": fake_dataset}):
                normalized, selectors, starts = _load_real_action_chunks(
                    SimpleNamespace(dataset_root=root, chunks="0:1,1:2"),
                    SimpleNamespace(chunk_size=3, max_action_dim=32),
                )
            self.assertEqual(selectors, ((0, 1), (1, 2)))
            self.assertEqual(starts, [1, 12])
            self.assertEqual(normalized.shape, (2, 3, 2))
            np.testing.assert_allclose(normalized[0, :, 0], [-0.9, -0.8, -0.7], atol=1e-6)
            np.testing.assert_allclose(normalized[1, :, 0], [0.2, 0.3, 0.4], atol=1e-6)

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
