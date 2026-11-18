"""CPU-only safeguards for independent-episode SAP data-scale experiments."""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "analysis"))

from two_sample_sap_structure_v4 import (  # noqa: E402
    _choose_candidates,
    _read_action_windows,
    action_strata,
    aggregate_by_episode,
    balanced_order,
    episode_split,
    make_mask,
    paired_bootstrap_gap,
    parse_flow_times,
    parse_positive_ints,
    main,
)


class SapStructureV4Test(unittest.TestCase):
    def test_four_sample_sizes_parse_correctly(self):
        self.assertEqual(parse_positive_ints("2,32,128,512", minimum=2), (2, 32, 128, 512))
        for spec in ("2,2", "0,2", "2,x", ""):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_positive_ints(spec)
        self.assertEqual(parse_flow_times("0.1,0.5,0.9"), (.1, .5, .9))
        for spec in ("0.0", "1.0", "0.4,abc"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_flow_times(spec)

    def test_split_by_episode_is_disjoint_and_task_aware(self):
        records = [{"episode_index": i, "task": str(i // 10)} for i in range(40)]
        train, test = episode_split(records, .2, 3)
        a = {r["episode_index"] for r in train}
        b = {r["episode_index"] for r in test}
        self.assertFalse(a & b)
        self.assertEqual(len(a | b), 40)
        self.assertEqual(len(test), 8)
        self.assertEqual(episode_split(records, .2, 3), (train, test))

    def test_candidate_episode_selection_keeps_unique_episodes(self):
        records = [
            {"episode_index": i, "episode_length": 77,
             "task": str(i // 20)} for i in range(80)
        ]
        chosen = _choose_candidates(records, 50, seed=11, horizon=50)
        self.assertEqual(len(chosen), 50)
        self.assertEqual(len({r["episode_index"] for r in chosen}), 50)
        self.assertTrue(all(0 <= r["offset"] <= 27 for r in chosen))
        self.assertEqual(chosen, _choose_candidates(records, 50, seed=11, horizon=50))
        self.assertGreaterEqual(len({r["task"] for r in chosen[:4]}), 3)
        with self.assertRaises(ValueError):
            _choose_candidates(records, 81, seed=0, horizon=50)

    def test_reader_scans_all_shards_by_episode_id_not_stale_metadata_path(self):
        # In the actual local LIBERO copy, metadata points episode 1571 at
        # file-064.parquet, but that shard contains only episodes 162-164.
        # The global episode-ID scan still finds all 133 frames of 1571.
        with tempfile.TemporaryDirectory() as temp:
            dataset_root = Path(temp)
            (dataset_root / "data").mkdir()
            action_rows = [
                {"episode_index": 1571, "index": j, "frame_index": j,
                 "action": [float(j), float(2 * j)]} for j in range(100)
            ]
            # A second episode is actually truncated despite its metadata.
            action_rows += [
                {"episode_index": 1572, "index": j, "frame_index": j,
                 "action": [float(j), float(2 * j)]} for j in range(30)
            ]

            class FakeTable:
                def __init__(self, values):
                    self.values = values

                def to_pydict(self):
                    return self.values

            class FakeDataset:
                schema = types.SimpleNamespace(names=["episode_index", "frame_index", "index", "action"])

                def to_table(self, columns, filter):
                    self.requested_columns = columns
                    filter_name, filter_ids = filter
                    assert filter_name == "episode_index"
                    included = [row for row in action_rows
                                if row["episode_index"] in filter_ids]
                    return FakeTable({
                        column: [row[column] for row in included]
                        for column in columns
                    })

            pa = types.ModuleType("pyarrow")
            pa.__path__ = []
            pa_dataset = types.ModuleType("pyarrow.dataset")
            paths_seen = []

            def fake_dataset(path, format):
                paths_seen.append(path)
                self.assertEqual(path, str(dataset_root / "data"))
                self.assertEqual(format, "parquet")
                return FakeDataset()

            pa_dataset.dataset = fake_dataset
            pa_dataset.field = lambda name: types.SimpleNamespace(isin=lambda ids: (name, ids))
            pa.dataset = pa_dataset
            candidates = [
                {"episode_index": 1571, "offset": 10, "episode_length": 100,
                 "dataset_from_index": 258505,
                 "data/chunk_index": 0, "data/file_index": 64},
                {"episode_index": 1572, "offset": 10, "episode_length": 100,
                 "dataset_from_index": 258605,
                 "data/chunk_index": 0, "data/file_index": 64},
            ]
            with patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.dataset": pa_dataset}):
                normalized, accepted, rejected = _read_action_windows(
                    dataset_root, candidates,
                    {"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"},
                    {"action": {"q01": [0, 0], "q99": [100, 200]}},
                    horizon=50,
                )
            self.assertEqual(normalized.shape, (1, 50, 2))
            self.assertEqual(paths_seen, [str(dataset_root / "data")])
            self.assertEqual(accepted[0]["episode_index"], 1571)
            self.assertAlmostEqual(float(normalized[0, 0, 0]), -.8, places=5)
            self.assertEqual(len(rejected), 1)
            self.assertIn("episode=1572", rejected[0])
            self.assertIn("actual_rows=30, metadata_rows=100", rejected[0])

    def test_strata_and_nested_balancing(self):
        chunks = np.zeros((6, 50, 7), dtype=np.float32)
        chunks[0, 25:, -1] = 2.0
        chunks[1, 10:, -1] = -2.0
        chunks[2, :, 0] = np.linspace(0, 10, 50)
        chunks[3, :, 1] = np.linspace(0, 5, 50)
        strata = action_strata(chunks)
        self.assertEqual(strata[:2], ["transition", "transition"])
        self.assertIn("movement", strata)
        self.assertIn("steady", strata)
        ordering = balanced_order(strata, seed=23)
        self.assertEqual(sorted(ordering), list(range(6)))
        self.assertEqual(ordering, balanced_order(strata, seed=23))
        self.assertEqual(len(set(ordering[:2])), 2)

    def test_new_masks_are_seeded_and_do_not_leak_full_actions(self):
        rng = np.random.default_rng(70)
        mask = make_mask(8, 50, 40, rng)
        self.assertEqual(mask.shape, (8, 50))
        self.assertTrue((mask.sum(axis=1) == 40).all())
        np.testing.assert_array_equal(mask, make_mask(8, 50, 40, np.random.default_rng(70)))
        with self.assertRaises(ValueError):
            make_mask(1, 50, 50, np.random.default_rng(0))

    def test_bootstrap_is_paired_by_real_episode(self):
        sap = {1: .20, 2: .26, 3: .22, 4: .24}
        shuffled = {1: .30, 2: .36, 3: .32, 4: .34}
        stats = paired_bootstrap_gap(sap, shuffled, iterations=300)
        self.assertEqual(stats["episodes"], 4)
        self.assertAlmostEqual(stats["shuffled_minus_sap"], .1)
        self.assertTrue(all(abs(x - .1) < 1e-10 for x in stats["ci95"]))
        with self.assertRaises(ValueError):
            paired_bootstrap_gap(sap, {1: .3}, iterations=300)

    def test_repeat_masks_are_not_counted_as_independent_episodes(self):
        rows = [
            {"model": "sap", "episode_index": 100, "mask_steps": 40, "error": 1.},
            {"model": "sap", "episode_index": 100, "mask_steps": 40, "error": 3.},
            {"model": "sap", "episode_index": 200, "mask_steps": 40, "error": 10.},
            {"model": "sap", "episode_index": 100, "mask_steps": 20, "error": 999.},
        ]
        result = aggregate_by_episode(rows, "error", mask_steps=40)
        self.assertEqual(result["sap"], {100: 2.0, 200: 10.0})

    def test_summary_writes_numeric_figure_without_fabricating_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for n in (2, 32, 128, 512):
                folder = root / "eval" / f"n{n}-seed0"
                folder.mkdir(parents=True)
                blob = {"num_chunks": n, "test_episodes": 200,
                        "prepared_test_sha256": "same-test-set", "masks": {"40": {
                            "models": {
                                "init": {"flow_relative_mse": .9},
                                "sap": {"flow_relative_mse": .8},
                                "shuffled": {"flow_relative_mse": .82},
                            },
                            "paired_gap": {"shuffled_minus_sap": .02, "ci95": [.01, .03]},
                        }}}
                (folder / "metrics_v4.json").write_text(json.dumps(blob), encoding="utf-8")
            destination = root / "summary"
            main(["summarize", "--input-root", str(root), "--output", str(destination)])
            self.assertTrue((destination / "sample_size_summary_v4.json").exists())
            self.assertTrue((destination / "sample_size_v4.png").exists())
            self.assertTrue((destination / "sample_size_v4.pdf").exists())

    def test_all_cli_help_without_torch(self):
        script = ROOT / "examples" / "analysis" / "two_sample_sap_structure_v4.py"
        for command in ("prepare", "train", "evaluate", "summarize"):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, str(script), command, "--help"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
