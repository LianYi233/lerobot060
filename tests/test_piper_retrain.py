"""CPU tests for raw-data auditing and selecting real, saved evaluation checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lerobot.utils.piper_data_audit import audit_arrays, audit_dataset
from lerobot.utils.piper_training_eval import (
    PIPER_NAMES,
    episode_stratified_indices,
    update_best_saved_checkpoints,
)


class PiperRetrainTest(unittest.TestCase):
    def test_same_frame_equality_does_not_cross_episode_boundary(self):
        states = np.repeat(np.array([0, 1, 2, 100, 101])[:, None], 7, axis=1)
        # Deliberately shuffled storage: audit sorts by episode and frame first.
        order = [4, 1, 3, 0, 2]
        report = audit_arrays(
            states[order],
            states[order],
            np.array([0, 0, 0, 1, 1])[order],
            np.array([0, 1, 2, 0, 1])[order],
            np.array([0, 0.1, 0.2, 0, 0.1])[order],
            10,
        )
        same = report["comparisons"]["action_t_vs_state_t"]
        following = report["comparisons"]["action_t_vs_state_t_plus_1"]
        self.assertEqual(same["all_coordinates_equal_fraction_at_1e-6"], 1)
        self.assertEqual(following["pairs"], 3)
        self.assertAlmostEqual(following["joint_mae_deg"], np.rad2deg(1))
        self.assertEqual(following["gripper_mae_dataset_units"], 1)
        for frame, timestamp in (([0, 0], [0, 0.1]), ([0, 2], [0, 0.2]), ([0, 1], [0, 0])):
            with self.assertRaises(ValueError):
                audit_arrays(np.zeros((2, 7)), np.zeros((2, 7)), [0, 0], frame, timestamp, 10)

    def test_audit_raw_parquet_reorders_names_and_checks_metadata(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            state = np.array([np.arange(7), np.arange(7) + 0.1])
            info = {
                "fps": 10,
                "total_frames": 2,
                "total_episodes": 1,
                "features": {
                    "action": {"names": PIPER_NAMES[::-1], "shape": [7]},
                    "observation.state": {"names": PIPER_NAMES, "shape": [7]},
                },
            }
            (root / "meta/info.json").write_text(json.dumps(info))
            pq.write_table(
                pa.table(
                    {
                        "action": state[:, ::-1].tolist(),
                        "observation.state": state.tolist(),
                        "episode_index": [0, 0],
                        "frame_index": [0, 1],
                        "timestamp": [0.0, 0.1],
                    }
                ),
                root / "data/chunk-000/file-000.parquet",
            )
            report = audit_dataset(root)
            self.assertEqual(report["comparisons"]["action_t_vs_state_t"]["joint_mae_deg"], 0)
            info["total_frames"] = 3
            (root / "meta/info.json").write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "counts"):
                audit_dataset(root)

    def test_stratification_is_deterministic_unique_and_covers_episodes(self):
        episodes = np.repeat([2, 7, 9], [4, 10, 22])
        for budget in (1, 3, 8, 100):
            indices = episode_stratified_indices(episodes, budget, 17)
            np.testing.assert_array_equal(indices, episode_stratified_indices(episodes, budget, 17))
            self.assertEqual(len(indices), min(budget, len(episodes)))
            self.assertEqual(len(np.unique(indices)), len(indices))
            if budget >= 3:
                self.assertEqual(set(episodes[indices]), {2, 7, 9})

    def test_best_links_use_validation_and_only_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save(step, joint, gripper):
                checkpoint = root / "checkpoints" / f"{step:06d}"
                (checkpoint / "pretrained_model").mkdir(parents=True)
                (checkpoint / "pretrained_model/config.json").write_text("{}")
                return update_best_saved_checkpoints(
                    root,
                    checkpoint,
                    step,
                    {
                        "action_train/h8/joint_mae_deg": 0,
                        "action_val/h8/joint_mae_deg": joint,
                        "action_val/h8/gripper_mae": gripper,
                    },
                    8,
                )

            save(1000, 4, 0.03)
            result = save(2000, 3, 0.05)
            self.assertEqual(result["split"], "val")
            self.assertEqual((root / "checkpoints/best_joint").resolve().name, "002000")
            self.assertEqual((root / "checkpoints/best_gripper").resolve().name, "001000")
            # Restart/reload preserves the previous winner rather than resetting to latest.
            save(3000, 5, 0.06)
            self.assertEqual((root / "checkpoints/best_joint").resolve().name, "002000")
            with self.assertRaisesRegex(ValueError, "completed"):
                update_best_saved_checkpoints(root, root / "checkpoints/004000", 4000, {}, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
