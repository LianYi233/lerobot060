"""Read-only audit tests; no policy, GPU, CAN or RealSense dependencies."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit", ROOT / "piper_checkpoint_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditTests(unittest.TestCase):
    def checkpoint(self, root, episodes=None):
        path = root / "checkpoints/003000/pretrained_model"
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"type": "pi05", "train_action_projections": True}))
        if episodes != "missing":
            (path / "train_config.json").write_text(
                json.dumps({"steps": 3000, "batch_size": 16, "dataset": {"episodes": episodes}})
            )
        (path / "policy_preprocessor.json").write_text(
            json.dumps({"steps": [{"state_file": "normalization.safetensors"}]})
        )
        (path / "normalization.safetensors").write_bytes(b"test-only-state")
        return path

    def test_missing_selection_is_not_mistaken_for_all_episodes(self):
        for episodes, expected in (
            ("missing", "unknown: train_config.json missing"),
            ([0], "explicit episode subset"),
            (None, "all episodes before eval_split"),
        ):
            with self.subTest(episodes=episodes), tempfile.TemporaryDirectory() as directory:
                path = self.checkpoint(Path(directory), episodes)
                report = AUDIT.checkpoint_report(path)
                self.assertEqual(report["training"]["episode_scope"], expected)
                self.assertFalse(report["weights_verified"])
                self.assertIsNone(report["small_file_sha256"]["policy_postprocessor.json"])
                self.assertEqual(len(report["small_file_sha256"]["normalization.safetensors"]), 64)

    def test_checkpoint_state_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.checkpoint(Path(directory))
            (path / "policy_preprocessor.json").write_text(
                json.dumps({"steps": [{"state_file": "../../../outside"}]})
            )
            with self.assertRaisesRegex(ValueError, "escapes checkpoint"):
                AUDIT.checkpoint_report(path)

    def test_comparison_reports_semantics_and_processor_differences(self):
        original = {
            "versions": {"transformers": "5.3.0"},
            "repo_sources": {"model": "same"},
            "transformers_sources": {"gemma": "old"},
            "embedding_probe": {"text_scale_from_unit_embedding_weights": 1},
            "checkpoint": {"small_file_sha256": {"normalization": "a"}},
        }
        current = deepcopy(original)
        current["versions"]["transformers"] = "5.5.4"
        current["transformers_sources"]["gemma"] = "new"
        current["embedding_probe"]["text_scale_from_unit_embedding_weights"] = 4
        current["checkpoint"]["small_file_sha256"]["normalization"] = "b"
        self.assertEqual(
            set(AUDIT.compare_reports(current, original)["different_fields"]),
            {"versions/transformers", "transformers_sources", "embedding_probe", "checkpoint/normalization"},
        )
        self.assertEqual(AUDIT.compare_reports(original, original)["different_fields"], [])

    def test_cli_metadata_only_does_not_import_ml_or_modify_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.checkpoint(root, [0])
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.dict(sys.modules, {"torch": None, "transformers": None, "piper_sdk": None}),
            ):
                AUDIT.main(["--policy_path", str(path), "--output", str(root / "report.json")])
            result = json.loads((root / "report.json").read_text())
            self.assertEqual(result["checkpoint"]["training"]["episodes"], [0])
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                AUDIT.main(["--policy_path", str(path), "--output", str(root / "report.json")])

    def test_probe_failure_is_reported_without_losing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.checkpoint(root)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(AUDIT, "probe_embeddings", side_effect=ImportError("bad dependency")),
            ):
                AUDIT.main(
                    ["--policy_path", str(path), "--probe_embeddings", "--output", str(root / "report.json")]
                )
            result = json.loads((root / "report.json").read_text())
            self.assertIn("bad dependency", result["embedding_probe"]["error"])
            self.assertIn("training", result["checkpoint"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
