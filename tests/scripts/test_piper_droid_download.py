"""Test pinned DROID acquisition and verification without network, ML packages or real weights."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "examples/piper/download_pi05_droid.py"
SPEC = importlib.util.spec_from_file_location("droid_download", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DroidDownloadTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "droid with spaces"
        self.root.mkdir()
        self.config = {
            "type": "pi05",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "max_state_dim": 32,
            "max_action_dim": 32,
            "chunk_size": 15,
        }
        self.weights = b"small test payload, not actual model weights"
        self.payloads = {
            "config.json": json.dumps(self.config).encode(),
            "policy_preprocessor.json": b'{"steps": []}',
            "policy_postprocessor.json": b'{"steps": []}',
            "model.safetensors": self.weights,
        }
        self.sha_patch = patch.object(MODULE, "WEIGHTS_SHA256", hashlib.sha256(self.weights).hexdigest())
        self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def populate(self):
        for name, data in self.payloads.items():
            (self.root / name).write_bytes(data)

    def test_download_uses_fixed_revision_selected_endpoint_and_no_token(self):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            (kwargs["local_dir"] / kwargs["filename"]).write_bytes(self.payloads[kwargs["filename"]])

        fake_hub = types.SimpleNamespace(hf_hub_download=fake_download)
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "0"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            MODULE.download(self.root, "https://example.test")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
            MODULE.check_download(self.root)
        self.assertEqual([call["filename"] for call in calls], list(MODULE.FILES))
        for call in calls:
            self.assertEqual(call["repo_id"], "lerobot/pi05_droid")
            self.assertEqual(call["revision"], "72824c0a93f00ce5bb8bedb7feb58953ba1da364")
            self.assertEqual(call["endpoint"], "https://example.test")
            self.assertIs(call["token"], False)
            self.assertEqual(call["local_dir"], self.root)

    def test_corrupt_or_different_weights_rejected(self):
        self.populate()
        (self.root / "model.safetensors").write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"), contextlib.redirect_stdout(io.StringIO()):
            MODULE.check_download(self.root)

    def test_missing_file_and_wrong_architecture_rejected(self):
        self.populate()
        (self.root / "policy_postprocessor.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            MODULE.check_download(self.root)
        self.populate()
        self.config["type"] = "pi0"
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "Unexpected type"):
            MODULE.check_download(self.root)

    def test_check_only_never_downloads(self):
        self.populate()
        output = io.StringIO()
        with (
            patch.object(sys, "argv", [str(SCRIPT), "--output_dir", str(self.root), "--check_only"]),
            patch.object(MODULE, "download") as download,
            contextlib.redirect_stdout(output),
        ):
            MODULE.main()
        download.assert_not_called()
        self.assertIn("DROID_FILES_OK", output.getvalue())
        self.assertIn("model loading/GPU training not tested", output.getvalue())


if __name__ == "__main__":
    unittest.main()
