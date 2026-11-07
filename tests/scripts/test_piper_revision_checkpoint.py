"""Exercise byte-exact compact checkpoints without torch, safetensors or a GPU."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "revision_checkpoint", ROOT / "examples/piper/revision_checkpoint.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_model(path, tensors, metadata=None):
    header = {"__metadata__": metadata or {"format": "pt"}}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        offset = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class RevisionCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="piper checkpoint ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.checkpoint = self.root / "base", self.root / "run/pretrained_model"
        self.bundle, self.restored = self.root / "compact", self.root / "deploy/pretrained_model"
        self.backbone = bytes(range(256)) * 32
        write_model(self.base / MODULE.MODEL, [
            ("backbone", "U8", [len(self.backbone)], self.backbone),
            ("prompt", "F32", [1], struct.pack("<f", 0)),
            ("base_only", "I32", [1], struct.pack("<i", 9)),
        ])
        # The trained file changes both key order and offsets, adds/removes tensors,
        # and has different metadata and header padding from its base.
        write_model(self.checkpoint / MODULE.MODEL, [
            ("prompt", "F32", [1], struct.pack("<f", 5)),
            ("backbone", "U8", [len(self.backbone)], self.backbone),
            ("extra", "I32", [1], struct.pack("<i", 7)),
        ], {"format": "pt", "description": "real checkpoint"})
        for name in MODULE.REQUIRED_SIDECARS:
            (self.checkpoint / name).write_text(json.dumps({"name": name, "normalization": "task-specific"}))
        (self.checkpoint / "normalizers").mkdir()
        (self.checkpoint / "normalizers/state.safetensors").write_bytes(b"saved processor statistics")
        (self.checkpoint / "empty_directory").mkdir()

    def pack(self):
        return MODULE.pack_checkpoint(self.checkpoint, self.base, self.bundle)

    def modify_manifest(self, callback):
        path = self.bundle / MODULE.MANIFEST
        manifest = json.loads(path.read_text())
        callback(manifest)
        path.write_text(json.dumps(manifest))

    def test_roundtrip_preserves_exact_weights_keys_header_and_all_processors(self):
        manifest = self.pack()
        self.assertEqual(manifest["stored_tensor_keys"], ["prompt", "extra"])
        self.assertEqual(manifest["reused_tensor_keys"], ["backbone"])
        self.assertEqual(manifest["base_only_tensor_keys"], ["base_only"])
        self.assertEqual(manifest["stored_tensor_bytes"], 8)
        self.assertTrue(manifest["reconstruction_verified"])
        self.assertLess(manifest["compact_bytes"], manifest["source_checkpoint_bytes"])
        actual_bytes = sum(path.stat().st_size for path in self.bundle.rglob("*") if path.is_file())
        self.assertEqual(manifest["compact_bytes"], actual_bytes)
        self.assertEqual(MODULE.verify_checkpoint(self.bundle, self.base), manifest)
        result = MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertEqual(result, self.restored)
        original_files = {p.relative_to(self.checkpoint): p.read_bytes()
                          for p in self.checkpoint.rglob("*") if p.is_file()}
        restored_files = {p.relative_to(self.restored): p.read_bytes()
                          for p in self.restored.rglob("*") if p.is_file()}
        self.assertEqual(original_files, restored_files)
        self.assertTrue((self.restored / "empty_directory").is_dir())

    def test_zero_length_tensors_and_unchanged_model_need_no_delta_bytes(self):
        tensors = [("empty", "F32", [0, 4], b""), ("scalar", "F32", [], struct.pack("<f", 3))]
        write_model(self.base / MODULE.MODEL, tensors)
        write_model(self.checkpoint / MODULE.MODEL, tensors)
        manifest = self.pack()
        self.assertEqual(manifest["stored_tensor_bytes"], 0)
        MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertEqual(
            (self.checkpoint / MODULE.MODEL).read_bytes(), (self.restored / MODULE.MODEL).read_bytes()
        )

    def test_dtype_changes_are_stored_even_when_tensor_name_is_reused(self):
        write_model(self.checkpoint / MODULE.MODEL, [("prompt", "BF16", [2], b"\x80\x3f\x00\x40")])
        manifest = self.pack()
        self.assertEqual(manifest["stored_tensor_keys"], ["prompt"])
        MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertEqual(
            (self.checkpoint / MODULE.MODEL).read_bytes(), (self.restored / MODULE.MODEL).read_bytes()
        )

    def test_official_base_wrapper_alias_reuses_only_byte_identical_tensors(self):
        write_model(self.checkpoint / MODULE.MODEL, [
            ("model.backbone", "U8", [len(self.backbone)], self.backbone),
            ("model.prompt", "F32", [1], struct.pack("<f", 42)),
        ])
        manifest = self.pack()
        self.assertEqual(manifest["reused_tensor_keys"], ["model.backbone"])
        self.assertEqual(manifest["stored_tensor_keys"], ["model.prompt"])
        self.assertEqual(manifest["segments"][0]["base_key"], "backbone")
        self.assertEqual(manifest["base_only_tensor_keys"], ["base_only"])
        self.assertEqual(manifest["reused_tensor_bytes"], len(self.backbone))
        MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertEqual(
            (self.checkpoint / MODULE.MODEL).read_bytes(), (self.restored / MODULE.MODEL).read_bytes()
        )

    def test_rejects_wrong_base_even_if_tensor_names_and_file_size_match(self):
        self.pack()
        path = self.base / MODULE.MODEL
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "Base model checksum mismatch"):
            MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertFalse(self.restored.exists())

    def test_rejects_corrupted_delta_and_processor_files(self):
        self.pack()
        for name in (MODULE.DELTA, "files/policy_preprocessor.json"):
            with self.subTest(name=name):
                path = self.bundle / name
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
                self.assertFalse(self.restored.exists())
                path.write_bytes(original)

    def test_checks_final_model_hash_in_addition_to_payload_checksums(self):
        self.pack()
        self.modify_manifest(lambda value: value.update(source_model_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Reconstructed model checksum mismatch"):
            MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertFalse(self.restored.exists())
        self.assertEqual(list(self.restored.parent.glob(".pretrained_model.tmp-*")), [])

    def test_rejects_path_traversal_in_file_and_directory_inventories(self):
        self.pack()
        original = (self.bundle / MODULE.MANIFEST).read_bytes()
        mutations = (
            lambda value: value["files"].update({"../escape": {"bytes": 0, "sha256": "0" * 64}}),
            lambda value: value.update(directories=["../../escape"]),
            lambda value: value.update(directories=["/escape"]),
            lambda value: value.update(directories=["unsafe\\path"]),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.modify_manifest(mutation)
                with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                    MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
                self.assertFalse(self.restored.exists())
                (self.bundle / MODULE.MANIFEST).write_bytes(original)

    def test_rejects_tensor_offset_tampering(self):
        self.pack()
        self.modify_manifest(lambda value: value["segments"][0].update(offset=2**63))
        with self.assertRaisesRegex(ValueError, "Invalid delta tensor offset"):
            MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertFalse(self.restored.exists())

    def test_refuses_existing_outputs_without_modifying_them(self):
        self.bundle.mkdir()
        sentinel = self.bundle / "keep.txt"
        sentinel.write_text("existing data")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.pack()
        self.assertEqual(sentinel.read_text(), "existing data")
        sentinel.unlink()
        self.bundle.rmdir()
        self.pack()
        self.restored.mkdir(parents=True)
        sentinel = self.restored / "keep.txt"
        sentinel.write_text("existing deployment")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            MODULE.materialize_checkpoint(self.bundle, self.base, self.restored)
        self.assertEqual(sentinel.read_text(), "existing deployment")

    def test_refuses_symlinked_inputs_outputs_and_nested_sidecars(self):
        link = self.root / "base-link"
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            MODULE.pack_checkpoint(self.checkpoint, link, self.bundle)
        (self.checkpoint / "linked_statistics").symlink_to(self.base / MODULE.MODEL)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            self.pack()
        (self.checkpoint / "linked_statistics").unlink()
        (self.checkpoint / "linked_directory").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            self.pack()
        (self.checkpoint / "linked_directory").unlink()
        self.bundle.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            self.pack()

    def test_refuses_output_inside_checkpoint_and_missing_processors(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.pack_checkpoint(self.checkpoint, self.base, self.checkpoint / "compact")
        (self.checkpoint / "policy_postprocessor.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing required checkpoint file"):
            self.pack()

    def test_unknown_extra_bundle_files_are_not_silently_copied(self):
        self.pack()
        (self.bundle / "untracked.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "Unexpected or missing files"):
            MODULE.verify_checkpoint(self.bundle, self.base)

    def test_refuses_low_disk_space_and_removes_only_its_temporary_output(self):
        with patch.object(MODULE.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
            with self.assertRaisesRegex(ValueError, "Insufficient disk space"):
                self.pack()
        self.assertFalse(self.bundle.exists())
        self.assertTrue((self.checkpoint / MODULE.MODEL).exists())
        self.assertEqual(list(self.root.glob(".compact*")), [])

    def test_malformed_safetensors_offsets_and_duplicate_json_keys_are_rejected(self):
        headers = (
            b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
            b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{}}',
            b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,1]}}',
        )
        for header in headers:
            with self.subTest(header=header):
                (self.checkpoint / MODULE.MODEL).write_bytes(struct.pack("<Q", len(header)) + header + b"xx")
                with self.assertRaises(ValueError):
                    self.pack()
                self.assertFalse(self.bundle.exists())


if __name__ == "__main__":
    unittest.main()
