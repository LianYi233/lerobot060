"""Numerical provenance and paired-control setup without GPU dependencies."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_module(name, relative_path):
    path = Path(__file__).resolve().parents[2] / "src/lerobot" / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_protocol = _load_module("ntk_test_protocol", "utils/ntk_protocol.py")
_ntk = _load_module("ntk_test_metrics", "utils/ntk.py")
with patch.dict(sys.modules, {"lerobot.utils.ntk_protocol": _protocol, "lerobot.utils.ntk": _ntk}):
    _analysis = _load_module("ntk_test_analysis", "scripts/analyze_pi05_ntk_stages.py")
make_parser = _analysis.make_parser
prepare_reference = _analysis.prepare_reference
validate_manifest_extension = _analysis.validate_manifest_extension
LEGACY_MEASUREMENT_DIGESTS = _protocol.LEGACY_MEASUREMENT_DIGESTS
MEASUREMENT_FIELDS = _protocol.MEASUREMENT_FIELDS
measurement_digest = _protocol.measurement_digest
measurement_identity = _protocol.measurement_identity
validate_measurement_pair = _protocol.validate_measurement_pair


class ProtocolTests(unittest.TestCase):
    def test_auto_plot_uses_available_cumulative_steps_and_correct_cpc_label(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "results.json").write_text(json.dumps({"manifest": {
                "scope": "both",
                "stages": [
                    {"total_step": step, "config": {"cabo_enabled": True}} for step in (0, 750, 1000)
                ],
            }}))
            args = make_parser().parse_args([
                "--schedule=prompt-only", "--output-dir", directory,
                "--reference-results=original.json", "--font-path=times.ttf",
            ])
            draw = Mock()
            module = SimpleNamespace(plot_comparison=draw)
            with patch.dict(sys.modules, {"lerobot.scripts.plot_pi05_ntk_comparison": module}):
                _analysis.plot_outputs(args)
            self.assertEqual(draw.call_count, 2)
            for call in draw.call_args_list:
                self.assertEqual(call.kwargs["steps"], (0, 750, 1000))
                self.assertEqual(call.kwargs["control_label"], "Prompt learning + CPC (no stages)")
                self.assertEqual(call.kwargs["font_path"], "times.ttf")
            self.assertEqual(draw.call_args.kwargs["staged_results"], "original.json")

    def test_numerical_path_matches_audited_original_results(self):
        self.assertEqual(measurement_digest(), next(iter(LEGACY_MEASUREMENT_DIGESTS.values())))
        self.assertIsNone(measurement_identity({"source_digest": "unknown"}))

    def test_legacy_pair_and_changed_protocol(self):
        old = {field: "same" for field in MEASUREMENT_FIELDS}
        old.update(source_digest=next(iter(LEGACY_MEASUREMENT_DIGESTS)), device="cuda:0")
        new = dict(old, measurement_digest=measurement_digest(), source_digest="new CLI", device="cuda:1")
        validate_measurement_pair(old, new)
        for field in (*MEASUREMENT_FIELDS, "measurement_digest", "device"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_measurement_pair(old, dict(new, **{field: "different"}))

    def test_audited_cache_extension_retains_strict_input_checks(self):
        old = {
            "source_digest": next(iter(LEGACY_MEASUREMENT_DIGESTS)),
            "sample_digest": "same inputs",
            "stages": [{"name": "before", "weights": {"mtime_ns": 123}}],
        }
        new = dict(old, source_digest="CLI change", measurement_digest=measurement_digest())
        new["stages"] = old["stages"] + [{"name": "priming"}]
        validate_manifest_extension(old, new)
        for changed in (
            dict(new, sample_digest="other inputs"),
            dict(new, stages=[{"name": "before", "weights": {"mtime_ns": 124}}]),
            dict(new, source_digest="unknown", measurement_digest="new algorithm"),
        ):
            with self.assertRaises(ValueError):
                validate_manifest_extension(old, changed)

    def test_reference_inherits_actual_probe_settings_and_checks_architecture(self):
        config = {"chunk_size": 50}
        manifest = {
            "source_digest": next(iter(LEGACY_MEASUREMENT_DIGESTS)),
            "stages": [{"total_step": step, "config": config} for step in (0, 750, 1000, 4000)],
            "dataset_root": "/original/data",
            "dataset_repo_id": "libero",
            "sample_indices": [4, 7, 8],
            "seeds": [2, 5],
            "scope": "both",
            "output_probes": 3,
            "probe_seed": 99,
            "sketch_dim": 100,
            "sketch_seed": 44,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps({"manifest": manifest, "records": [{"seed": 2}]}))
            args = make_parser().parse_args(
                ["--reference-results", str(path), "--dataset-root", "/moved/data"]
            )
            stages = copy.deepcopy(manifest["stages"][:3])
            prepare_reference(args, stages)
            self.assertEqual(args.indices, "4,7,8")
            self.assertEqual(args.seeds, "2,5")
            self.assertEqual(args.dataset_root, "/moved/data")
            self.assertEqual(args.output_probes, 3)
            stages[1]["config"]["chunk_size"] = 25
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                prepare_reference(args, stages)
            stages = [{"total_step": 3000, "config": config}]
            with self.assertRaisesRegex(ValueError, "cumulative"):
                prepare_reference(args, stages)


if __name__ == "__main__":
    unittest.main()
