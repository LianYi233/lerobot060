"""Check that matching probe images/actions never changes the measured model."""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _module(name, relative):
    root = Path(__file__).resolve().parents[2] / "src/lerobot"
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(
    sys.modules,
    {
        "lerobot.utils.ntk": _module("test_reference_ntk", "utils/ntk.py"),
        "lerobot.utils.ntk_protocol": _module("test_reference_protocol", "utils/ntk_protocol.py"),
    },
):
    analysis = _module("test_paired_analyzer", "scripts/analyze_pi05_ntk_stages.py")


class PairedInputsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = analysis.make_parser().parse_args(
            ["--schedule=prompt-only", "--dataset-root", str(self.root)]
        )
        self.stages = [self.stage("control", step) for step in (0, 750)]
        self.reference = {
            "manifest": {
                "stages": [self.stage("staged", step) for step in (0, 750)],
                "sample_digest": "1" * 64,
                "sample_indices": [2, 4],
                "dataset_root": str(self.root),
            }
        }
        self.samples = [{"action": "sample A"}, {"action": "sample B"}]
        self.selected = []

    def stage(self, group, step):
        path = self.root / group / "checkpoints" / f"{step:06d}" / "pretrained_model"
        path.mkdir(parents=True)
        for filename in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            (path / filename).write_text("{}")
        return {
            "checkpoint": str(path),
            "total_step": step,
            "local_step": step,
            "name": "before" if step == 0 else "step_750",
            "config": {},
        }

    def load(self, reference, actual=None):
        # Use the selected checkpoint's preprocessing pipeline exactly as load_samples
        # does. No model/processor weights or GPU runtime are needed for routing tests.
        def read_samples(args, preprocessing_stages):
            self.selected.append(Path(preprocessing_stages[0]["checkpoint"]).resolve())
            return self.samples, [2, 4]

        with (
            patch.object(analysis, "load_samples", side_effect=read_samples),
            patch.object(analysis, "fingerprint_samples", return_value=actual or "1" * 64),
        ):
            return analysis.load_paired_samples(self.args, self.stages, reference)

    def test_reference_processor_replays_inputs_but_keeps_all_control_checkpoints(self):
        original = copy.deepcopy(self.stages)
        source = self.reference["manifest"]["stages"][0]["checkpoint"]
        self.assertEqual(self.load(self.reference), (self.samples, [2, 4], "1" * 64, source))
        self.assertEqual(self.selected[-1], Path(source))
        self.assertEqual(self.stages, original)

    def test_explicitly_recorded_processor_precedes_old_stage_zero(self):
        recorded = self.stage("effective-processor", 0)["checkpoint"]
        self.reference["manifest"]["sample_preprocessor"] = recorded
        self.assertEqual(self.load(self.reference)[-1], recorded)
        self.assertEqual(self.selected[-1], Path(recorded))

    def test_explicit_relocated_reference_accepts_model_directory_and_its_parent(self):
        relocated = Path(self.stage("relocated", 0)["checkpoint"])
        original = copy.deepcopy(self.stages)
        for argument in (relocated.parent, relocated):
            with self.subTest(directory=argument):
                self.args.reference_preprocessor = str(argument)
                self.load(self.reference)
                self.assertEqual(self.selected[-1], relocated)
                self.assertEqual(self.stages, original)

    def test_unavailable_historical_path_requires_exact_digest_from_control_inputs(self):
        self.reference["manifest"]["stages"][0]["checkpoint"] = str(self.root / "deleted")
        self.load(self.reference)
        self.assertEqual(self.selected[-1], Path(self.stages[0]["checkpoint"]))
        with self.assertRaises(ValueError) as error:
            self.load(self.reference, actual="2" * 64)
        self.assertIn("1" * 64, str(error.exception))
        self.assertIn("2" * 64, str(error.exception))

    def test_explicit_reference_processor_cannot_override_digest_check(self):
        self.args.reference_preprocessor = self.reference["manifest"]["stages"][0]["checkpoint"]
        with self.assertRaises(ValueError) as error:
            self.load(self.reference, actual="3" * 64)
        self.assertIn("1" * 64, str(error.exception))
        self.assertIn("3" * 64, str(error.exception))

    def test_standalone_analysis_uses_its_own_initial_processor(self):
        source = self.stages[0]["checkpoint"]
        self.assertEqual(self.load(None, actual="4" * 64), (self.samples, [2, 4], "4" * 64, source))
        self.assertEqual(self.selected[-1], Path(source))

    def test_missing_explicit_override_is_not_silently_replaced(self):
        self.args.reference_preprocessor = str(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            self.load(self.reference)
        self.assertFalse(self.selected)


if __name__ == "__main__":
    unittest.main()
