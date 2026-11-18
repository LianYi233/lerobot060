"""Focused offline comparison checks; fixtures are tiny test data, not experiments."""

import copy
import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from lerobot.scripts.plot_pi05_ntk_comparison import (
    DEFAULT_STEPS,
    load_results,
    make_figure,
    parse_steps,
    plot_comparison,
    validate_document,
    validate_pair,
)
from lerobot.utils.ntk_protocol import COMPARABLE_FIELDS, LEGACY_MEASUREMENT_DIGESTS


def measurement_fixture(control=False):
    names = ("before", "step_750", "step_1000", "final") if control else (
        "before", "priming", "stage2", "final"
    )
    config = dict.fromkeys(COMPARABLE_FIELDS, 1)
    groups = {"backbone/vlm": [["vlm.weight", [2, 2]]], "backbone/action": [["action.weight", [4]]]}
    manifest = {
        "schema": 1,
        "protocol": "shared-Rademacher test protocol",
        "scope": "backbone",
        "dtype": "float32",
        "device": "cuda:1" if control else "cuda:0",
        "torch_version": "test-version",
        "seeds": [0, 3],
        "output_probes": 4,
        "probe_seed": 1729,
        "sketch_dim": 8,
        "sketch_seed": 2026,
        "dataset_repo_id": "test-only",
        "dataset_root": "/moved/path" if control else "/original/path",
        "sample_indices": [0, 1, 2],
        "sample_digest": "same-processed-test-inputs",
        "measurement_digest": "same-numerical-test-protocol",
        "source_digest": "different-non-numerical-script" if control else "original-script",
        "stages": [
            {"name": name, "local_step": step if control or step < 4000 else 3000,
             "total_step": step, "config": {**config, "training_stage": "flow" if control else name}}
            for step, name in zip(DEFAULT_STEPS, names, strict=True)
        ],
    }
    records = [
        {
            "stage": stage["name"], "total_step": stage["total_step"], "local_step": stage["local_step"],
            "seed": seed,
            "parameter_groups": copy.deepcopy(groups),
            "groups": {
                group: {
                    "effective_rank": 1.1 + seed / 10 + index / 3,
                    "parameter_normalized_energy": (0.5 if control else 1) * (index + 1) / (seed + 1),
                    "parameter_count": 4,
                    "parameter_representation": "countsketch",
                }
                for index, group in enumerate(groups)
            },
        }
        for stage in manifest["stages"] for seed in manifest["seeds"]
    ]
    return {"manifest": manifest, "records": records}


def validated(document, label="Fixture", steps=DEFAULT_STEPS):
    return validate_document(document, "test-only-results.json", label, steps)


class TestNTKComparison(unittest.TestCase):
    def test_cumulative_updates_align_different_stage_names_and_local_steps(self):
        staged = validated(measurement_fixture())
        control = validated(measurement_fixture(control=True))
        validate_pair(staged, control)
        self.assertEqual(control.records[(750, 3)]["stage"], "step_750")
        self.assertEqual(staged.records[(750, 3)]["stage"], "priming")
        self.assertEqual(staged.records[(4000, 3)]["local_step"], 3000)
        self.assertEqual(control.records[(4000, 3)]["local_step"], 4000)

    def test_probe_preprocessing_and_numerical_mismatches_are_rejected(self):
        staged = validated(measurement_fixture())
        mutations = {
            "sample_digest": "different-normalized-images",
            "probe_seed": 11,
            "output_probes": 8,
            "sketch_seed": 99,
            "sketch_dim": 16,
            "dtype": "bfloat16",
            "torch_version": "other-version",
            "measurement_digest": "other-numerical-protocol",
            "device": "cpu",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                document = measurement_fixture(control=True)
                document["manifest"][field] = value
                with self.assertRaises(ValueError):
                    validate_pair(staged, validated(document))

    def test_audited_legacy_identity_is_accepted_but_unknown_source_is_not(self):
        document = measurement_fixture()
        old_source, digest = next(iter(LEGACY_MEASUREMENT_DIGESTS.items()))
        document["manifest"].pop("measurement_digest")
        document["manifest"]["source_digest"] = old_source
        staged = validated(document)
        control = measurement_fixture(control=True)
        control["manifest"]["measurement_digest"] = digest
        validate_pair(staged, validated(control))
        document["manifest"]["source_digest"] = "unaudited-old-script"
        with self.assertRaisesRegex(ValueError, "unverified"):
            validated(document)

    def test_missing_duplicates_and_stage_mislabelling_are_rejected(self):
        for mutation in ("missing", "duplicate", "wrong_step", "extra_seed"):
            with self.subTest(mutation=mutation):
                document = measurement_fixture()
                if mutation == "missing":
                    document["records"].pop()
                elif mutation == "duplicate":
                    document["records"].append(copy.deepcopy(document["records"][0]))
                elif mutation == "wrong_step":
                    document["records"][0]["total_step"] = 750
                else:
                    document["records"][0]["seed"] = 999
                with self.assertRaises(ValueError):
                    validated(document)

    def test_architecture_and_signatures_are_checked(self):
        staged = validated(measurement_fixture())
        control = measurement_fixture(control=True)
        for stage in control["manifest"]["stages"]:
            stage["config"]["num_prompt_tokens"] = 7
        with self.assertRaisesRegex(ValueError, "num_prompt_tokens"):
            validate_pair(staged, validated(control))
        malformed = measurement_fixture()
        malformed["records"][0]["groups"]["backbone/vlm"]["parameter_count"] = 5
        with self.assertRaisesRegex(ValueError, "parameter count"):
            validated(malformed)
        malformed = measurement_fixture()
        malformed["records"][0]["parameter_groups"]["backbone/vlm"][0][0] = "other.weight"
        with self.assertRaisesRegex(ValueError, "signatures"):
            validated(malformed)

    def test_nonfinite_metrics_and_invalid_rank_are_rejected(self):
        for metric, value in (("effective_rank", 4), ("effective_rank", float("nan")),
                              ("parameter_normalized_energy", 0),
                              ("parameter_normalized_energy", float("inf"))):
            with self.subTest(metric=metric, value=value):
                document = measurement_fixture()
                document["records"][0]["groups"]["backbone/vlm"][metric] = value
                with self.assertRaises(ValueError):
                    validated(document)

    def test_partial_schedule_must_be_explicit(self):
        document = measurement_fixture(control=True)
        document["manifest"]["stages"].pop()
        document["records"] = [record for record in document["records"] if record["total_step"] < 4000]
        with self.assertRaisesRegex(ValueError, "missing cumulative"):
            validated(document)
        self.assertEqual(len(validated(document, steps=(0, 750, 1000)).stages), 3)
        self.assertEqual(parse_steps("0,750,1000"), (0, 750, 1000))

    def test_scatter_coordinates_and_shared_limits_preserve_every_observation(self):
        import matplotlib.pyplot as plt
        import numpy as np

        datasets = [validated(measurement_fixture()), validated(measurement_fixture(control=True))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, axes, _ = make_figure(datasets, DEFAULT_STEPS)
        try:
            self.assertEqual(axes.shape, (2, 4))
            for row_index, dataset in enumerate(datasets):
                for col, step in enumerate(DEFAULT_STEPS):
                    for index, module in enumerate(("vlm", "action")):
                        expected = [
                            [dataset.records[(step, seed)]["groups"][f"backbone/{module}"][metric]
                             for metric in ("effective_rank", "parameter_normalized_energy")]
                            for seed in dataset.manifest["seeds"]
                        ]
                        np.testing.assert_array_equal(
                            axes[row_index, col].collections[index].get_offsets(), expected
                        )
                    self.assertEqual(axes[row_index, col].get_xlim(), axes[0, 0].get_xlim())
                    self.assertEqual(axes[row_index, col].get_ylim(), axes[0, 0].get_ylim())
        finally:
            plt.close(fig)

    def test_output_roundtrip_and_no_outputs_for_incompatible_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_path, control_path = root / "staged.json", root / "control.json"
            staged_path.write_text(json.dumps(measurement_fixture()))
            control_path.write_text(json.dumps(measurement_fixture(control=True)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                output = plot_comparison(control_path, root / "figure", staged_results=staged_path, dpi=50)
            for extension in ("png", "pdf", "svg"):
                self.assertTrue((output / f"ntk_prompt_comparison.{extension}").is_file())
            with (output / "ntk_prompt_comparison_points.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 32)
            source = load_results(staged_path, "Three-stage prompt learning")
            first = source.records[(0, 0)]["groups"]["backbone/vlm"]
            self.assertEqual(float(rows[0]["effective_rank"]), first["effective_rank"])
            self.assertEqual(
                float(rows[0]["parameter_normalized_energy"]), first["parameter_normalized_energy"]
            )
            metadata = json.loads((output / "ntk_prompt_comparison_metadata.json").read_text())
            self.assertEqual(metadata["plotted_points"], 32)
            self.assertEqual(metadata["inputs"][0]["sha256"], source.file_digest)
            bad = measurement_fixture(control=True)
            bad["manifest"]["sample_digest"] = "wrong-inputs"
            control_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "sample_digest"):
                plot_comparison(control_path, root / "must-not-exist", staged_results=staged_path)
            self.assertFalse((root / "must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
