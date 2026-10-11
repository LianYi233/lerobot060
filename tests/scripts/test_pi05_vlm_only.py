"""Test two-source GPU dispatch and normalization-matched controls without CUDA."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lerobot.scripts.libero_vlm_only import prepare_baseline, validate_checkpoint  # noqa: E402

TRAIN = ROOT / "examples/training/train_pi05_vlm_only.sh"
PAIR = ROOT / "examples/training/train_pi05_vlm_only_pair.sh"
EVAL = ROOT / "run_eval_libero_vlm_only.sh"


class VLMOnlyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vlm only ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.dataset = self.work / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/stats.json").write_text(
            json.dumps(
                {
                    key: {"q01": [0] * size, "q99": [1] * size}
                    for key, size in (("observation.state", 8), ("action", 7))
                }
            )
        )
        self.tokenizer = self.work / "tokenizer"
        self.tokenizer.mkdir()
        for kind in ("base", "finetuned"):
            self.make_model(self.source(kind), kind, source=True)
            for step in (0, 3000):
                self.make_model(self.model(kind, step), kind)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "DRY_RUN": "true",
            "DATASET_ROOT": str(self.dataset),
            "TOKENIZER_PATH": str(self.tokenizer),
            "BASE_PRETRAINED_PATH": str(self.source("base")),
            "FINETUNED_PRETRAINED_PATH": str(self.source("finetuned")),
            "OUTPUT_ROOT": str(self.work / "training"),
            "LOG_ROOT": str(self.work / "logs"),
            "CKPT_ROOT": str(self.work / "checkpoints"),
            "BASE_OUTPUT": str(self.work / "eval"),
        }

    def source(self, kind):
        return self.work / "sources" / kind

    def model(self, kind, step):
        return self.work / "checkpoints" / f"pi05-{kind}-vlm1-seed0/checkpoints/{step:06d}/pretrained_model"

    def make_model(self, path, kind, source=False):
        path.mkdir(parents=True)
        mode = "MEAN_STD" if kind == "finetuned" or source else "QUANTILES"
        norm = {"ACTION": mode, "STATE": mode, "VISUAL": "IDENTITY"}
        config = {
            "type": "pi05",
            "dtype": "bfloat16",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "normalization_mapping": norm,
        }
        weights = {"model.backbone.weight": np.ones((4, 4), dtype=np.float32)}
        if not source:
            config.update(
                vlm_prompt_only=True,
                num_vlm_prompt_tokens=1,
                num_prompt_tokens=0,
                training_stage="flow",
                next_action_pretrain_steps=0,
                next_action_bridge_steps=0,
                cabo_enabled=False,
                train_action_projections=False,
                use_peft=False,
            )
            weights["model.vlm_prompt_tokens.weight"] = np.ones((1, 2048), dtype=np.float32)
            weights["model.prompt_tokens.weight"] = np.empty((0, 1024), dtype=np.float32)
            (path / "train_config.json").write_text(
                json.dumps({"policy": {"pretrained_path": str(self.source(kind))}})
            )
        (path / "config.json").write_text(json.dumps(config))
        save_file(weights, path / "model.safetensors")
        for filename, registry, features in (
            ("policy_preprocessor.json", "normalizer_processor", {"observation.state": 8, "action": 7}),
            ("policy_postprocessor.json", "unnormalizer_processor", {"action": 7}),
        ):
            missing_stats = source and kind == "base"
            feature_map = (
                {}
                if missing_stats
                else {
                    key: {"type": "ACTION" if key == "action" else "STATE", "shape": [size]}
                    for key, size in features.items()
                }
            )
            step = {
                "registry_name": registry,
                "config": {"eps": 1e-8, "norm_map": norm, "features": feature_map},
            }
            if not missing_stats:
                step["state_file"] = registry + ".safetensors"
                stats = {
                    f"{key}.{stat}": np.ones(size, dtype=np.float32)
                    for key, size in features.items()
                    for stat in (("mean", "std") if mode == "MEAN_STD" else ("q01", "q99"))
                }
                save_file(stats, path / step["state_file"])
            (path / filename).write_text(json.dumps({"steps": [step]}))

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["bash", str(script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [
            shlex.split(line)
            for line in result.stdout.splitlines()
            if line.startswith("env CUDA_VISIBLE_DEVICES=")
        ]

    def test_default_pair_assigns_distinct_gpus_sources_ports_and_prompt_only_recipe(self):
        commands = self.commands(self.run_script(PAIR, "0"))
        self.assertEqual(len(commands), 2)
        for index, (kind, command) in enumerate(zip(("base", "finetuned"), commands, strict=True)):
            for arg in (
                f"CUDA_VISIBLE_DEVICES={index}",
                "--num_processes=1",
                "--batch_size=32",
                f"--main_process_port={29531 + index}",
                f"--policy.path={self.source(kind)}",
                "--policy.vlm_prompt_only=true",
                "--policy.num_vlm_prompt_tokens=1",
                "--policy.num_prompt_tokens=0",
                "--policy.cabo_enabled=false",
                "--policy.train_action_projections=false",
                "--policy.next_action_pretrain_steps=0",
                "--policy.next_action_bridge_steps=0",
                "--steps=3000",
                "--policy.dtype=bfloat16",
                "--policy.optimizer_lr=2.5e-5",
            ):
                self.assertIn(arg, command)
            self.assertIn(
                f"--preserve_pretrained_normalization={'true' if kind == 'finetuned' else 'false'}", command
            )
            self.assertNotIn("--multi_gpu", command)
        self.assertFalse((self.work / "training").exists())

    def test_ddp_pair_keeps_global_batch_32_and_supports_custom_tokens(self):
        commands = self.commands(
            self.run_script(PAIR, "2", TRAIN_LAYOUT="ddp", GPU_IDS="2,3", VLM_PROMPT_TOKENS="16")
        )
        for command in commands:
            for arg in (
                "CUDA_VISIBLE_DEVICES=2,3",
                "--num_processes=2",
                "--multi_gpu",
                "--batch_size=16",
                "--policy.num_vlm_prompt_tokens=16",
                "--seed=2",
            ):
                self.assertIn(arg, command)

    def test_invalid_or_conflicting_settings_fail_before_launch(self):
        for env in (
            {"GPU_IDS": "0,0"},
            {"GPU_IDS": "0"},
            {"TRAIN_LAYOUT": "wrong"},
            {"NUM_PROCESSES": "2"},
            {"PRETRAINED_PATH": "wrong"},
            {"DRY_RUN": "yes"},
        ):
            self.assertNotEqual(self.run_script(PAIR, "0", **env).returncode, 0)
        for arg in (
            "--policy.num_prompt_tokens=16",
            "--policy.vlm_prompt_only=false",
            "--policy.cabo_enabled=true",
        ):
            self.assertNotEqual(self.run_script(TRAIN, "base", "0", arg).returncode, 0)
        self.assertNotEqual(
            self.run_script(TRAIN, "base", "0", GPU_IDS="0,1", NUM_PROCESSES="1").returncode, 0
        )

    def test_missing_second_source_or_existing_output_prevents_both_launches(self):
        (self.work / "training/pi05-finetuned-vlm1-seed0").mkdir(parents=True)
        result = self.run_script(PAIR, "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertNotIn("Both sources passed", result.stdout)
        self.assertFalse((self.work / "logs").exists())

    def test_matched_base_uses_original_weights_and_initial_quantile_statistics(self):
        output = self.work / "baseline"
        source_before = {p.name: p.read_bytes() for p in self.source("base").iterdir()}
        prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["num_vlm_prompt_tokens"], 0)
        self.assertFalse(config["vlm_prompt_only"])
        self.assertEqual(config["normalization_mapping"]["ACTION"], "QUANTILES")
        self.assertEqual((output / "model.safetensors").resolve(), self.source("base") / "model.safetensors")
        self.assertEqual(
            (output / "normalizer_processor.safetensors").read_bytes(),
            (self.model("base", 0) / "normalizer_processor.safetensors").read_bytes(),
        )
        self.assertEqual(source_before, {p.name: p.read_bytes() for p in self.source("base").iterdir()})
        # Reuse succeeds; changed processor contents cannot silently reuse the old baseline.
        prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")
        (output / "policy_postprocessor.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Baseline processor changed"):
            prepare_baseline(self.source("base"), self.model("base", 0), output, 1, "base")

    def test_processor_drift_and_wrong_source_are_rejected(self):
        state = self.model("finetuned", 3000) / "normalizer_processor.safetensors"
        tensors = load_file(state)
        tensors["action.mean"][0] += 1
        save_file(tensors, state)
        with self.assertRaisesRegex(ValueError, "statistics differ"):
            validate_checkpoint(self.model("finetuned", 3000), 1, reference=self.model("finetuned", 0))
        with self.assertRaisesRegex(ValueError, "different source path"):
            validate_checkpoint(self.model("base", 0), 1, source=self.source("finetuned"))
        result = self.run_script(EVAL, "both", "all", "all", "0", DRY_RUN="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Evaluating source=", result.stdout)
        self.assertFalse((self.work / "eval").exists())

    def test_dry_eval_plans_six_conditions_without_writing_baselines(self):
        result = self.run_script(EVAL, "both", "all", "all", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Evaluating source="), 6)
        self.assertIn("/000000/pretrained_model", result.stdout)
        self.assertIn("/003000/pretrained_model", result.stdout)
        self.assertIn("matched_baseline/pretrained_model", result.stdout)
        self.assertFalse(list((self.work / "checkpoints").glob("*/matched_baseline")))
        self.assertFalse((self.work / "eval").exists())

    def test_real_parallel_dispatch_waits_for_both_and_propagates_failure(self):
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        accelerate = bin_dir / "accelerate"
        accelerate.write_text(
            f"#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n"
            "source = next(x.split('=',1)[1] for x in sys.argv if x.startswith('--policy.path='))\n"
            "kind = Path(source).name\nwork = Path(os.environ['CAPTURE_ROOT'])\n"
            "(work / (kind + '.ready')).touch()\nlimit = time.monotonic() + 10\n"
            "while not all((work / (x + '.ready')).exists() for x in ['base', 'finetuned']):\n"
            "    if time.monotonic() > limit: sys.exit(19)\n    time.sleep(0.02)\n"
            "(work / (kind + '.json')).write_text(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'args': sys.argv}))\n"
            "print('mock completed ' + kind, flush=True)\nsys.exit(7 if kind == 'base' else 0)\n"
        )
        trainer = bin_dir / "lerobot-train"
        trainer.write_text("#!/bin/sh\nexit 0\n")
        for path in (accelerate, trainer):
            path.chmod(0o755)
        cache = dict.fromkeys(
            ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"), str(self.work / "cache")
        )
        result = self.run_script(
            PAIR,
            "0",
            DRY_RUN="false",
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            CAPTURE_ROOT=str(self.work),
            **cache,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        for kind, gpu in (("base", "0"), ("finetuned", "1")):
            self.assertEqual(json.loads((self.work / f"{kind}.json").read_text())["gpu"], gpu)
            self.assertIn(
                f"mock completed {kind}", (self.work / f"logs/pi05-{kind}-vlm1-seed0.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
