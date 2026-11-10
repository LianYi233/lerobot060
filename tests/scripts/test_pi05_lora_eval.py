"""CPU-only contracts for the matched-4,000-update LoRA evaluation path.

Shell captures and isolated factory execution avoid importing torch, downloading
models, starting training, or creating LIBERO environments.
"""

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
EVAL_LAUNCHER = ROOT / "run_eval_pi05_revision_4000.sh"
SHARED_EVAL = ROOT / "run_eval_libero-full.sh"
SEQUENTIAL_EVAL = ROOT / "run_eval_pi05_revision_4000_sequential.sh"
FACTORY = ROOT / "src/lerobot/policies/factory.py"
CONFIG = ROOT / "src/lerobot/policies/pi05/configuration_pi05.py"


def load_function(source, name, namespace):
    """Execute the real function with explicit dependency stubs, not a copied implementation."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), f"<isolated {name}>", "exec"), namespace)
    return namespace[name]


class LoraEvaluationLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-lora-eval-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.env = dict(
            os.environ,
            CKPT_ROOT=str(self.work / "checkpoints"),
            BASE_OUTPUT=str(self.work / "eval"),
            GPU_ID="2",
            EPISODES_PER_TASK="10",
            LIBERO_CONFIG_PATH=str(self.work / "libero-config"),
        )
        for name in ("BASE_CKPT", "RUN_NAME_OVERRIDE", "EVAL_STEP"):
            self.env.pop(name, None)

    def checkpoint(self, variant, seed="7"):
        return (
            self.work
            / "checkpoints"
            / f"pi05-revision-{variant}-seed{seed}-total4000"
            / "checkpoints"
            / "004000"
            / "pretrained_model"
        )

    def capture_eval(self, variant, *, mode="all", seed="7"):
        self.checkpoint(variant, seed).mkdir(parents=True)
        binary_dir = self.work / "capture-bin"
        binary_dir.mkdir(exist_ok=True)
        fake_python = binary_dir / "python"
        fake_python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "names = ('CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'PYOPENGL_PLATFORM', "
            "'MUJOCO_EGL_DEVICE_ID', 'LIBERO_CONFIG_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')\n"
            "Path(os.environ['EVAL_CAPTURE']).write_text(json.dumps({"
            "'argv': sys.argv[1:], 'env': {name: os.environ.get(name) for name in names}}))\n"
            "sys.exit(23)\n"
        )
        fake_python.chmod(0o755)
        capture = self.work / f"capture-{variant}-{seed}.json"
        result = subprocess.run(
            ["bash", str(EVAL_LAUNCHER), variant, mode, seed],
            env=dict(
                self.env,
                EVAL_CAPTURE=str(capture),
                PATH=f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        # The stub intentionally stops before any model/environment work; the
        # launcher must propagate that failure rather than invent an eval result.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("evaluation failed (exit=23)", result.stdout)
        return json.loads(capture.read_text()), result

    def test_lora_final_adapter_uses_004000_and_selected_gpu(self):
        captured, result = self.capture_eval("lora_r16")
        self.assertIn(f"--policy.path={self.checkpoint('lora_r16')}", captured["argv"])
        self.assertIn("flow_checkpoint=004000 cumulative_updates=4000", result.stdout)
        self.assertIn("training_seed=7", result.stdout)
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(captured["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(captured["env"]["TRANSFORMERS_OFFLINE"], "1")
        self.assertFalse(any(arg.startswith("--seed=") for arg in captured["argv"]))
        self.assertFalse(list((self.work / "eval").rglob("eval_info.json")))

    def test_lora_reuses_exact_shared_libero_protocol(self):
        direct, _ = self.capture_eval("direct_dual")
        lora, _ = self.capture_eval("lora_r16")

        def protocol(arguments):
            return [arg for arg in arguments if not arg.startswith(("--policy.path=", "--output_dir="))]

        self.assertEqual(protocol(direct["argv"]), protocol(lora["argv"]))
        for arg in (
            "--env.type=libero",
            "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
            "--env.control_mode=relative",
            "--env.max_parallel_tasks=1",
            "--eval.batch_size=1",
            "--eval.n_episodes=10",
            "--policy.n_action_steps=10",
            "--policy.use_amp=false",
            "--policy.device=cuda",
            "--policy.compile_model=false",
            "--policy.gradient_checkpointing=false",
        ):
            self.assertIn(arg, lora["argv"])
        for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBERO_CONFIG_PATH"):
            self.assertEqual(direct["env"][name], lora["env"][name])
        self.assertIsNone(lora["env"]["MUJOCO_EGL_DEVICE_ID"])

        tree = ast.parse((ROOT / "src/lerobot/configs/eval.py").read_text())
        config_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "EvalPipelineConfig"
        )
        seed_field = next(
            node for node in config_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "seed"
        )
        self.assertEqual(ast.literal_eval(seed_field.value), 1000)

    def test_lora_suite_and_episode_overrides_still_use_shared_protocol(self):
        self.env["EPISODES_PER_TASK"] = "3"
        captured, _ = self.capture_eval("lora_r16", mode="libero_10", seed="2")
        self.assertIn("--env.task=libero_10", captured["argv"])
        self.assertIn("--eval.n_episodes=3", captured["argv"])
        self.assertIn(f"--policy.path={self.checkpoint('lora_r16', '2')}", captured["argv"])

    def test_lora_missing_final_checkpoint_fails_before_python(self):
        result = subprocess.run(
            ["bash", str(EVAL_LAUNCHER), "lora_r16", "all", "7"],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(f"ERROR: checkpoint not found: {self.checkpoint('lora_r16')}", result.stdout)

    def test_sequential_explicit_lora_uses_004000_without_changing_default_schedule(self):
        output = self.work / "eval" / (
            "libero060-all-pi05-revision-lora_r16-seed7-total4000-004000-libero-all4-resume"
        )
        output.mkdir(parents=True)
        suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        (output / "eval_info.json").write_text(
            json.dumps({
                "per_group": {suite: {"n_episodes": 100, "pc_success": 75.0} for suite in suites},
                "overall": {"n_episodes": 400, "pc_success": 75.0},
            })
        )
        result = subprocess.run(
            ["bash", str(SEQUENTIAL_EVAL), "7", "lora_r16"],
            env=dict(self.env, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[SKIP] lora_r16", result.stdout)
        self.assertNotIn("[RUN]", result.stdout)
        self.assertTrue(
            any(line.split() == ["lora_r16", *(["75.00%"] * 5)] for line in result.stdout.splitlines())
        )
        self.assertIn(
            "VARIANTS=(direct_dual full_reference dual_prompt_only action_flow_warmup)",
            SEQUENTIAL_EVAL.read_text(),
        )


class LoraBaseManifestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-lora-base-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.adapter = self.work / "adapter"
        self.base = self.work / "base"
        self.adapter.mkdir()
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"base weights")
        (self.base / "config.json").write_text("{}")
        (self.adapter / "config.json").write_text(json.dumps({"type": "pi05", "adaptation_mode": "lora"}))
        (self.adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(self.base)})
        )
        embedded = SHARED_EVAL.read_text().split("<<'PY' 2>&1 | tee -a \"$LOG\"\n", 1)[1]
        embedded = embedded.split("\nPY\n", 1)[0]
        self.manifest = load_function(embedded, "lora_base_manifest", {"json": json, "Path": Path})

    def test_lora_manifest_tracks_local_base_changes(self):
        before = self.manifest(self.adapter)
        self.assertEqual(before["peft_base"]["path"], str(self.base.resolve()))
        self.assertEqual(
            [item[0] for item in before["peft_base"]["files"]], ["config.json", "model.safetensors"]
        )
        (self.base / "model.safetensors").write_bytes(b"different frozen base weights")
        self.assertNotEqual(before, self.manifest(self.adapter))

    def test_missing_local_base_is_rejected(self):
        (self.base / "model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "local base is missing or incomplete"):
            self.manifest(self.adapter)

    def test_prompt_manifests_keep_original_shape(self):
        (self.adapter / "config.json").write_text(json.dumps({"type": "pi05"}))
        self.assertEqual(self.manifest(self.adapter), {})


class _FakeModule:
    def to(self, _device):
        return self


class LoraFactoryLoadingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi05-lora-factory-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.adapter_path = self.work / "adapter"
        self.base_path = self.work / "base"
        self.adapter_path.mkdir()
        self.base_path.mkdir()
        (self.base_path / "model.safetensors").write_bytes(b"base")
        target_assignment = next(
            node
            for node in ast.parse(CONFIG.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PI05_LORA_TARGET_MODULES"
                for target in node.targets
            )
        )
        self.targets = ast.literal_eval(target_assignment.value)
        self.events = []
        self.validation_error = None
        self.adapter_config = SimpleNamespace(
            target_modules=self.targets,
            modules_to_save=[],
            base_model_name_or_path=str(self.base_path),
        )
        self.config = SimpleNamespace(
            type="pi05",
            device="cpu",
            pretrained_path=self.adapter_path,
            pretrained_revision=None,
            adaptation_mode="lora",
            use_peft=True,
            input_features={},
            output_features={},
            num_prompt_tokens=0,
            num_vlm_prompt_tokens=0,
        )

        def validate(adapter_config):
            self.events.append(("validate", adapter_config))
            if self.validation_error:
                raise ValueError(self.validation_error)

        class BasePolicy(_FakeModule):
            pass

        self.policy = BasePolicy()
        self.policy.config = self.config
        self.policy._validate_peft_config = validate
        self.policy.model = SimpleNamespace(_freeze_backbones=lambda: self.events.append(("freeze",)))

        class Wrapper(_FakeModule):
            def get_base_model(wrapper_self):
                return self.policy

        self.wrapper = Wrapper()

        def load_base(**kwargs):
            self.events.append(("base", kwargs))
            return self.policy

        def load_config(path, **kwargs):
            self.events.append(("peft_config", path, kwargs))
            return self.adapter_config

        def load_adapter(policy, path, **kwargs):
            self.events.append(("adapter", path, kwargs))
            self.assertIs(policy, self.policy)
            return self.wrapper

        peft = ModuleType("peft")
        peft.PeftConfig = SimpleNamespace(from_pretrained=load_config)
        peft.PeftModel = SimpleNamespace(from_pretrained=load_adapter)
        module_patch = patch.dict(sys.modules, {"peft": peft})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        namespace = {
            "logging": logging,
            "Path": Path,
            "PI05_LORA_TARGET_MODULES": self.targets,
            "torch": SimpleNamespace(nn=SimpleNamespace(Module=_FakeModule)),
            "FeatureType": SimpleNamespace(ACTION="action"),
            "get_policy_class": lambda _name: SimpleNamespace(from_pretrained=load_base),
            "env_to_policy_features": lambda _cfg: {"action": SimpleNamespace(type="action")},
            "validate_visual_features_consistency": lambda *_args: None,
        }
        self.make_policy = load_function(FACTORY.read_text(), "make_policy", namespace)

    def load(self):
        return self.make_policy(self.config, env_cfg=SimpleNamespace())

    def test_local_lora_is_validated_before_attachment(self):
        self.assertIs(self.load(), self.wrapper)
        self.assertEqual(
            [event[0] for event in self.events], ["peft_config", "base", "validate", "adapter", "freeze"]
        )
        self.assertEqual(self.events[0][1:], (str(self.adapter_path.resolve()), {"local_files_only": True}))
        self.assertEqual(self.events[1][1]["pretrained_name_or_path"], str(self.base_path.resolve()))
        self.assertTrue(self.events[1][1]["local_files_only"])
        self.assertIs(self.events[1][1]["config"], self.config)
        self.assertTrue(self.events[3][2]["local_files_only"])
        self.assertTrue(self.events[3][2]["is_trainable"])

    def test_missing_adapter_does_not_try_hub_loading(self):
        self.config.pretrained_path = self.work / "missing-adapter"
        with self.assertRaisesRegex(FileNotFoundError, "local checkpoint directory"):
            self.load()
        self.assertEqual(self.events, [])

    def test_missing_base_does_not_load_or_attach_a_model(self):
        self.adapter_config.base_model_name_or_path = str(self.work / "missing-base")
        with self.assertRaisesRegex(FileNotFoundError, "original local base checkpoint"):
            self.load()
        self.assertEqual([event[0] for event in self.events], ["peft_config"])

    def test_incompatible_lora_adapter_is_rejected_before_attachment(self):
        self.validation_error = (
            "PI05 LoRA cannot use modules_to_save/full_training_modules or learned prompts"
        )
        self.adapter_config.modules_to_save = ["model.prompt_tokens"]
        with self.assertRaisesRegex(ValueError, "cannot use modules_to_save"):
            self.load()
        self.assertEqual([event[0] for event in self.events], ["peft_config", "base", "validate"])

    def test_prompt_mode_rejects_baseline_lora_even_with_zero_prompt_counts(self):
        self.config.adaptation_mode = "prompt"
        with self.assertRaisesRegex(ValueError, "requires adaptation_mode='lora'"):
            self.load()
        self.assertEqual([event[0] for event in self.events], ["peft_config"])

    def test_enabled_prompt_bank_guard_is_preserved(self):
        self.config.adaptation_mode = "prompt"
        self.config.num_prompt_tokens = self.config.num_vlm_prompt_tokens = 16
        self.adapter_config.target_modules = "dummy-target-modules"
        with self.assertRaisesRegex(ValueError, "lacks saved learned prompt tokens"):
            self.load()
        self.assertEqual([event[0] for event in self.events], ["peft_config"])

    def test_prompt_adapter_loading_retains_existing_behavior(self):
        self.config.adaptation_mode = "prompt"
        self.config.num_prompt_tokens = self.config.num_vlm_prompt_tokens = 16
        self.adapter_config.target_modules = "dummy-target-modules"
        self.adapter_config.modules_to_save = ["model.vlm_prompt_tokens", "model.prompt_tokens"]
        self.adapter_config.base_model_name_or_path = "existing/base-repository"
        self.assertIs(self.load(), self.wrapper)
        self.assertEqual([event[0] for event in self.events], ["peft_config", "base", "adapter", "freeze"])
        self.assertEqual(self.events[0][2], {})
        self.assertNotIn("local_files_only", self.events[1][1])
        self.assertNotIn("local_files_only", self.events[2][2])


class PeftCheckpointSaveTest(unittest.TestCase):
    def setUp(self):
        self.training_state_save = Mock()
        self.save_checkpoint = load_function(
            (ROOT / "src/lerobot/common/train_utils.py").read_text(),
            "save_checkpoint",
            {
                "PRETRAINED_MODEL_DIR": "pretrained_model",
                "save_training_state": self.training_state_save,
            },
        )

    def save_stub_checkpoint(self, *, peft_config=None, use_peft=None):
        # There are no real files, model weights, optimizers, or training loops.
        # Omitting use_peft models a reward configuration that lacks that field.
        policy_config = SimpleNamespace(save_pretrained=Mock())
        if use_peft is not None:
            policy_config.use_peft = use_peft
        policy = SimpleNamespace(config=policy_config, save_pretrained=Mock())
        config = SimpleNamespace(
            peft=peft_config,
            policy=policy_config if use_peft is not None else None,
            save_pretrained=Mock(),
        )
        preprocessor = SimpleNamespace(save_pretrained=Mock())
        postprocessor = SimpleNamespace(save_pretrained=Mock())
        checkpoint = Path("checkpoints/004000")
        pretrained = checkpoint / "pretrained_model"
        model_state = {"example": "model tensor"}
        optim_state = {"example": "optimizer tensor"}
        optimizer = object()
        scheduler = object()
        self.save_checkpoint(
            checkpoint, 4000, config, policy, optimizer, scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            num_processes=1,
            batch_size=32,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )
        policy.save_pretrained.assert_called_once_with(pretrained, state_dict=model_state)
        config.save_pretrained.assert_called_once_with(pretrained)
        preprocessor.save_pretrained.assert_called_once_with(pretrained)
        postprocessor.save_pretrained.assert_called_once_with(pretrained)
        self.training_state_save.assert_called_once_with(
            checkpoint, 4000, optimizer, scheduler,
            num_processes=1,
            batch_size=32,
            optim_state_dict=optim_state,
        )
        return policy_config.save_pretrained, pretrained

    def test_new_peft_settings_save_policy_config(self):
        save, pretrained = self.save_stub_checkpoint(peft_config=SimpleNamespace(), use_peft=False)
        save.assert_called_once_with(pretrained)

    def test_reloaded_adapter_saves_policy_config_without_new_peft_settings(self):
        save, pretrained = self.save_stub_checkpoint(peft_config=None, use_peft=True)
        save.assert_called_once_with(pretrained)

    def test_ordinary_policy_does_not_add_an_extra_config_save(self):
        save, _ = self.save_stub_checkpoint(peft_config=None, use_peft=False)
        save.assert_not_called()

    def test_reward_config_without_peft_marker_preserves_existing_behavior(self):
        save, _ = self.save_stub_checkpoint(peft_config=None, use_peft=None)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
