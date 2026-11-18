"""Checkpoint metadata checks that run without loading PyTorch or a PI05 model."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_module(name, filename):
    # Import the actual lightweight modules without utils/__init__.py, which imports torch.
    path = Path(__file__).resolve().parents[2] / "src/lerobot/utils" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_protocol = _load_module("ntk_control_test_protocol", "ntk_protocol.py")
with patch.dict(sys.modules, {"lerobot.utils.ntk_protocol": _protocol}):
    _control = _load_module("ntk_control_test_module", "ntk_control.py")
PROMPT_ONLY_STEPS = _control.PROMPT_ONLY_STEPS
resolve_prompt_only_stages = _control.resolve_prompt_only_stages


class PromptOnlyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "control"
        for step in PROMPT_ONLY_STEPS:
            path = self.run / f"checkpoints/{step:06d}/pretrained_model"
            path.mkdir(parents=True)
            config = {
                "training_stage": "flow",
                "next_action_pretrain_steps": 0,
                "next_action_bridge_steps": 0,
                "num_vlm_prompt_tokens": 16,
                "num_prompt_tokens": 16,
                "cabo_enabled": False,
                "cabo_prompt_update_ratio": 2.0,
                "freeze_vision_encoder": True,
                "train_expert_only": True,
            }
            (path / "config.json").write_text(json.dumps(config))
            (path / "train_config.json").write_text(json.dumps({"steps": 4000, "seed": 0}))
            (path / "policy_preprocessor.json").write_text("{}")
            (path / "model.safetensors").touch()
            state = path.parent / "training_state"
            state.mkdir()
            (state / "training_step.json").write_text(json.dumps({"step": step}))

    def args(self, through_step=4000):
        return SimpleNamespace(run_dir=self.run, through_step=through_step)

    def change_json(self, step, filename, updates):
        path = self.run / f"checkpoints/{step:06d}/pretrained_model/{filename}"
        content = json.loads(path.read_text())
        content.update(updates)
        path.write_text(json.dumps(content))

    def test_single_run_uses_matching_local_and_cumulative_updates(self):
        stages = resolve_prompt_only_stages(self.args())
        self.assertEqual([stage["name"] for stage in stages], ["before", "step_750", "step_1000", "final"])
        self.assertEqual([stage["total_step"] for stage in stages], list(PROMPT_ONLY_STEPS))
        self.assertEqual([stage["local_step"] for stage in stages], list(PROMPT_ONLY_STEPS))

    def test_partial_run_does_not_require_later_checkpoints(self):
        (self.run / "checkpoints/004000/pretrained_model/model.safetensors").unlink()
        for through_step in PROMPT_ONLY_STEPS[:-1]:
            with self.subTest(through_step=through_step):
                stages = resolve_prompt_only_stages(self.args(through_step))
                self.assertEqual(stages[-1]["total_step"], through_step)
        with self.assertRaisesRegex(FileNotFoundError, "final: missing"):
            resolve_prompt_only_stages(self.args())

    def test_missing_intermediate_is_not_substituted(self):
        (self.run / "checkpoints/000750/pretrained_model/model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "step_750: missing"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_staged_checkpoint(self):
        self.change_json(750, "config.json", {"training_stage": "next_action"})
        with self.assertRaisesRegex(ValueError, "requires a flow checkpoint"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_wrong_completed_step(self):
        state = self.run / "checkpoints/000750/training_state/training_step.json"
        state.write_text('{"step": 751}')
        with self.assertRaisesRegex(ValueError, "751 local updates, expected 750"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_inconsistent_cpc_setting(self):
        self.change_json(1000, "config.json", {"cabo_enabled": True})
        with self.assertRaisesRegex(ValueError, "cabo_enabled"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_inconsistent_training_seed(self):
        self.change_json(1000, "train_config.json", {"seed": 1})
        with self.assertRaisesRegex(ValueError, "same training seed"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_wrong_training_budget(self):
        self.change_json(4000, "train_config.json", {"steps": 3000})
        with self.assertRaisesRegex(ValueError, "steps=4000"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_changed_prompt_capacity(self):
        self.change_json(750, "config.json", {"num_prompt_tokens": 8})
        with self.assertRaisesRegex(ValueError, "num_prompt_tokens"):
            resolve_prompt_only_stages(self.args())

    def test_rejects_aliased_model_files(self):
        first = self.run / "checkpoints/000000/pretrained_model/model.safetensors"
        second = self.run / "checkpoints/000750/pretrained_model/model.safetensors"
        second.unlink()
        second.symlink_to(first)
        with self.assertRaisesRegex(ValueError, "distinct model files"):
            resolve_prompt_only_stages(self.args())


if __name__ == "__main__":
    unittest.main()
