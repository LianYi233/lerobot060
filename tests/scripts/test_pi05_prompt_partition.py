"""Run production freeze/optimizer-selection methods without ML dependencies.

Parameter stand-ins expose requires_grad/numel/grad; they do not simulate autograd,
attention, checkpoint loading or a GPU training step. Real tensor tests also live
in tests/policies/pi0_pi05/test_pi05_prompt.py.
"""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[2] / "src/lerobot/policies/pi05/modeling_pi05.py"


def load_production_methods():
    tree = ast.parse(SOURCE.read_text())
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    methods = {
        "PI05Pytorch": {"_freeze_vlm", "_freeze_backbones", "trainable_expert_blocks"},
        "PI05Policy": {"_is_cabo_active", "get_optim_params"},
    }
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("GemmaConfig", "get_gemma_config"):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = []
            node.keywords = []
            node.decorator_list = []
            node.body = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in methods[node.name]
            ]
            selected.append(node)
    namespace = {"PI05TrainingStage": SimpleNamespace(NEXT_ACTION="next_action")}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class Parameter:
    def __init__(self, *shape):
        self.shape = shape
        self.requires_grad = True
        self.grad = object()

    def numel(self):
        return math.prod(self.shape)

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class Module:
    def __init__(self, *shape):
        self.weight = Parameter(*shape)
        self.training = True

    def parameters(self):
        return iter([self.weight])

    def eval(self):
        self.training = False
        return self


class PromptPartitionTest(unittest.TestCase):
    def test_16_32_64_banks_are_the_only_parameters_in_optimizer(self):
        production = load_production_methods()
        for tokens, expected in ((16, 49152), (32, 98304), (64, 196608)):
            with self.subTest(tokens=tokens):
                model = production["PI05Pytorch"]()
                model.config = SimpleNamespace(
                    train_action_projections=False,
                    train_action_expert_last_n_layers=0,
                    training_stage="flow",
                    num_prompt_tokens=tokens,
                    cabo_enabled=False,
                )
                model.vlm_prompt_tokens = Module(tokens, production["get_gemma_config"]("gemma_2b").width)
                model.prompt_tokens = Module(tokens, production["get_gemma_config"]("gemma_300m").width)
                model.action_in_proj, model.action_out_proj = Module(32, 1024), Module(1024, 32)
                model.time_mlp_in, model.time_mlp_out = Module(1024, 1024), Module(1024, 1024)
                model.paligemma_with_expert = SimpleNamespace(paligemma=Module(10), gemma_expert=Module(20))
                modules = [
                    model.vlm_prompt_tokens,
                    model.prompt_tokens,
                    model.action_in_proj,
                    model.action_out_proj,
                    model.time_mlp_in,
                    model.time_mlp_out,
                    model.paligemma_with_expert.paligemma,
                    model.paligemma_with_expert.gemma_expert,
                ]
                parameters = [module.weight for module in modules]
                model.parameters = lambda: iter(parameters)
                policy = production["PI05Policy"]()
                policy.model, policy.config, policy.parameters = model, model.config, model.parameters
                for _ in range(2):
                    # An external trainer accidentally enabling everything must not leak backbone parameters.
                    for parameter in parameters:
                        parameter.requires_grad_(True)
                        parameter.grad = object()
                    optimizer_parameters = policy.get_optim_params()
                    self.assertEqual(optimizer_parameters, parameters[:2])
                    self.assertEqual(sum(p.numel() for p in optimizer_parameters), expected)
                    self.assertTrue(all(p.requires_grad for p in parameters[:2]))
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in parameters[2:]))
                    self.assertFalse(model.paligemma_with_expert.paligemma.training)
                    self.assertFalse(model.paligemma_with_expert.gemma_expert.training)


if __name__ == "__main__":
    unittest.main()
