"""Exercise the real training entry point with delayed ranks, without ML dependencies.

Only the outer train() function is compiled from source. Dataset/model/W&B work is
replaced by a filesystem writer and Accelerator's barrier by threading.Barrier.
This checks startup ordering and overwrite protection, not NCCL/GPU execution.
"""

import ast
import concurrent.futures
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_entry_point(namespace):
    path = ROOT / "src/lerobot/scripts/lerobot_train.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    function.decorator_list = []
    # CLI parsing and heavyweight modules are outside this startup-order regression.
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["train"]


class TrainingStartupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.events = []
        self.lock = threading.Lock()
        import_utils = types.ModuleType("lerobot.utils.import_utils")
        import_utils.require_package = lambda *args, **kwargs: None
        accelerate_utils = types.ModuleType("accelerate.utils")
        accelerate_utils.DistributedType = types.SimpleNamespace(
            NO="NO", MULTI_GPU="MULTI_GPU", MULTI_CPU="MULTI_CPU", FSDP="FSDP"
        )
        self.modules = patch.dict(sys.modules, {
            "lerobot.utils.import_utils": import_utils,
            "accelerate.utils": accelerate_utils,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def record(self, event):
        with self.lock:
            self.events.append(event)

    def config(self, rank, release=None, resume=False):
        def validate():
            if release is not None and not release.wait(timeout=5):
                raise TimeoutError("The fast rank never reached the startup boundary")
            if self.output.exists() and not resume:
                raise FileExistsError(f"Output directory {self.output} already exists and resume is False")
            self.record(("validated", rank))

        return types.SimpleNamespace(
            job=types.SimpleNamespace(is_remote=False),
            piper_eval=types.SimpleNamespace(enabled=False),
            cabo_active=False,
            rank=rank,
            validate=validate,
        )

    def entry_point(self, barrier, release, priming=False):
        owner = self

        class Accelerator:
            def __init__(self, rank):
                self.rank = rank

            def wait_for_everyone(self):
                # Permit the delayed rank to validate, then wait before creating any files.
                release.set()
                barrier.wait(timeout=5)

            def end_training(self):
                owner.record(("ended", self.rank))

        def write_output(cfg, accelerator, stage="flow"):
            if cfg.rank == 0:
                self.output.mkdir(exist_ok=True)
                self.record((stage, cfg.rank))
                # Without the production barrier this releases rank 1 AFTER mkdir,
                # deterministically reproducing the reported FileExistsError.
                release.set()

        namespace = {
            "_make_training_accelerator": lambda cfg, existing: Accelerator(cfg.rank),
            "_pi05_next_action_pretraining_active": lambda cfg: priming,
            "_run_pi05_next_action_pretraining": lambda cfg, accel: write_output(cfg, accel, "priming"),
            "_train_single_stage": write_output,
        }
        return load_entry_point(namespace)

    def test_delayed_rank_validates_before_any_stage_creates_directory(self):
        for priming in (False, True):
            with self.subTest(priming=priming):
                if self.output.exists():
                    self.output.rmdir()
                self.events.clear()
                release = threading.Event()
                entry = self.entry_point(threading.Barrier(2), release, priming=priming)
                configs = [self.config(0), self.config(1, release)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(entry, cfg) for cfg in configs]
                    for future in futures:
                        future.result(timeout=10)
                for stage in (("priming", "flow") if priming else ("flow",)):
                    for rank in (0, 1):
                        self.assertLess(self.events.index(("validated", rank)), self.events.index((stage, 0)))
                self.assertIn(("ended", 0), self.events)
                self.assertIn(("ended", 1), self.events)

    def test_preexisting_directory_is_still_rejected_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / "existing-checkpoint.txt"
        marker.write_text("keep")
        entry = self.entry_point(threading.Barrier(1), threading.Event())
        with self.assertRaises(FileExistsError):
            entry(self.config(0))
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.events, [])

    def test_single_rank_and_explicit_resume_can_reach_training(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                self.events.clear()
                entry = self.entry_point(threading.Barrier(1), threading.Event())
                entry(self.config(0, resume=resume))
                self.assertEqual(self.events, [("validated", 0), ("flow", 0), ("ended", 0)])


if __name__ == "__main__":
    unittest.main()
