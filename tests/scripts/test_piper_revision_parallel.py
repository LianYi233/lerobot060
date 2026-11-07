"""CPU-only scheduling tests: real threads, mocked nvidia-smi, no robot or GPU."""

import importlib.util
import subprocess
import threading
import time
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "piper_revision_parallel", ROOT / "examples/training/piper_revision_parallel.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def snapshot(*, free=64, total=80, pids=()):
    return {"total_gib": total, "free_gib": free, "utilization_percent": 50, "pids": list(pids)}


def completed(stdout="", *, stderr="", returncode=0):
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, stderr)


class FakeClock:
    """Advance the admission clock while yielding briefly to actual workers."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.001)


class GpuSnapshotTests(unittest.TestCase):
    def test_selected_gpu_mib_conversion_and_distinct_processes(self):
        with patch.object(MODULE.subprocess, "run", side_effect=[
            completed("81920, 40960, 87\n"), completed("202\n101\n202\n"),
        ]) as run:
            result = MODULE.gpu_snapshot("1")
        self.assertEqual(result, {
            "total_gib": 80, "free_gib": 40, "utilization_percent": 87, "pids": [101, 202],
        })
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands, [
            ["nvidia-smi", "-i", "1", "--query-gpu=memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            ["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        ])
        self.assertTrue(all(call.kwargs["timeout"] == 10 for call in run.call_args_list))

    def test_empty_compute_query_is_idle(self):
        with patch.object(MODULE.subprocess, "run", side_effect=[completed("81920, 81920, 0"), completed()]):
            self.assertEqual(MODULE.gpu_snapshot("0")["pids"], [])

    def test_unknown_memory_or_utilization_is_never_treated_as_free(self):
        for raw in ("", "N/A, 40000, 10", "81920, nan, 10", "81920, 100000, 10",
                    "81920, 40000, N/A", "81920, 40000, 101", "0, 0, 0",
                    "81920, 40000", "81920, 40000, 10\n81920, 40000, 10"):
            with self.subTest(raw=raw), patch.object(MODULE.subprocess, "run", return_value=completed(raw)):
                with self.assertRaisesRegex(RuntimeError, "Unknown GPU.*refusing to launch"):
                    MODULE.gpu_snapshot("0")

    def test_unknown_compute_query_is_never_treated_as_idle(self):
        for raw in ("N/A", "[Not Supported]", "-1", "0", "abc", "101, 202"):
            with self.subTest(raw=raw), patch.object(MODULE.subprocess, "run", side_effect=[
                completed("81920, 40000, 10"), completed(raw),
            ]):
                with self.assertRaisesRegex(RuntimeError, "Unknown GPU.*compute PID"):
                    MODULE.gpu_snapshot("0")

    def test_failed_missing_or_timed_out_nvidia_smi_refuses_launch(self):
        with patch.object(MODULE.subprocess, "run", return_value=completed(returncode=1, stderr="GPU not found")):
            with self.assertRaisesRegex(RuntimeError, "GPU not found"):
                MODULE.gpu_snapshot("0")
        for error in (FileNotFoundError("missing nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10)):
            with self.subTest(error=type(error)), patch.object(MODULE.subprocess, "run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "Cannot inspect GPU.*refusing to launch"):
                    MODULE.gpu_snapshot("0")


class ParallelWaveTests(unittest.TestCase):
    def run_jobs(self, jobs, callback, **overrides):
        options = dict(
            gpu="0", stagger_seconds=0, min_free_gib=30, poll_seconds=0.001,
            check_wave_space=lambda wave: None, first_ready=lambda job: True, report=lambda line: None,
        )
        options.update(overrides)
        return MODULE.run_in_waves(jobs, callback, **options)

    def test_no_jobs_does_not_query_gpu(self):
        with patch.object(MODULE, "gpu_snapshot") as query:
            self.assertEqual(self.run_jobs([], Mock()), [])
        query.assert_not_called()

    def test_single_remaining_job_has_no_stagger_or_readiness_wait(self):
        disk_check, ready = Mock(), Mock(side_effect=AssertionError("no second job"))
        with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()) as query:
            result = self.run_jobs(["4"], lambda job, index, total: (job, index, total),
                                   stagger_seconds=180, check_wave_space=disk_check, first_ready=ready)
        self.assertEqual(result, [("4", 1, 1)])
        disk_check.assert_called_once_with(["4"])
        query.assert_called_once_with("0")
        ready.assert_not_called()

    def test_initial_busy_or_insufficient_gpu_starts_nothing(self):
        cases = (
            (snapshot(pids=[123]), 30, RuntimeError, "existing training.*cannot be hot-switched"),
            (snapshot(free=20), 30, RuntimeError, "before starting the first task"),
            (snapshot(), 100, ValueError, "threshold can never be satisfied"),
        )
        for state, threshold, error, pattern in cases:
            callback, disk_check = Mock(), Mock()
            with self.subTest(state=state, threshold=threshold), patch.object(MODULE, "gpu_snapshot", return_value=state):
                with self.assertRaisesRegex(error, pattern):
                    self.run_jobs(["2", "3"], callback, min_free_gib=threshold, check_wave_space=disk_check)
            callback.assert_not_called()
            disk_check.assert_not_called()

    def test_threads_overlap_and_next_wave_waits_for_full_callbacks(self):
        jobs = [{"task_id": task} for task in ("2", "3", "4", "1")]
        lock = threading.Lock()
        barriers = [threading.Barrier(2), threading.Barrier(2)]
        first_postprocess = threading.Event()
        second_done = threading.Event()
        release_first = threading.Event()
        third_started = threading.Event()
        active, maximum = 0, 0
        finished, calls, spaces, premature = set(), [], [], []

        def release_postprocess():
            ready = first_postprocess.wait(2) and second_done.wait(2)
            premature.append(not ready or third_started.wait(0.01))
            release_first.set()

        observer = threading.Thread(target=release_postprocess)
        observer.start()

        def callback(job, index, total):
            nonlocal active, maximum
            with lock:
                if index >= 3:
                    self.assertTrue({"2", "3"} <= finished)
                    third_started.set()
                active += 1
                maximum = max(maximum, active)
                calls.append((job["task_id"], index, total))
            barriers[(index - 1) // 2].wait(timeout=2)
            if index == 1:
                first_postprocess.set()
                self.assertTrue(release_first.wait(2))
            with lock:
                finished.add(job["task_id"])
                active -= 1
            if index == 2:
                second_done.set()
            return job["task_id"]

        try:
            with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
                result = self.run_jobs(jobs, callback, check_wave_space=lambda wave: spaces.append(
                    [job["task_id"] for job in wave]
                ))
        finally:
            release_first.set()
            observer.join(timeout=2)
        self.assertFalse(observer.is_alive())
        self.assertEqual(premature, [False])
        self.assertEqual(maximum, 2)
        self.assertEqual(result, ["2", "3", "4", "1"])
        self.assertEqual(sorted(calls, key=lambda item: item[1]), [("2", 1, 4), ("3", 2, 4), ("4", 3, 4), ("1", 4, 4)])
        self.assertEqual(spaces, [["2", "3"], ["4", "1"]])

    def test_each_second_launch_gate_and_thirty_second_wait_messages(self):
        for gate, threshold in (("stagger", 180), ("ready", 20), ("memory", 25), ("pids", 30)):
            with self.subTest(gate=gate):
                clock = FakeClock()
                peer_started = threading.Event()
                first_started = threading.Event()
                messages, starts = [], []
                query_count = 0

                def query(gpu):
                    nonlocal query_count
                    query_count += 1
                    if query_count == 1:
                        return snapshot()
                    if gate == "memory" and clock.now < threshold:
                        return snapshot(free=20, pids=[10])
                    if gate == "pids" and clock.now < threshold:
                        return snapshot(pids=[10, 20])
                    return snapshot(pids=[10])

                def callback(job, index, total):
                    if index == 1:
                        first_started.set()
                        self.assertTrue(peer_started.wait(2))
                    else:
                        starts.append(clock.now)
                        peer_started.set()
                    return job

                def ready(job):
                    return first_started.is_set() and (gate != "ready" or clock.now >= threshold)

                with patch.object(MODULE, "time", clock), patch.object(MODULE, "gpu_snapshot", side_effect=query):
                    result = self.run_jobs(
                        ["task2", "task3"], callback, first_ready=ready, poll_seconds=5,
                        stagger_seconds=threshold if gate == "stagger" else 0,
                        report=lambda line: messages.append((clock.now, line)),
                    )
                self.assertEqual(result, ["task2", "task3"])
                self.assertGreaterEqual(starts[0], threshold)
                waits = [(stamp, line) for stamp, line in messages if "PARALLEL WAIT" in line]
                self.assertTrue(waits)
                self.assertTrue(all("task2" in line and "task3" in line for _, line in waits))
                if gate == "stagger":
                    self.assertGreaterEqual(len(waits), 6)
                    self.assertTrue(all(b[0] - a[0] >= 30 for a, b in zip(waits, waits[1:])))

    def test_successfully_completed_first_job_does_not_need_ready_signal(self):
        with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
            results = self.run_jobs(["2", "3"], lambda job, index, total: job, first_ready=lambda job: False)
        self.assertEqual(results, ["2", "3"])

    def test_first_failure_before_admission_starts_no_peer_or_later_wave(self):
        started, spaces = [], []

        def callback(job, index, total):
            started.append(job)
            raise RuntimeError("first training failed")

        with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
            with self.assertRaisesRegex(RuntimeError, "first training failed"):
                self.run_jobs(["2", "3", "4", "1"], callback, first_ready=lambda job: False,
                              check_wave_space=lambda wave: spaces.append(list(wave)))
        self.assertEqual(started, ["2"])
        self.assertEqual(spaces, [["2", "3"]])

    def test_failed_launched_peer_waits_for_healthy_peer_and_no_next_wave(self):
        for failing_index in (1, 2):
            with self.subTest(failing_index=failing_index):
                barrier = threading.Barrier(2)
                cancel_event = threading.Event()
                started, completed_peers, healthy_cancelled = [], [], []

                def callback(job, index, total):
                    started.append(job)
                    barrier.wait(timeout=2)
                    if index == failing_index:
                        raise RuntimeError("one launched task failed")
                    time.sleep(0.005)
                    healthy_cancelled.append(cancel_event.is_set())
                    completed_peers.append(job)

                with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
                    with self.assertRaisesRegex(RuntimeError, "one launched task failed"):
                        self.run_jobs(["2", "3", "4", "1"], callback, cancel_event=cancel_event)
                self.assertCountEqual(started, ["2", "3"])
                self.assertEqual(completed_peers, ["3" if failing_index == 1 else "2"])
                self.assertEqual(healthy_cancelled, [False])

    def test_each_wave_rechecks_external_gpu_use_before_start(self):
        finished = set()
        started = []

        def callback(job, index, total):
            started.append(job)
            finished.add(job)

        def disk_check(wave):
            return None

        def query(gpu):
            return snapshot(pids=[99]) if finished == {"2", "3"} else snapshot()

        with patch.object(MODULE, "gpu_snapshot", side_effect=query):
            with self.assertRaisesRegex(RuntimeError, "existing training"):
                self.run_jobs(["2", "3", "4", "1"], callback, check_wave_space=disk_check)
        self.assertCountEqual(started, ["2", "3"])

    def test_disk_failure_before_a_wave_starts_none_of_that_wave(self):
        started = []

        def disk_check(wave):
            if "4" in wave:
                raise RuntimeError("insufficient combined checkpoint space")

        with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
            with self.assertRaisesRegex(RuntimeError, "combined checkpoint space"):
                self.run_jobs(["2", "3", "4", "1"], lambda job, index, total: started.append(job),
                              check_wave_space=disk_check)
        self.assertCountEqual(started, ["2", "3"])

    def test_ctrl_c_sets_event_before_waiting_for_worker_cleanup(self):
        cancel_event = threading.Event()
        started = threading.Event()
        cleaned_up = threading.Event()

        def callback(job, index, total):
            started.set()
            self.assertTrue(cancel_event.wait(2))
            cleaned_up.set()

        def ready(job):
            self.assertTrue(started.wait(2))
            raise KeyboardInterrupt

        with patch.object(MODULE, "gpu_snapshot", return_value=snapshot()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_jobs(["2", "3"], callback, first_ready=ready, cancel_event=cancel_event)
        self.assertTrue(cancel_event.is_set())
        self.assertTrue(cleaned_up.is_set())

    def test_failed_resource_query_cancels_owned_worker_without_launching_peer(self):
        cancel_event = threading.Event()
        started = threading.Event()
        cleaned_up = threading.Event()
        calls = []

        def callback(job, index, total):
            calls.append(job)
            started.set()
            self.assertTrue(cancel_event.wait(2))
            cleaned_up.set()

        with patch.object(MODULE, "gpu_snapshot", side_effect=[snapshot(), RuntimeError("nvidia-smi failed")]):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi failed"):
                self.run_jobs(["2", "3"], callback, first_ready=lambda job: started.wait(2), cancel_event=cancel_event)
        self.assertEqual(calls, ["2"])
        self.assertTrue(cleaned_up.is_set())

    def test_already_cancelled_queue_never_queries_gpu(self):
        cancel_event = threading.Event()
        cancel_event.set()
        with patch.object(MODULE, "gpu_snapshot") as query:
            with self.assertRaises(CancelledError):
                self.run_jobs(["2"], Mock(), cancel_event=cancel_event)
        query.assert_not_called()

    def test_invalid_thresholds_refuse_before_gpu_queries(self):
        for options in ({"stagger_seconds": -1}, {"stagger_seconds": float("nan")},
                        {"min_free_gib": 0}, {"min_free_gib": float("inf")}, {"poll_seconds": 0}):
            with self.subTest(options=options), patch.object(MODULE, "gpu_snapshot") as query:
                with self.assertRaises(ValueError):
                    self.run_jobs(["2"], Mock(), **options)
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
