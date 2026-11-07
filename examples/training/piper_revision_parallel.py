"""Read-only GPU checks and two-job waves for the Piper revision launcher.

This module neither changes the training recipe nor manages checkpoints. The
caller owns the output-group lock and each synchronous ``run_job`` callback must
include training, audit and compact export. For cancellation, pass the same Event
that the callback observes when cleaning up its own training subprocess.
"""

import csv
import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from typing import TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")


def _query_gpu(gpu: str, query: str) -> str:
    command = ["nvidia-smi", "-i", gpu, query, "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot inspect GPU {gpu} with nvidia-smi; refusing to launch: {error}") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Cannot inspect GPU {gpu} with nvidia-smi; refusing to launch: {detail}")
    return result.stdout.strip()


def gpu_snapshot(gpu: str) -> dict:
    """Return GiB memory, utilization_percent and distinct compute PIDs.

    nvidia-smi's memory values are MiB. Missing/unsupported/malformed observations
    are errors, not evidence that the GPU is free. No GPU settings are changed.
    """
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("gpu must be one explicit nvidia-smi device identifier")
    raw = _query_gpu(gpu, "--query-gpu=memory.total,memory.free,utilization.gpu")
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    try:
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("expected exactly one GPU and three metrics")
        total, free, utilization = (float(value) for value in rows[0])
        if not all(math.isfinite(value) for value in (total, free, utilization)):
            raise ValueError("non-finite or unsupported GPU metrics")
        if total <= 0 or not 0 <= free <= total or not 0 <= utilization <= 100:
            raise ValueError("out-of-range GPU metrics")
    except ValueError as error:
        raise RuntimeError(f"Unknown GPU {gpu} memory/utilization: {raw!r}; refusing to launch") from error

    raw_pids = _query_gpu(gpu, "--query-compute-apps=pid")
    pids = set()
    for line in raw_pids.splitlines():
        value = line.strip()
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise RuntimeError(f"Unknown GPU {gpu} compute PID: {line!r}; refusing to launch")
        pids.add(int(value))
    return {
        "total_gib": total / 1024,
        "free_gib": free / 1024,
        "utilization_percent": utilization,
        "pids": sorted(pids),
    }


def _job_name(job) -> str:
    if isinstance(job, Mapping):
        if job.get("run_name"):
            return str(job["run_name"])
        if "task_id" in job:
            return f"task{job['task_id']}"
    return str(job)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Piper parallel queue cancellation requested")


def run_in_waves(
    jobs: Iterable[Job],
    run_job: Callable[[Job, int, int], Result],
    *,
    gpu: str,
    stagger_seconds: float = 180,
    min_free_gib: float = 30.0,
    check_wave_space: Callable[[list[Job]], object],
    first_ready: Callable[[Job], bool],
    poll_seconds: float = 5,
    report: Callable[[str], object] = print,
    cancel_event: threading.Event | None = None,
) -> list[Result]:
    """Run adjacent pairs, preserving input order and completing each whole wave.

    ``run_job(job, index, total)`` receives a one-based index. The second job waits
    for the stagger interval, first_ready (or successful completion of the first
    job), free memory, and fewer than two GPU compute PIDs. Normal job failure
    does not interrupt an already-started peer; it prevents all later waves.
    Ctrl+C/monitor failures set cancel_event *before* waiting for worker cleanup.
    """
    for name, value, allow_zero in (
        ("stagger_seconds", stagger_seconds, True),
        ("min_free_gib", min_free_gib, False),
        ("poll_seconds", poll_seconds, False),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    jobs = list(jobs)
    if not jobs:
        return []

    executor = None
    active = []
    results = []
    try:
        _check_cancel(cancel_event)
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="piper-task")
        for start in range(0, len(jobs), 2):
            _check_cancel(cancel_event)
            initial = gpu_snapshot(gpu)
            if initial["total_gib"] < min_free_gib:
                raise ValueError(
                    f"GPU {gpu} total memory {initial['total_gib']:.1f} GiB is smaller than "
                    f"min_free_gib={min_free_gib:.1f}; this threshold can never be satisfied"
                )
            if initial["pids"]:
                raise RuntimeError(
                    f"GPU {gpu} has existing training/compute processes {initial['pids']}; "
                    "the old sequential queue cannot be hot-switched. Use this queue after completion; "
                    "no existing process was stopped."
                )
            if initial["free_gib"] < min_free_gib:
                raise RuntimeError(
                    f"GPU {gpu} has only {initial['free_gib']:.1f} GiB free; at least "
                    f"{min_free_gib:.1f} GiB is required before starting the first task of a wave"
                )
            wave = jobs[start:start + 2]
            check_wave_space(wave)
            _check_cancel(cancel_event)
            report(f"[PARALLEL WAVE {start // 2 + 1}] " + " + ".join(_job_name(job) for job in wave))
            first = executor.submit(run_job, wave[0], start + 1, len(jobs))
            active = [first]
            submitted = time.monotonic()
            last_report = None

            if len(wave) == 2:
                while True:
                    _check_cancel(cancel_event)
                    if first.done():
                        first.result()  # Do not dispatch a peer after known failure.
                        ready = True
                    else:
                        ready = bool(first_ready(wave[0]))
                    snapshot = gpu_snapshot(gpu)
                    now = time.monotonic()
                    elapsed = now - submitted
                    reasons = []
                    if elapsed < stagger_seconds:
                        reasons.append(f"stagger remaining {stagger_seconds - elapsed:.0f} s")
                    if not ready:
                        reasons.append("first task adaptation/evaluation not ready")
                    if snapshot["free_gib"] < min_free_gib:
                        reasons.append(f"free GPU memory below {min_free_gib:.1f} GiB")
                    if len(snapshot["pids"]) >= 2:
                        reasons.append(f"GPU already has {len(snapshot['pids'])} compute processes")
                    resource = f"GPU {gpu} free={snapshot['free_gib']:.1f} GiB; elapsed={elapsed:.0f} s"
                    if not reasons:
                        _check_cancel(cancel_event)
                        if first.done():
                            first.result()
                        report(f"[PARALLEL START] {_job_name(wave[1])} after {_job_name(wave[0])}: {resource}")
                        active.append(executor.submit(run_job, wave[1], start + 2, len(jobs)))
                        break
                    if last_report is None or now - last_report >= 30:
                        report(f"[PARALLEL WAIT] {_job_name(wave[1])} after {_job_name(wave[0])}: "
                               + "; ".join(reasons) + f"; {resource}")
                        last_report = now
                    time.sleep(poll_seconds)

            # Observe results only after both callbacks have finished, including
            # audit/compact. A failed job must not cancel a healthy running peer.
            while not all(future.done() for future in active):
                _check_cancel(cancel_event)
                wait(active, timeout=poll_seconds)
            _check_cancel(cancel_event)
            results.extend(future.result() for future in active)
        return results
    except BaseException:
        if cancel_event is not None:
            cancel_event.set()
        for future in active:
            future.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
