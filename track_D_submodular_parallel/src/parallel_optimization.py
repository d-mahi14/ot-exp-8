"""Parallel Program Optimization Module for Track D.

Implements:
1. Workload execution engines with:
   - Sequential execution baseline
   - Static chunk scheduling (ProcessPoolExecutor / ThreadPoolExecutor)
   - Dynamic work queue scheduling
   - Work-stealing scheduler simulation
2. Parallel Paradigms (Data Parallelism vs Pipeline Parallelism models)
3. Parallel performance metrics:
   - Execution time T(p)
   - Speedup S(p) = T(1) / T(p)
   - Efficiency E(p) = S(p) / p
   - Parallel Overhead T_overhead(p) = p * T(p) - T(1)
4. Theoretical scaling models:
   - Amdahl's Law curve fitting (fixed workload size)
   - Gustafson-Barsis' Law curve fitting (scaled workload size)
"""

from __future__ import annotations

import collections
import concurrent.futures
import math
import multiprocessing
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def synthetic_compute_kernel(task_args: Tuple[int, int, float]) -> float:
    """Compute-intensive kernel (deterministic numerical matrix operation releasing GIL).

    Arguments:
        task_id: unique task integer
        iterations: number of loops to execute
        seed_val: float input
    Returns:
        float computed scalar
    """
    task_id, iterations, seed_val = task_args
    rng = np.random.default_rng(int(task_id + 42))
    dim = 160
    mat = rng.standard_normal((dim, dim))
    loops = max(1, iterations // 3000)
    scale = 1.0 / np.sqrt(float(dim))
    for _ in range(loops):
        # Trigonometric projection keeps matrix elements strictly bounded in [-1, 1]
        mat = np.dot(np.sin(mat), np.cos(mat)) * scale + seed_val
    return float(mat[0, 0])


@dataclass
class ParallelBenchmarkResult:
    worker_count: int
    execution_time_sec: float
    speedup: float
    efficiency: float
    overhead_sec: float
    throughput_tasks_per_sec: float
    scheduler: str
    num_tasks: int


def run_sequential_workload(
    tasks: Sequence[Tuple[int, int, float]],
    kernel_fn: Callable[[Tuple[int, int, float]], float] = synthetic_compute_kernel,
) -> Tuple[List[float], float]:
    """Execute all tasks sequentially on a single thread/process."""
    t0 = time.perf_counter()
    results = [kernel_fn(t) for t in tasks]
    t1 = time.perf_counter()
    return results, t1 - t0


def run_static_parallel_workload(
    tasks: Sequence[Tuple[int, int, float]],
    num_workers: int,
    backend: str = "thread",
    kernel_fn: Callable[[Tuple[int, int, float]], float] = synthetic_compute_kernel,
) -> Tuple[List[float], float]:
    """Execute tasks partitioned evenly across num_workers."""
    t0 = time.perf_counter()
    chunksize = max(1, len(tasks) // (num_workers * 2))
    pool_cls = (
        concurrent.futures.ThreadPoolExecutor
        if backend == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )
    with pool_cls(max_workers=num_workers) as executor:
        results = list(executor.map(kernel_fn, tasks, chunksize=chunksize))
    t1 = time.perf_counter()
    return results, t1 - t0


def run_dynamic_parallel_workload(
    tasks: Sequence[Tuple[int, int, float]],
    num_workers: int,
    backend: str = "thread",
    kernel_fn: Callable[[Tuple[int, int, float]], float] = synthetic_compute_kernel,
) -> Tuple[List[float], float]:
    """Execute tasks dynamically: workers pull chunks from a central task queue."""
    t0 = time.perf_counter()
    pool_cls = (
        concurrent.futures.ThreadPoolExecutor
        if backend == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )
    with pool_cls(max_workers=num_workers) as executor:
        futures = [executor.submit(kernel_fn, t) for t in tasks]
        results = [f.result() for f in futures]
    t1 = time.perf_counter()
    return results, t1 - t0


class WorkStealingQueueSimulation:
    """Simulates multi-threaded / multi-worker work stealing on heterogeneous task loads.

    Each worker has a double-ended queue (deque). Workers pop tasks from the top (LIFO/FIFO).
    When a worker's deque is empty, it attempts to steal tasks from the bottom of a random victim's deque.
    """

    def __init__(self, num_workers: int, tasks: Sequence[Tuple[int, int, float]]) -> None:
        self.num_workers = num_workers
        self.deques: List[collections.deque] = [collections.deque() for _ in range(num_workers)]
        self.locks: List[threading.Lock] = [threading.Lock() for _ in range(num_workers)]
        self.results: Dict[int, float] = {}
        self.result_lock = threading.Lock()
        self.steal_counts = [0] * num_workers
        self.tasks_processed = [0] * num_workers

        # Initial round-robin distribution to worker deques
        for i, task in enumerate(tasks):
            w = i % num_workers
            self.deques[w].append(task)

    def _worker_loop(self, worker_id: int, total_tasks: int) -> None:
        while True:
            with self.result_lock:
                if len(self.results) >= total_tasks:
                    break

            task = None
            # Attempt local pop
            with self.locks[worker_id]:
                if self.deques[worker_id]:
                    task = self.deques[worker_id].popleft()

            # If local empty, attempt work stealing from another worker
            if task is None:
                for target_id in range(self.num_workers):
                    if target_id == worker_id:
                        continue
                    with self.locks[target_id]:
                        if len(self.deques[target_id]) > 1:
                            task = self.deques[target_id].pop()  # steal from tail
                            self.steal_counts[worker_id] += 1
                            break

            if task is not None:
                res = synthetic_compute_kernel(task)
                self.tasks_processed[worker_id] += 1
                with self.result_lock:
                    self.results[task[0]] = res
            else:
                time.sleep(0.0005)  # brief backoff before retrying steal

    def execute(self, total_tasks: int) -> float:
        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=self._worker_loop, args=(w, total_tasks))
            for w in range(self.num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t1 = time.perf_counter()
        return t1 - t0


def amdahl_speedup_formula(p: np.ndarray, s: float) -> np.ndarray:
    """Amdahl's Law: Speedup for fixed workload size.

    S(p) = 1 / ( (1 - s) + s / p )
    where s is the parallelizable fraction (0 <= s <= 1).
    (1 - s) is the serial bottleneck.
    """
    return 1.0 / ((1.0 - s) + (s / np.maximum(p, 1e-9)))


def gustafson_speedup_formula(p: np.ndarray, alpha: float) -> np.ndarray:
    """Gustafson's Law: Scaled speedup for workload that scales with processors.

    S(p) = p - alpha * (p - 1)
    where alpha is the serial fraction of the scaled problem.
    """
    return p - alpha * (p - 1.0)


def fit_amdahl_law(workers: Sequence[int], measured_speedups: Sequence[float]) -> Tuple[float, float]:
    """Fit Amdahl's Law to empirical speedup measurements.

    Returns:
        (parallel_fraction s, serial_fraction = 1 - s)
    """
    p_arr = np.array(workers, dtype=float)
    s_arr = np.array(measured_speedups, dtype=float)

    # Bound s in [0.0, 1.0]
    popt, _ = curve_fit(
        amdahl_speedup_formula,
        p_arr,
        s_arr,
        p0=[0.85],
        bounds=(0.0, 1.0),
    )
    s_parallel = float(popt[0])
    s_serial = 1.0 - s_parallel
    return s_parallel, s_serial


def fit_gustafson_law(workers: Sequence[int], measured_speedups: Sequence[float]) -> float:
    """Fit Gustafson's Law to empirical speedup measurements.

    Returns:
        serial_fraction alpha
    """
    p_arr = np.array(workers, dtype=float)
    s_arr = np.array(measured_speedups, dtype=float)

    popt, _ = curve_fit(
        gustafson_speedup_formula,
        p_arr,
        s_arr,
        p0=[0.1],
        bounds=(0.0, 1.0),
    )
    return float(popt[0])


@dataclass
class DataParallelSimulationResult:
    num_workers: int
    computation_time: float
    allreduce_comm_time: float
    total_step_time: float
    effective_throughput: float


def simulate_data_parallel_step(
    num_workers: int,
    batch_size: int = 1024,
    model_params_bytes: int = 40_000_000,  # ~10M float32 params
    flop_per_sample: float = 1e7,
    worker_tflops: float = 0.5,  # 0.5 TFLOPS per worker
    interconnect_bandwidth_gbps: float = 10.0,  # 10 Gbps ring all-reduce
    latency_sec: float = 1e-4,
) -> DataParallelSimulationResult:
    """Simulate Distributed Data Parallel (DDP) step timing with Ring All-Reduce communication model.

    In Ring All-Reduce across P workers:
        Total bytes transferred per worker = 2 * (P - 1) / P * ModelSizeBytes
        Latency terms = 2 * (P - 1) * Latency
    Computation per worker = (batch_size / P) * flop_per_sample / worker_tflops
    """
    local_batch = batch_size / num_workers
    comp_time = (local_batch * flop_per_sample) / (worker_tflops * 1e12)

    if num_workers > 1:
        ring_factor = 2.0 * (num_workers - 1) / num_workers
        bandwidth_bytes_per_sec = (interconnect_bandwidth_gbps * 1e9) / 8.0
        comm_bytes_time = (ring_factor * model_params_bytes) / bandwidth_bytes_per_sec
        comm_latency_time = 2.0 * (num_workers - 1) * latency_sec
        comm_time = comm_bytes_time + comm_latency_time
    else:
        comm_time = 0.0

    total_time = comp_time + comm_time
    throughput = batch_size / total_time

    return DataParallelSimulationResult(
        num_workers=num_workers,
        computation_time=comp_time,
        allreduce_comm_time=comm_time,
        total_step_time=total_time,
        effective_throughput=throughput,
    )


@dataclass
class PipelineParallelSimulationResult:
    num_stages: int
    num_microbatches: int
    stage_compute_time: float
    bubble_fraction: float
    total_step_time: float
    ideal_speedup: float
    effective_speedup: float


def simulate_pipeline_parallel_1f1b(
    num_stages: int,
    num_microbatches: int,
    total_forward_backward_time: float = 1.0,
) -> PipelineParallelSimulationResult:
    """Simulate 1F1B (One Forward, One Backward) pipeline schedule bubble fraction.

    In 1F1B pipeline with K stages and M microbatches:
        Bubble overhead fraction = (K - 1) / (M + K - 1)
        Total execution time = (M + K - 1) * (total_time / (K * M))
    """
    k = num_stages
    m = num_microbatches
    time_per_stage_mb = total_forward_backward_time / (k * m)

    # Pipeline schedule duration in units of microbatch-stage
    total_slots = m + k - 1
    total_time = total_slots * time_per_stage_mb
    bubble_frac = (k - 1) / float(total_slots)

    ideal_speedup = float(k)
    effective_speedup = total_forward_backward_time / total_time

    return PipelineParallelSimulationResult(
        num_stages=k,
        num_microbatches=m,
        stage_compute_time=time_per_stage_mb,
        bubble_fraction=bubble_frac,
        total_step_time=total_time,
        ideal_speedup=ideal_speedup,
        effective_speedup=effective_speedup,
    )
