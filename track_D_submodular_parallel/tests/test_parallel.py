"""Unit tests for Parallel Program Optimization module in Track D."""

import numpy as np
import pytest

from track_D_submodular_parallel.src.parallel_optimization import (
    WorkStealingQueueSimulation,
    amdahl_speedup_formula,
    fit_amdahl_law,
    fit_gustafson_law,
    gustafson_speedup_formula,
    run_dynamic_parallel_workload,
    run_sequential_workload,
    run_static_parallel_workload,
    simulate_data_parallel_step,
    simulate_pipeline_parallel_1f1b,
    synthetic_compute_kernel,
)


def test_synthetic_compute_kernel_determinism() -> None:
    """Check that compute kernel returns identical deterministic results for same input."""
    task = (1, 100, 3.1415)
    r1 = synthetic_compute_kernel(task)
    r2 = synthetic_compute_kernel(task)
    assert r1 == r2
    assert isinstance(r1, float)


def test_workload_correctness_across_schedulers() -> None:
    """Check that sequential, static, dynamic, and work-stealing schedulers produce matching outputs."""
    tasks = [(i, 150, float(i) * 0.5) for i in range(16)]

    # 1. Sequential
    res_seq, t_seq = run_sequential_workload(tasks)
    assert len(res_seq) == 16

    # 2. Static parallel
    res_static, t_static = run_static_parallel_workload(tasks, num_workers=2)
    assert np.allclose(res_seq, res_static, atol=1e-10)

    # 3. Dynamic parallel
    res_dyn, t_dyn = run_dynamic_parallel_workload(tasks, num_workers=2)
    assert np.allclose(res_seq, res_dyn, atol=1e-10)

    # 4. Work-stealing simulation
    ws_sim = WorkStealingQueueSimulation(num_workers=2, tasks=tasks)
    t_ws = ws_sim.execute(total_tasks=len(tasks))
    assert len(ws_sim.results) == 16
    for i in range(16):
        assert np.isclose(ws_sim.results[i], res_seq[i], atol=1e-10)


def test_amdahl_law_fit_synthetic() -> None:
    """Verify Amdahl's Law fitting recovers known parallelizable fraction s."""
    true_s = 0.85
    workers = np.array([1, 2, 4, 8])
    speedups = amdahl_speedup_formula(workers, true_s)

    fitted_s, fitted_serial = fit_amdahl_law(workers, speedups)
    assert np.isclose(fitted_s, true_s, atol=1e-4)
    assert np.isclose(fitted_serial, 1.0 - true_s, atol=1e-4)


def test_gustafson_law_fit_synthetic() -> None:
    """Verify Gustafson's Law fitting recovers known serial fraction alpha."""
    true_alpha = 0.12
    workers = np.array([1, 2, 4, 8])
    speedups = gustafson_speedup_formula(workers, true_alpha)

    fitted_alpha = fit_gustafson_law(workers, speedups)
    assert np.isclose(fitted_alpha, true_alpha, atol=1e-4)


def test_data_parallel_scaling_model() -> None:
    """Verify Data Parallel ring all-reduce simulation metrics."""
    # Compute-dominated regime (high FLOPs per sample, high bandwidth)
    res_1 = simulate_data_parallel_step(num_workers=1, flop_per_sample=1e8, interconnect_bandwidth_gbps=100.0)
    res_4 = simulate_data_parallel_step(num_workers=4, flop_per_sample=1e8, interconnect_bandwidth_gbps=100.0)

    # 1 worker has zero allreduce communication time
    assert res_1.allreduce_comm_time == 0.0
    # 4 workers compute in 1/4 the time plus communication overhead
    assert res_4.computation_time < res_1.computation_time
    assert res_4.allreduce_comm_time > 0.0
    # Effective throughput with 4 workers should be higher than 1 worker in compute-bound regime
    assert res_4.effective_throughput > res_1.effective_throughput

    # Communication-dominated regime (low FLOPs, slow link)
    res_comm_bound = simulate_data_parallel_step(num_workers=8, flop_per_sample=1e6, interconnect_bandwidth_gbps=1.0)
    assert res_comm_bound.allreduce_comm_time > res_comm_bound.computation_time


def test_pipeline_parallel_1f1b_bubble_behavior() -> None:
    """Verify 1F1B pipeline bubble fraction shrinks as microbatch count M grows."""
    # With K=4 stages:
    # M=4 microbatches: bubble fraction = (4-1) / (4+4-1) = 3/7 ~ 42.8%
    res_m4 = simulate_pipeline_parallel_1f1b(num_stages=4, num_microbatches=4)
    # M=16 microbatches: bubble fraction = (4-1) / (16+4-1) = 3/19 ~ 15.8%
    res_m16 = simulate_pipeline_parallel_1f1b(num_stages=4, num_microbatches=16)

    assert res_m16.bubble_fraction < res_m4.bubble_fraction
    assert res_m16.effective_speedup > res_m4.effective_speedup
