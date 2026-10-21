"""
Tests for Track G: xv6 Operating-System Optimization.
Validates the Stride Scheduler implementation, Buffer Cache hit/miss profiling,
patch integrity, empirical metrics, and core OS stability.
"""

import os
import json
import csv
import pytest

TRACK_G_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATCH_FILE = os.path.join(TRACK_G_DIR, "patch", "xv6_stride_and_bcache.patch")
RESULTS_DIR = os.path.join(TRACK_G_DIR, "results")
RAW_RUN_FILE = os.path.join(RESULTS_DIR, "raw_xv6_run.json")
CSV_METRICS_FILE = os.path.join(RESULTS_DIR, "xv6_metrics.csv")
MD_REPORT_FILE = os.path.join(RESULTS_DIR, "xv6_metrics.md")
PNG_PLOT_FILE = os.path.join(RESULTS_DIR, "xv6_benchmarks.png")


def test_xv6_patch_exists_and_valid():
    """Verify that the xv6 unified diff patch exists and contains all required modifications."""
    assert os.path.exists(PATCH_FILE), f"Patch file missing: {PATCH_FILE}"
    with open(PATCH_FILE, "r", encoding="utf-8") as f:
        patch_text = f.read()

    assert len(patch_text) > 1000, "Patch file is too small or empty"
    # Check key function additions and modifications in the diff
    assert "setpriority" in patch_text, "Patch must include setpriority"
    assert "getpinfo" in patch_text, "Patch must include getpinfo"
    assert "getbcachestats" in patch_text, "Patch must include getbcachestats"
    assert "pstat.h" in patch_text, "Patch must include pstat.h"
    assert "_schedtest" in patch_text, "Patch must register _schedtest in Makefile"
    assert "_bcachetest" in patch_text, "Patch must register _bcachetest in Makefile"


def test_xv6_metrics_data_integrity():
    """Verify that raw benchmark outputs from QEMU were collected and have valid structure."""
    assert os.path.exists(RAW_RUN_FILE), f"Raw QEMU output missing: {RAW_RUN_FILE}"
    with open(RAW_RUN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "fork_output" in data
    assert "sched_output" in data
    assert "bcache_output" in data

    # Core OS verification: forktest must have passed
    assert "fork test OK" in data["fork_output"], "forktest did not report OK in QEMU run"

    # Scheduling benchmark tags
    assert "SNAPSHOT_SCHED" in data["sched_output"], "SNAPSHOT_SCHED tag missing from schedtest"
    assert "RESULT_SCHED" in data["sched_output"], "RESULT_SCHED tag missing from schedtest"

    # Buffer cache benchmark tags
    assert "RESULT_BCACHE phase=warm" in data["bcache_output"], "Warm phase tag missing"
    assert "RESULT_BCACHE phase=thrash" in data["bcache_output"], "Thrash phase tag missing"


def test_stride_scheduling_proportionality():
    """Verify that Stride Scheduling allocated CPU ticks and turnaround times proportionally."""
    with open(RAW_RUN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    sched_out = data["sched_output"]
    # Extract SNAPSHOT_SCHED rtime1=X rtime2=Y rtime3=Z
    for line in sched_out.splitlines():
        if "SNAPSHOT_SCHED" in line:
            parts = line.split()
            r1 = int(parts[1].split("=")[1])
            r2 = int(parts[2].split("=")[1])
            r3 = int(parts[3].split("=")[1])

            # Process 3 (30 tickets) must receive more runtime than Process 2 (20 tickets)
            # which must receive more than Process 1 (10 tickets)
            assert r3 > r2 > r1, f"Expected rtime3 > rtime2 > rtime1, got {r3}, {r2}, {r1}"
            # Verify ratio is roughly proportional to 3:2:1
            ratio21 = r2 / max(r1, 1)
            ratio31 = r3 / max(r1, 1)
            assert 1.2 <= ratio21 <= 3.0, f"Expected ratio21 around 2.0, got {ratio21:.2f}"
            assert 2.0 <= ratio31 <= 5.0, f"Expected ratio31 around 3.0, got {ratio31:.2f}"

    # Extract turnaround times from RESULT_SCHED lines
    turnarounds = {}
    for line in sched_out.splitlines():
        if "RESULT_SCHED" in line:
            # RESULT_SCHED child=X ... tickets=Y ... turnaround=Z
            parts = dict(kv.split("=") for kv in line.replace("RESULT_SCHED ", "").split())
            child = int(parts["child"])
            turnaround = int(parts["turnaround"])
            turnarounds[child] = turnaround

    # High-priority child (Child 3, 30 tickets) must complete before lower-priority children
    assert turnarounds[3] < turnarounds[2] < turnarounds[1], (
        f"Expected turnaround order T3 < T2 < T1, got {turnarounds}"
    )


def test_bcache_hit_rate_bounds():
    """Verify buffer cache hit rate on warm working sets and thrashing degradation."""
    with open(RAW_RUN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    bcache_out = data["bcache_output"]
    warm_hit_pct = None
    thrash_miss_pct = None

    for line in bcache_out.splitlines():
        if "RESULT_BCACHE phase=warm" in line:
            parts = dict(kv.split("=") for kv in line.replace("RESULT_BCACHE ", "").split())
            warm_hit_pct = int(parts["hit_pct"])
        elif "RESULT_BCACHE phase=thrash" in line:
            parts = dict(kv.split("=") for kv in line.replace("RESULT_BCACHE ", "").split())
            thrash_miss_pct = int(parts["miss_pct"])

    assert warm_hit_pct is not None, "Warm hit percentage not found"
    assert thrash_miss_pct is not None, "Thrash miss percentage not found"

    # Warm hit rate should be near 100% (at least 95%)
    assert warm_hit_pct >= 95, f"Expected warm hit pct >= 95%, got {warm_hit_pct}%"
    # Thrashing miss rate should reflect evictions (> 30%)
    assert thrash_miss_pct >= 30, f"Expected thrash miss pct >= 30%, got {thrash_miss_pct}%"


def test_documentation_and_reports_exist():
    """Verify that all report files and plots were generated."""
    assert os.path.exists(CSV_METRICS_FILE), f"CSV metrics file missing: {CSV_METRICS_FILE}"
    assert os.path.exists(MD_REPORT_FILE), f"Markdown report file missing: {MD_REPORT_FILE}"
    assert os.path.exists(PNG_PLOT_FILE), f"Plot image missing: {PNG_PLOT_FILE}"

    with open(CSV_METRICS_FILE, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 10, "CSV metrics file should contain multiple rows"

    with open(MD_REPORT_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "# Track G: xv6 Operating-System Optimization Report" in md_text
        assert "Stride Scheduling" in md_text
        assert "Buffer Cache" in md_text
