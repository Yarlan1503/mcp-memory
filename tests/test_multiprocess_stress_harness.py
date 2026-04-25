"""Smoke test for the multi-process tool-layer stress harness."""

from __future__ import annotations

from scripts.multiprocess_stress import HarnessConfig, run_harness


def test_multiprocess_stress_harness_smoke(tmp_path):
    summary = run_harness(
        HarnessConfig(
            db_path=str(tmp_path / "multiprocess-stress.db"),
            processes=2,
            iterations=6,
            timeout_ms=10_000,
            seed_entities=4,
            seed_observations=4,
            queue_timeout=45,
        )
    )

    assert summary["worker_done"] == 2
    assert summary["operations"] == 12
    assert summary["process_exitcodes"] == [0, 0]
    assert all(count == 0 for count in summary["errors_by_type"].values())
    assert set(summary["latencies_by_op"]) == {
        "add_observations",
        "add_reflection",
        "search_semantic",
        "consolidation_report",
        "find_split_candidates",
        "long_read_then_write",
    }
