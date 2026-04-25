"""Tests for heavy tool backpressure behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event

from mcp_memory import backpressure


def test_bounded_heavy_tool_returns_structured_busy_payload_and_releases_slot(monkeypatch):
    """Concurrent saturation fails fast and later calls can acquire the slot."""
    monkeypatch.setattr(backpressure, "HEAVY_TOOL_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(backpressure, "HEAVY_TOOL_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(backpressure, "_heavy_tool_slots", BoundedSemaphore(1))

    entered = Event()
    release = Event()

    @backpressure.bounded_heavy_tool
    def heavy_tool(value: str) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=1), "test timed out waiting to release slot"
        return {"ok": True, "value": value}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(heavy_tool, "first")
        assert entered.wait(timeout=1), "first call did not acquire slot"

        saturated = executor.submit(heavy_tool, "second").result(timeout=1)

        assert saturated == {
            "error": "server_busy",
            "code": "heavy_tool_backpressure",
            "message": "Too many heavy maintenance operations are running (limit=1). Retry later.",
            "retryable": True,
        }

        release.set()
        assert first.result(timeout=1) == {"ok": True, "value": "first"}

    # The slot must be released after the in-flight call completes.
    assert heavy_tool("after-release") == {"ok": True, "value": "after-release"}
