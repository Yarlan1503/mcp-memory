"""Smoke checks for the MCP stdio stress harness helpers.

The full harness starts real MCP server subprocesses and is intended to be run
manually because it depends on local model cache/network availability. These
tests keep the script importable and verify the error taxonomy used by reports.
"""

from __future__ import annotations

from scripts.mcp_stdio_stress import _classify_error, _percentile


def test_mcp_stdio_stress_error_taxonomy():
    assert _classify_error("McpError -32001 request timeout") == "-32001"
    assert _classify_error("McpError -32000 server error") == "-32000"
    assert _classify_error("RuntimeError: Not connected") == "Not connected"
    assert _classify_error("sqlite3.OperationalError: database is locked") == "database locked"
    assert _classify_error("cannot start a transaction within a transaction") == "transaction errors"
    assert _classify_error("Already borrowed: connection") == "Already borrowed"
    assert _classify_error("something surprising") == "other"


def test_mcp_stdio_stress_percentiles():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile([], 95) is None
    assert _percentile(values, 95) == 4.0
    assert _percentile(values, 99) == 4.0
