"""Tests for entity management tools in mcp_memory.tools.entity_mgmt."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Pre-load server module to break the circular import between
# mcp_memory.server and mcp_memory.tools.entity_mgmt.
import mcp_memory.server  # noqa: F401
import mcp_memory.tools.entity_mgmt as entity_mgmt_module


def _patch_store(store):
    """Temporarily patch the global store reference in entity_mgmt."""
    import mcp_memory.server as _server_mod

    original = _server_mod.store
    _server_mod.store = store
    return original


class TestAnalyzeEntitySplit:
    """Tests for analyze_entity_split tool."""

    def test_small_entity_no_split(self, store_with_schema):
        """Entity with few observations returns needs_split=False."""
        entity_id = store_with_schema.upsert_entity("SmallEntity", "Persona")
        store_with_schema.add_observations(entity_id, ["obs 1", "obs 2"])
        original = _patch_store(store_with_schema)
        try:
            result = entity_mgmt_module.analyze_entity_split("SmallEntity")
        finally:
            import mcp_memory.server as _server_mod

            _server_mod.store = original

        assert "error" not in result
        assert "analysis" in result
        assert result["analysis"]["needs_split"] is False
        assert result["analysis"]["observation_count"] == 2

    def test_nonexistent_entity_error(self, store_with_schema):
        """Non-existent entity returns an error dict."""
        original = _patch_store(store_with_schema)
        try:
            result = entity_mgmt_module.analyze_entity_split("NonExistent")
        finally:
            import mcp_memory.server as _server_mod

            _server_mod.store = original

        assert "error" in result
        assert "Entity not found" in result["error"]


class TestProposeEntitySplitTool:
    """Tests for propose_entity_split_tool."""

    def test_small_entity_returns_none(self, store_with_schema):
        """Entity below threshold returns proposal=None."""
        entity_id = store_with_schema.upsert_entity("SmallEntity", "Componente")
        store_with_schema.add_observations(entity_id, ["obs 1", "obs 2"])
        original = _patch_store(store_with_schema)
        try:
            result = entity_mgmt_module.propose_entity_split_tool("SmallEntity")
        finally:
            import mcp_memory.server as _server_mod

            _server_mod.store = original

        assert "error" not in result
        assert result["proposal"] is None
        assert "does not need splitting" in result["message"]


class TestFindSplitCandidates:
    """Tests for find_split_candidates tool."""

    def test_empty_db_returns_empty(self, store_with_schema):
        """Empty database returns an empty candidates list."""
        original = _patch_store(store_with_schema)
        try:
            result = entity_mgmt_module.find_split_candidates()
        finally:
            import mcp_memory.server as _server_mod

            _server_mod.store = original

        assert "error" not in result
        assert result["candidates"] == []


class TestConsolidationReport:
    """Tests for consolidation_report tool."""

    def test_empty_db_counts_zero(self, store_with_schema):
        """Empty DB report has all counts set to 0 and empty lists/dicts."""
        original = _patch_store(store_with_schema)
        try:
            result = entity_mgmt_module.consolidation_report(stale_days=90.0)
        finally:
            import mcp_memory.server as _server_mod

            _server_mod.store = original

        assert "error" not in result
        assert "summary" in result
        s = result["summary"]
        assert s["total_entities"] == 0
        assert s["total_observations"] == 0
        assert s["split_candidates_count"] == 0
        assert s["flagged_observations_count"] == 0
        assert s["stale_entities_count"] == 0
        assert s["large_entities_count"] == 0

        assert result["split_candidates"] == []
        assert result["flagged_observations"] == {}
        assert result["stale_entities"] == []
        assert result["large_entities"] == []
