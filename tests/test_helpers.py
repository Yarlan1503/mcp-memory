"""Tests for mcp_memory._helpers module."""

import sys
from unittest.mock import patch

sys.path.insert(0, "src")
from mcp_memory._helpers import _entity_to_output


class TestGetEngine:
    """Tests for _get_engine helper."""

    def test_get_engine_returns_none_when_module_missing(self, monkeypatch):
        """When mcp_memory.embeddings is not importable, return None."""
        from mcp_memory._helpers import _get_engine

        # Setting a module to None in sys.modules causes ImportError on next import
        monkeypatch.setitem(sys.modules, "mcp_memory.embeddings", None)
        result = _get_engine()
        assert result is None


class TestEntityToOutput:
    """Tests for _entity_to_output helper."""

    def test_entity_to_output_basic(self):
        """Convert a row dict + observations to EntityOutput."""
        row = {"name": "TestEntity", "entity_type": "Testing"}
        observations = ["obs1", "obs2"]

        result = _entity_to_output(row, observations)

        assert result == {
            "name": "TestEntity",
            "entityType": "Testing",
            "status": "activo",
            "observations": ["obs1", "obs2"],
        }

    def test_entity_to_output_with_relations(self):
        """Include relations in output when passed."""
        row = {"name": "TestEntity", "entity_type": "Testing", "status": "archivado"}
        observations = ["obs1"]
        relations = [{"from_entity": "A", "to_entity": "B", "relation_type": "knows"}]

        result = _entity_to_output(row, observations, relations)

        assert result == {
            "name": "TestEntity",
            "entityType": "Testing",
            "status": "archivado",
            "observations": ["obs1"],
            "relations": [{"from_entity": "A", "to_entity": "B", "relation_type": "knows"}],
        }
