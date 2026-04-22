"""Tests for open_nodes batch prefetch optimization."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_open_nodes.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


class TestOpenNodesBatch:
    """Integration tests for open_nodes with batch prefetch."""

    def test_open_nodes_single_entity_keys(self, store, monkeypatch):
        """open_nodes with 1 entity returns expected keys."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        eid = store.upsert_entity("SingleEntity", "Test")
        store.add_observations(eid, ["obs1"])
        store.create_relation(eid, eid, "related_to")
        store.add_reflection("entity", eid, "nolan", "test reflection")

        result = open_nodes(["SingleEntity"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 1

        ent = entities[0]
        assert "name" in ent
        assert "entityType" in ent
        assert "status" in ent
        assert "observations" in ent
        assert "relations" in ent
        assert "reflections" in ent
        assert ent["name"] == "SingleEntity"

    def test_open_nodes_multiple_entities_order_preserved(self, store, monkeypatch):
        """open_nodes with 3 entities preserves input order."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        e1 = store.upsert_entity("EntityA", "Test")
        e2 = store.upsert_entity("EntityB", "Test")
        e3 = store.upsert_entity("EntityC", "Test")
        store.add_observations(e1, ["obs A"])
        store.add_observations(e2, ["obs B"])
        store.add_observations(e3, ["obs C"])

        result = open_nodes(["EntityC", "EntityA", "EntityB"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 3
        assert entities[0]["name"] == "EntityC"
        assert entities[1]["name"] == "EntityA"
        assert entities[2]["name"] == "EntityB"

    def test_open_nodes_filters_kinds(self, store, monkeypatch):
        """open_nodes with kinds filter only returns matching observations."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        eid = store.upsert_entity("KindEntity", "Test")
        store.add_observations(eid, ["generic obs"], kind="generic")
        store.add_observations(eid, ["hallazgo obs"], kind="hallazgo")
        store.add_observations(eid, ["metadata obs"], kind="metadata")

        result = open_nodes(["KindEntity"], kinds=["hallazgo"])
        assert "error" not in result
        obs = result["entities"][0]["observations"]
        assert len(obs) == 1
        assert obs[0]["kind"] == "hallazgo"

    def test_open_nodes_include_superseded(self, store, monkeypatch):
        """open_nodes with include_superseded=True includes superseded observations."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        eid = store.upsert_entity("SupersedeEntity", "Test")
        store.add_observations(eid, ["old obs"])
        old_obs = store.get_observations_with_ids(eid)[0]
        store.add_observations(eid, ["new obs"], supersedes=old_obs["id"])

        # Default: excludes superseded
        result_default = open_nodes(["SupersedeEntity"])
        obs_default = result_default["entities"][0]["observations"]
        assert len(obs_default) == 1
        assert obs_default[0]["content"] == "new obs"

        # Include superseded
        result_included = open_nodes(["SupersedeEntity"], include_superseded=True)
        obs_included = result_included["entities"][0]["observations"]
        contents = {o["content"] for o in obs_included}
        assert contents == {"old obs", "new obs"}

    def test_open_nodes_missing_entity_returns_empty(self, store, monkeypatch):
        """open_nodes with a nonexistent name returns empty entities list."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        result = open_nodes(["NonExistent"])
        assert "error" not in result
        assert result["entities"] == []

    def test_open_nodes_mixed_existing_and_missing(self, store, monkeypatch):
        """open_nodes skips missing entities but preserves order for existing ones."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        e1 = store.upsert_entity("Exists", "Test")
        store.add_observations(e1, ["obs1"])

        result = open_nodes(["Missing1", "Exists", "Missing2"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 1
        assert entities[0]["name"] == "Exists"

    def test_open_nodes_relations_and_reflections_batch(self, store, monkeypatch):
        """open_nodes correctly batches relations and reflections across entities."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store)

        e1 = store.upsert_entity("Alpha", "Test")
        e2 = store.upsert_entity("Beta", "Test")
        store.create_relation(e1, e2, "related_to", context="alpha->beta")
        store.add_reflection("entity", e1, "nolan", "reflection alpha", "insight")
        store.add_reflection("entity", e2, "sofia", "reflection beta", "satisfaccion")

        result = open_nodes(["Alpha", "Beta"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 2

        alpha = next(e for e in entities if e["name"] == "Alpha")
        beta = next(e for e in entities if e["name"] == "Beta")

        assert len(alpha["relations"]) == 1
        assert alpha["relations"][0]["relation_type"] == "related_to"
        assert alpha["relations"][0]["target_name"] == "Beta"
        assert len(alpha["reflections"]) == 1
        assert alpha["reflections"][0]["content"] == "reflection alpha"

        assert len(beta["relations"]) == 1
        assert beta["relations"][0]["relation_type"] == "related_to"
        assert beta["relations"][0]["target_name"] == "Alpha"
        assert len(beta["reflections"]) == 1
        assert beta["reflections"][0]["content"] == "reflection beta"
