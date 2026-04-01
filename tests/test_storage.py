"""Tests for MCP Memory storage layer."""

import pytest

from mcp_memory.storage import MemoryStore


class TestInitDb:
    def test_init_db_creates_all_tables(self, tmp_path):
        """Verify all expected tables are created."""
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        cur = store.db.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in cur.fetchall()}
        expected = {
            "entities",
            "observations",
            "relations",
            "db_metadata",
            "entity_access",
            "co_occurrences",
            "search_events",
            "search_results",
            "implicit_feedback",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_init_db_creates_indexes(self, tmp_path):
        """Verify indexes are created."""
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        cur = store.db.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        indexes = {r["name"] for r in cur.fetchall()}
        assert len(indexes) > 0


class TestUpsertEntity:
    def test_upsert_entity_creates_new(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("TestEntity", "Testing")
        assert eid > 0
        entity = store.get_entity_by_name("TestEntity")
        assert entity is not None
        assert entity["name"] == "TestEntity"
        assert entity["entity_type"] == "Testing"

    def test_upsert_entity_updates_existing(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store.upsert_entity("TestEntity", "Type1")
        store.upsert_entity("TestEntity", "Type2")  # upsert
        entities = store.db.execute("SELECT * FROM entities").fetchall()
        assert len(entities) == 1  # No duplicate
        assert entities[0]["entity_type"] == "Type2"


class TestGetEntity:
    def test_get_entity_by_name_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store.upsert_entity("TestEntity", "Testing")
        entity = store.get_entity_by_name("TestEntity")
        assert entity is not None
        assert entity["name"] == "TestEntity"

    def test_get_entity_by_name_not_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        entity = store.get_entity_by_name("NonExistent")
        assert entity is None


class TestObservations:
    def test_add_observations(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        count = store.add_observations(eid, ["obs1", "obs2", "obs3"])
        assert count == 3
        obs = store.get_observations(eid)
        assert len(obs) == 3
        assert "obs1" in obs

    def test_add_observations_skips_duplicates(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs1"])
        count = store.add_observations(eid, ["obs1"])  # duplicate
        assert count == 0  # Skipped


class TestRelations:
    def test_create_relation(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("Entity1", "Type")
        e2 = store.upsert_entity("Entity2", "Type")
        result = store.create_relation(e1, e2, "related_to")
        assert result is True

    def test_create_relation_duplicate(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("Entity1", "Type")
        e2 = store.upsert_entity("Entity2", "Type")
        store.create_relation(e1, e2, "related_to")
        result = store.create_relation(e1, e2, "related_to")  # duplicate
        assert result is False


class TestDeleteEntities:
    def test_delete_entities_by_names(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("Entity1", "Type")
        e2 = store.upsert_entity("Entity2", "Type")
        e3 = store.upsert_entity("Entity3", "Type")
        store.add_observations(e1, ["obs1"])

        deleted = store.delete_entities_by_names(["Entity1", "Entity2"])
        assert deleted == 2

        remaining = store.get_entity_by_name("Entity3")
        assert remaining is not None
        assert store.get_entity_by_name("Entity1") is None

        # Verify CASCADE deleted observations
        obs = store.get_observations(e1)
        assert len(obs) == 0


class TestSearchEntities:
    def test_search_entities_by_name(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store.upsert_entity("Apple", "Fruit")
        store.upsert_entity("Banana", "Fruit")
        results = store.search_entities("Apple")
        assert len(results) >= 1
        assert any(r["name"] == "Apple" for r in results)

    def test_search_entities_by_observation(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["unique_observation_text"])
        results = store.search_entities("unique_observation_text")
        assert len(results) >= 1


class TestSearchEventLogging:
    def test_log_search_event(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        event_id = store.log_search_event(
            query_text="test query",
            treatment=1,
            k_limit=10,
            num_results=5,
            duration_ms=123.45,
            engine_used="limbic",
        )
        assert event_id > 0


class TestAccessTracking:
    def test_record_access(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.init_access(eid)
        store.record_access(eid)
        store.record_access(eid)
        data = store.get_access_data([eid])
        assert data[eid]["access_count"] == 3  # 1 init + 2 record
