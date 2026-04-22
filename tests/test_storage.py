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


class TestGetEntityById:
    def test_get_entity_by_id_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("TestEntity", "Testing")
        entity = store.get_entity_by_id(eid)
        assert entity is not None
        assert entity["id"] == eid
        assert entity["name"] == "TestEntity"
        assert entity["entity_type"] == "Testing"
        assert "status" in entity
        assert "created_at" in entity
        assert "updated_at" in entity

    def test_get_entity_by_id_not_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        assert store.get_entity_by_id(99999) is None


class TestSearchFts:
    def test_search_fts_empty_query(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        assert store.search_fts("") == []
        assert store.search_fts("   ") == []

    def test_search_fts_with_match(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        if not store._fts_available:
            pytest.skip("FTS5 not available")
        eid = store.upsert_entity("Apple", "Fruit")
        store.add_observations(eid, ["crisp and sweet"])
        results = store.search_fts("Apple")
        assert len(results) >= 1
        assert any(r["entity_id"] == eid for r in results)

    def test_search_fts_no_match(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        if not store._fts_available:
            pytest.skip("FTS5 not available")
        eid = store.upsert_entity("Apple", "Fruit")
        store.add_observations(eid, ["crisp"])
        results = store.search_fts("Zebra")
        assert results == []

    def test_search_fts_not_available(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store._fts_available = False
        assert store.search_fts("anything") == []


class TestSearchEmbeddings:
    def test_search_embeddings_not_loaded(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store._vec_loaded = False
        assert store.search_embeddings(b"\x00" * 1536) == []

    def test_search_embeddings_happy_path(self, tmp_path):
        import struct

        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        if not store._vec_loaded:
            pytest.skip("sqlite-vec not available")
        eid = store.upsert_entity("Test", "Type")
        embedding = struct.pack(f"<{384}f", *[1.0] * 384)
        store.store_embedding(eid, embedding)
        results = store.search_embeddings(embedding, limit=5)
        assert len(results) >= 1
        assert results[0]["entity_id"] == eid
        assert abs(results[0]["distance"]) < 1e-9


class TestSyncFts:
    def test_sync_fts_after_upsert_and_observations(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        if not store._fts_available:
            pytest.skip("FTS5 not available")
        eid = store.upsert_entity("SyncTest", "Type")
        store.add_observations(eid, ["observation one", "observation two"])
        row = store.db.execute(
            "SELECT * FROM entity_fts WHERE rowid = ?", (eid,)
        ).fetchone()
        assert row is not None
        assert row["name"] == "SyncTest"
        assert "observation one" in row["obs_text"]


class TestInitAccess:
    def test_init_access_idempotent(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.init_access(eid)
        store.init_access(eid)
        data = store.get_access_data([eid])
        assert data[eid]["access_count"] == 1


class TestGetObservationsWithIds:
    def test_get_observations_with_ids_keys(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs1", "obs2"])
        obs_list = store.get_observations_with_ids(eid)
        assert len(obs_list) == 2
        for obs in obs_list:
            assert set(obs.keys()) == {
                "id",
                "content",
                "similarity_flag",
                "kind",
                "supersedes",
                "superseded_at",
            }
            assert obs["kind"] == "generic"
            assert obs["superseded_at"] is None


class TestGetObservationsBatch:
    def test_get_observations_batch_multiple(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        e3 = store.upsert_entity("E3", "T")
        store.add_observations(e1, ["a", "b"])
        store.add_observations(e2, ["c"])
        result = store.get_observations_batch([e1, e2, e3])
        assert result[e1] == ["a", "b"]
        assert result[e2] == ["c"]
        assert e3 not in result


class TestGetEntitiesBatch:
    def test_get_entities_batch_existing_and_missing(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        result = store.get_entities_batch([e1, 99999])
        assert e1 in result
        assert 99999 not in result
        assert result[e1]["name"] == "E1"


class TestGetObservationsWithIdsBatch:
    def test_get_observations_with_ids_batch(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        store.add_observations(e1, ["a", "b"])
        result = store.get_observations_with_ids_batch([e1])
        assert len(result[e1]) == 2
        for obs in result[e1]:
            assert "id" in obs
            assert "content" in obs
            assert "kind" in obs


class TestGetAccessData:
    def test_get_access_data_default_zero(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        data = store.get_access_data([eid])
        assert data[eid]["access_count"] == 0


class TestGetEntityDegrees:
    def test_get_entity_degrees_no_relations(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        degrees = store.get_entity_degrees([eid])
        assert degrees[eid] == 0

    def test_get_entity_degrees_with_relations(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.create_relation(e1, e2, "related_to")
        degrees = store.get_entity_degrees([e1, e2])
        assert degrees[e1] == 1
        assert degrees[e2] == 1


class TestRecordCoOccurrences:
    def test_record_co_occurrences_two_entities(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.record_co_occurrences([e1, e2])
        coocs = store.get_co_occurrences([e1, e2])
        assert (e1, e2) in coocs
        assert coocs[(e1, e2)]["co_count"] == 1

    def test_record_co_occurrences_three_entities(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        e3 = store.upsert_entity("E3", "T")
        store.record_co_occurrences([e1, e2, e3])
        coocs = store.get_co_occurrences([e1, e2, e3])
        assert len(coocs) == 3

    def test_record_co_occurrences_idempotent(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.record_co_occurrences([e1, e2])
        store.record_co_occurrences([e1, e2])
        coocs = store.get_co_occurrences([e1, e2])
        assert coocs[(e1, e2)]["co_count"] == 2


class TestGetCoOccurrences:
    def test_get_co_occurrences_no_pairs(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        assert store.get_co_occurrences([e1]) == {}

    def test_get_co_occurrences_with_pairs(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.record_co_occurrences([e1, e2])
        coocs = store.get_co_occurrences([e1, e2])
        assert (e1, e2) in coocs
        assert coocs[(e1, e2)]["co_count"] == 1


class TestGetAllEntities:
    def test_get_all_entities_nested_observations(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        store.add_observations(e1, ["obs1", "obs2"])
        entities = store.get_all_entities()
        assert len(entities) == 1
        assert entities[0]["observations"] == ["obs1", "obs2"]


class TestGetRelationById:
    def test_get_relation_by_id_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.create_relation(e1, e2, "related_to")
        row = store.db.execute(
            "SELECT id FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        rid = row["id"]
        rel = store.get_relation_by_id(rid)
        assert rel is not None
        assert rel["from_entity"] == e1
        assert rel["to_entity"] == e2
        assert rel["relation_type"] == "related_to"

    def test_get_relation_by_id_not_exists(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        assert store.get_relation_by_id(99999) is None


class TestGetRelationsForEntity:
    def test_get_relations_for_entity_directions(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        e1 = store.upsert_entity("E1", "T")
        e2 = store.upsert_entity("E2", "T")
        store.create_relation(e1, e2, "related_to")
        rels_from = store.get_relations_for_entity(e1)
        assert any(
            r["direction"] == "from" and r["target_name"] == "E2" for r in rels_from
        )
        rels_to = store.get_relations_for_entity(e2)
        assert any(
            r["direction"] == "to" and r["target_name"] == "E1" for r in rels_to
        )


class TestUpsertEntityUpdatedAt:
    def test_upsert_entity_updates_updated_at(self, tmp_path):
        import time

        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type1")
        first = store.get_entity_by_id(eid)["updated_at"]
        time.sleep(1.1)
        store.upsert_entity("Test", "Type2")
        second = store.get_entity_by_id(eid)["updated_at"]
        assert second > first


class TestSearchEntitiesAdvanced:
    def test_search_entities_by_type(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        store.upsert_entity("Apple", "Fruit")
        store.upsert_entity("Carrot", "Vegetable")
        results = store.search_entities("Vegetable")
        assert any(r["name"] == "Carrot" for r in results)


class TestDeleteObservations:
    def test_delete_observations_exact_match(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["keep", "remove"])
        deleted = store.delete_observations(eid, ["remove"])
        assert deleted == 1
        obs = store.get_observations(eid)
        assert "remove" not in obs
        assert "keep" in obs


class TestAddObservationsAdvanced:
    def test_add_observations_with_kind(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["special"], kind="custom")
        obs = store.get_observations_with_ids(eid)
        assert len(obs) == 1
        assert obs[0]["kind"] == "custom"

    def test_add_observations_with_supersedes(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["old"])
        old_id = store.get_observations_with_ids(eid)[0]["id"]
        store.add_observations(eid, ["new"], supersedes=old_id)
        obs_excluded = store.get_observations_with_ids(eid, exclude_superseded=True)
        assert all(o["content"] != "old" for o in obs_excluded)
        obs_included = store.get_observations_with_ids(eid, exclude_superseded=False)
        assert any(o["content"] == "old" for o in obs_included)


class TestAddObservationsBatch:
    def test_add_observations_batch_10_new(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        obs = [f"obs{i}" for i in range(10)]
        count = store.add_observations(eid, obs)
        assert count == 10
        assert store.get_observations(eid) == obs

    def test_add_observations_batch_mixed(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["dup1", "dup2", "dup3", "dup4", "dup5"])
        count = store.add_observations(eid, ["dup1", "new1", "dup2", "new2", "dup3", "new3", "dup4", "new4", "dup5", "new5"])
        assert count == 5
        obs = store.get_observations(eid)
        assert len(obs) == 10
        for o in ["new1", "new2", "new3", "new4", "new5"]:
            assert o in obs

    def test_add_observations_batch_kind(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["a", "b", "c"], kind="custom")
        obs = store.get_observations_with_ids(eid)
        assert len(obs) == 3
        for o in obs:
            assert o["kind"] == "custom"

    def test_add_observations_batch_idempotent(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        obs = ["x", "y", "z"]
        assert store.add_observations(eid, obs) == 3
        assert store.add_observations(eid, obs) == 0
        assert store.get_observations(eid) == obs

    def test_add_observations_batch_supersedes(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["old"])
        old_id = store.get_observations_with_ids(eid)[0]["id"]
        count = store.add_observations(eid, ["new1", "new2"], supersedes=old_id)
        assert count == 2
        obs_excluded = store.get_observations_with_ids(eid, exclude_superseded=True)
        assert all(o["content"] != "old" for o in obs_excluded)
        obs_included = store.get_observations_with_ids(eid, exclude_superseded=False)
        assert any(o["content"] == "old" for o in obs_included)
        # The last inserted observation should reference the superseded one
        last_obs = store.db.execute(
            "SELECT content, supersedes FROM observations WHERE entity_id = ? ORDER BY id DESC LIMIT 1",
            (eid,),
        ).fetchone()
        assert last_obs["content"] == "new2"
        assert last_obs["supersedes"] == old_id

    def test_add_observations_batch_internal_dedup(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        count = store.add_observations(eid, ["dup", "dup", "dup"])
        assert count == 1
        obs = store.get_observations(eid)
        assert obs.count("dup") == 1


class TestAutoCommit:
    def test_upsert_entity_auto_commit_false_no_commit(self):
        store = MemoryStore(":memory:")
        store.init_db()
        store.db.execute("BEGIN IMMEDIATE")
        store.upsert_entity("Test", "Type", auto_commit=False)
        count = store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 1
        store.db.rollback()
        count = store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 0

    def test_add_observations_auto_commit_false_no_commit(self):
        store = MemoryStore(":memory:")
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.db.execute("BEGIN IMMEDIATE")
        store.add_observations(eid, ["obs1"], auto_commit=False)
        count = store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert count == 1
        store.db.rollback()
        count = store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert count == 0

    def test_delete_observations_auto_commit_false_no_commit(self):
        store = MemoryStore(":memory:")
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs1", "obs2"])
        store.db.execute("BEGIN IMMEDIATE")
        store.delete_observations(eid, ["obs1"], auto_commit=False)
        count = store.db.execute(
            "SELECT COUNT(*) FROM observations WHERE content = 'obs1'"
        ).fetchone()[0]
        assert count == 0
        store.db.rollback()
        count = store.db.execute(
            "SELECT COUNT(*) FROM observations WHERE content = 'obs1'"
        ).fetchone()[0]
        assert count == 1

    def test_create_relation_auto_commit_false_no_commit(self):
        store = MemoryStore(":memory:")
        store.init_db()
        e1 = store.upsert_entity("A", "Type")
        e2 = store.upsert_entity("B", "Type")
        store.db.execute("BEGIN IMMEDIATE")
        store.create_relation(e1, e2, "relaciona", auto_commit=False)
        count = store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert count == 1
        store.db.rollback()
        count = store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert count == 0

    def test_crud_defaults_auto_commit_true(self):
        store = MemoryStore(":memory:")
        store.init_db()
        eid = store.upsert_entity("Test", "Type")
        assert store.get_entity_by_name("Test") is not None
        store.add_observations(eid, ["obs1"])
        assert store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
        store.create_relation(eid, eid, "self")
        assert store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 1
        store.delete_observations(eid, ["obs1"])
        assert store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0

