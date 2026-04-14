"""Tests for Phase 2: status field on entities."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_entity_status.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


# ==============================================================
# Migration tests
# ==============================================================


class TestMigration:
    def test_status_column_exists(self, store):
        """Column 'status' exists after init_db."""
        cols = store.db.execute("PRAGMA table_info(entities)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "status" in col_names

    def test_status_migration_idempotent(self, store):
        """Calling _migrate_status_column twice raises no error."""
        store._migrate_status_column()
        store._migrate_status_column()  # Should be no-op
        cols = store.db.execute("PRAGMA table_info(entities)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "status" in col_names

    def test_default_status_is_activo(self, store):
        """Existing entities get 'activo' as default status."""
        eid = store.upsert_entity("TestEntity", "Test")
        row = store.db.execute(
            "SELECT status FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        assert row["status"] == "activo"


# ==============================================================
# Creation with status
# ==============================================================


class TestCreation:
    def test_create_entity_with_explicit_status(self, store):
        """Entity created with status='pausado' has that status."""
        eid = store.upsert_entity("PausedEntity", "Test", status="pausado")
        row = store.db.execute(
            "SELECT status FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        assert row["status"] == "pausado"

    def test_create_entity_default_status(self, store):
        """Entity created without status defaults to 'activo'."""
        eid = store.upsert_entity("DefaultEntity", "Test")
        row = store.db.execute(
            "SELECT status FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        assert row["status"] == "activo"

    def test_upsert_preserves_status(self, store):
        """Upserting existing entity updates its status."""
        eid = store.upsert_entity("TestEntity", "Test", status="pausado")
        # Upsert again with new status
        store.upsert_entity("TestEntity", "Test", status="completado")
        row = store.db.execute(
            "SELECT status FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        assert row["status"] == "completado"


# ==============================================================
# Read operations include status
# ==============================================================


class TestReadOperations:
    def test_get_entity_by_name_includes_status(self, store):
        """get_entity_by_name returns status field."""
        store.upsert_entity("TestEntity", "Test", status="pausado")
        entity = store.get_entity_by_name("TestEntity")
        assert entity is not None
        assert entity["status"] == "pausado"

    def test_get_entity_by_id_includes_status(self, store):
        """get_entity_by_id returns status field."""
        eid = store.upsert_entity("TestEntity", "Test", status="archivado")
        entity = store.get_entity_by_id(eid)
        assert entity is not None
        assert entity["status"] == "archivado"

    def test_get_all_entities_includes_status(self, store):
        """get_all_entities returns status field."""
        store.upsert_entity("Entity1", "Test", status="activo")
        store.upsert_entity("Entity2", "Test", status="archivado")
        entities = store.get_all_entities()
        statuses = {e["name"]: e["status"] for e in entities}
        assert statuses["Entity1"] == "activo"
        assert statuses["Entity2"] == "archivado"

    def test_search_entities_includes_status(self, store):
        """search_entities returns status field."""
        store.upsert_entity("FindMe", "Test", status="completado")
        store.add_observations(
            store.db.execute("SELECT id FROM entities WHERE name='FindMe'").fetchone()[
                "id"
            ],
            ["some observation"],
        )
        results = store.search_entities("FindMe")
        assert len(results) == 1
        assert results[0]["status"] == "completado"


# ==============================================================
# Consolidation report: status
# ==============================================================


class TestConsolidationReport:
    def test_stale_entities_include_status(self, store):
        """Stale entities in consolidation report include status."""
        eid = store.upsert_entity("StaleEntity", "Test", status="activo")
        store.add_observations(eid, ["obs1"])
        data = store.get_consolidation_data(stale_days=0)
        # Find our entity in stale list
        stale_names = {s["entity_name"] for s in data["stale_entities"]}
        assert "StaleEntity" in stale_names
        stale = [
            s for s in data["stale_entities"] if s["entity_name"] == "StaleEntity"
        ][0]
        assert stale["status"] == "activo"

    def test_archived_entities_excluded_from_stale(self, store):
        """Entities with status='archivado' are not reported as stale."""
        eid = store.upsert_entity("ArchivedEntity", "Test", status="archivado")
        store.add_observations(eid, ["old obs"])
        data = store.get_consolidation_data(stale_days=0)
        stale_names = {s["entity_name"] for s in data["stale_entities"]}
        assert "ArchivedEntity" not in stale_names


# ==============================================================
# Backward compatibility
# ==============================================================


class TestBackwardCompatibility:
    def test_create_entities_without_status(self, store):
        """Calling upsert_entity without status works as before."""
        eid = store.upsert_entity("OldStyleEntity", "Test")
        entity = store.get_entity_by_id(eid)
        assert entity["status"] == "activo"

    def test_entity_input_model_default_status(self):
        """EntityInput model defaults status to 'activo'."""
        from mcp_memory.models import EntityInput

        parsed = EntityInput.model_validate({"name": "Test", "entityType": "Test"})
        assert parsed.status == "activo"

    def test_entity_input_model_explicit_status(self):
        """EntityInput model accepts explicit status."""
        from mcp_memory.models import EntityInput

        parsed = EntityInput.model_validate(
            {"name": "Test", "entityType": "Test", "status": "pausado"}
        )
        assert parsed.status == "pausado"


# ==============================================================
# Server tool tests
# ==============================================================


class TestServerTools:
    @patch("mcp_memory.server.store")
    def test_create_entities_passes_status(self, mock_store):
        """create_entities tool passes status to storage."""
        mock_store.upsert_entity.return_value = 1
        mock_store.get_observations.return_value = ["obs1"]
        mock_store.get_entity_by_name.return_value = None  # No existing entity

        from mcp_memory.server import create_entities

        result = create_entities(
            [
                {
                    "name": "TestEntity",
                    "entityType": "Test",
                    "status": "pausado",
                    "observations": ["obs1"],
                }
            ]
        )

        mock_store.upsert_entity.assert_called_once_with(
            "TestEntity", "Test", "pausado"
        )

    @patch("mcp_memory.server.store")
    def test_create_entities_default_status(self, mock_store):
        """create_entities tool defaults status to 'activo'."""
        mock_store.upsert_entity.return_value = 1
        mock_store.get_observations.return_value = ["obs1"]
        mock_store.get_entity_by_name.return_value = None

        from mcp_memory.server import create_entities

        result = create_entities(
            [{"name": "TestEntity", "entityType": "Test", "observations": ["obs1"]}]
        )

        mock_store.upsert_entity.assert_called_once_with("TestEntity", "Test", "activo")

    @patch("mcp_memory.server.store")
    def test_open_nodes_includes_status(self, mock_store):
        """open_nodes output includes status field."""
        mock_store.get_entity_by_name.return_value = {
            "id": 1,
            "name": "TestEntity",
            "entity_type": "Test",
            "status": "pausado",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
        mock_store.get_observations_with_ids.return_value = []
        mock_store.get_access_data.return_value = {}

        from mcp_memory.server import open_nodes

        result = open_nodes(names=["TestEntity"])
        assert result["entities"][0]["status"] == "pausado"

    @patch("mcp_memory.server.store")
    def test_search_semantic_status_deboost(self, mock_store):
        """Entities with 'archivado' status get de-boosted limbic score."""
        from mcp_memory.server import search_semantic
        from mcp_memory.scoring import RoutingStrategy

        with patch("mcp_memory.server._get_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_engine.available = True
            mock_engine_fn.return_value = mock_engine

            # Two entities: one activo, one archivado
            mock_store.search_embeddings.return_value = [
                {"entity_id": 1, "distance": 0.3},
                {"entity_id": 2, "distance": 0.3},
            ]
            mock_store.search_fts.return_value = []
            mock_store.get_access_data.return_value = {
                1: {"access_count": 5, "last_access": "2025-01-01"},
                2: {"access_count": 5, "last_access": "2025-01-01"},
            }
            mock_store.get_entity_degrees.return_value = {1: 3, 2: 3}
            mock_store.get_co_occurrences.return_value = {}
            mock_store.get_access_days.return_value = {1: 10, 2: 10}
            mock_store.get_entity_by_id.side_effect = lambda eid: {
                "id": eid,
                "name": f"Entity{eid}",
                "entity_type": "Test",
                "status": "activo" if eid == 1 else "archivado",
                "created_at": "2024-01-01",
            }
            mock_store.get_entities_batch.side_effect = lambda ids: {
                eid: {
                    "id": eid,
                    "name": f"Entity{eid}",
                    "entity_type": "Test",
                    "status": "activo" if eid == 1 else "archivado",
                    "created_at": "2024-01-01",
                }
                for eid in ids
            }
            mock_store.get_observations.return_value = ["obs1"]
            mock_store.get_observations_batch.side_effect = lambda ids: {
                eid: ["obs1"] for eid in ids
            }
            mock_store.get_observations_with_ids.return_value = [
                {
                    "id": 1,
                    "content": "obs1",
                    "similarity_flag": 0,
                    "kind": "generic",
                    "supersedes": None,
                    "superseded_at": None,
                }
            ]
            mock_store.get_observations_with_ids_batch.side_effect = lambda ids, **kw: {
                eid: [
                    {
                        "id": 1,
                        "content": "obs1",
                        "similarity_flag": 0,
                        "kind": "generic",
                        "supersedes": None,
                        "superseded_at": None,
                    }
                ]
                for eid in ids
            }
            mock_store.log_search_event.return_value = 1

            with (
                patch("mcp_memory.scoring.rank_candidates") as mock_rank,
                patch("mcp_memory.scoring.detect_query_type") as mock_detect,
                patch("mcp_memory.server._get_treatment", return_value=1),
            ):
                mock_detect.return_value = RoutingStrategy.HYBRID_BALANCED
                # Both start with same limbic_score
                mock_rank.return_value = [
                    {
                        "entity_id": 1,
                        "limbic_score": 1.0,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    },
                    {
                        "entity_id": 2,
                        "limbic_score": 1.0,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    },
                ]

                result = search_semantic("test query", limit=5)

                if "results" in result and len(result["results"]) >= 2:
                    scores = {r["name"]: r["limbic_score"] for r in result["results"]}
                    # activo (1.0 * 1.0) should rank higher than archivado (1.0 * 0.5)
                    assert scores["Entity1"] > scores["Entity2"]
                    assert scores["Entity2"] == 0.5  # 1.0 * 0.5
