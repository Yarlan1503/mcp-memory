"""Tests for Phase 1: kind + supersedes fields on observations."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_kind_supersedes.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


# ==============================================================
# Migration idempotent tests
# ==============================================================


class TestMigrations:
    def test_kind_column_added(self, store):
        """Column 'kind' exists after init_db."""
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "kind" in col_names

    def test_kind_migration_idempotent(self, store):
        """Calling _add_kind_column twice raises no error."""
        store._add_kind_column()
        store._add_kind_column()  # Should be no-op
        # Verify still has the column
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "kind" in col_names

    def test_supersedes_columns_added(self, store):
        """Both supersedes and superseded_at columns exist after init_db."""
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "supersedes" in col_names
        assert "superseded_at" in col_names

    def test_supersedes_migration_idempotent(self, store):
        """Calling _add_supersedes_columns twice raises no error."""
        store._add_supersedes_columns()
        store._add_supersedes_columns()
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "supersedes" in col_names
        assert "superseded_at" in col_names


# ==============================================================
# kind tests
# ==============================================================


class TestKind:
    def test_default_kind_is_generic(self, store):
        """Obs created without kind has kind='generic'."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["hello"])
        rows = store.db.execute(
            "SELECT kind FROM observations WHERE entity_id = ?", (eid,)
        ).fetchall()
        assert rows[0]["kind"] == "generic"

    def test_explicit_kind_saved(self, store):
        """Obs created with kind='hallazgo' has that kind."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["found something"], kind="hallazgo")
        rows = store.db.execute(
            "SELECT kind FROM observations WHERE entity_id = ?", (eid,)
        ).fetchall()
        assert rows[0]["kind"] == "hallazgo"

    def test_add_observations_with_kind(self, store):
        """storage.add_observations accepts kind param and applies to all obs."""
        eid = store.upsert_entity("TestEntity", "Test")
        count = store.add_observations(eid, ["obs1", "obs2", "obs3"], kind="metadata")
        assert count == 3
        rows = store.db.execute(
            "SELECT kind FROM observations WHERE entity_id = ?", (eid,)
        ).fetchall()
        kinds = {r["kind"] for r in rows}
        assert kinds == {"metadata"}

    def test_get_observations_with_ids_includes_kind(self, store):
        """Output of get_observations_with_ids includes kind field."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["obs1"], kind="hallazgo")
        store.add_observations(eid, ["obs2"], kind="generic")
        obs_data = store.get_observations_with_ids(eid)
        assert len(obs_data) == 2
        kinds = {o["kind"] for o in obs_data}
        assert "hallazgo" in kinds
        assert "generic" in kinds


# ==============================================================
# supersedes tests
# ==============================================================


class TestSupersedes:
    def test_supersedes_marks_old_obs(self, store):
        """When supersedes is set, old obs gets superseded_at."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["old observation"])
        old_obs = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()
        old_id = old_obs["id"]

        store.add_observations(eid, ["new observation"], supersedes=old_id)

        # Old obs should have superseded_at set
        old_row = store.db.execute(
            "SELECT superseded_at FROM observations WHERE id = ?", (old_id,)
        ).fetchone()
        assert old_row["superseded_at"] is not None

    def test_supersedes_sets_reference(self, store):
        """New obs has supersedes pointing to old."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["old observation"])
        old_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]

        store.add_observations(eid, ["new observation"], supersedes=old_id)

        # New obs should have supersedes = old_id
        new_row = store.db.execute(
            "SELECT supersedes FROM observations WHERE entity_id = ? ORDER BY id DESC LIMIT 1",
            (eid,),
        ).fetchone()
        assert new_row["supersedes"] == old_id

    def test_supersedes_nonexistent_obs_ignored(self, store):
        """No error if supersedes id doesn't exist — just ignored."""
        eid = store.upsert_entity("TestEntity", "Test")
        # No obs created, supersedes=9999 doesn't exist
        count = store.add_observations(eid, ["new obs"], supersedes=9999)
        assert count == 1

    def test_supersedes_wrong_entity_ignored(self, store):
        """No error if supersedes id belongs to different entity — just ignored."""
        eid1 = store.upsert_entity("Entity1", "Test")
        eid2 = store.upsert_entity("Entity2", "Test")
        store.add_observations(eid1, ["belongs to entity1"])
        other_obs_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid1,)
        ).fetchone()["id"]

        # Try to supersede an obs that belongs to eid1 from eid2
        count = store.add_observations(
            eid2, ["new for entity2"], supersedes=other_obs_id
        )
        assert count == 1
        # The other obs should NOT be superseded
        other_row = store.db.execute(
            "SELECT superseded_at FROM observations WHERE id = ?", (other_obs_id,)
        ).fetchone()
        assert other_row["superseded_at"] is None

    def test_double_supersede_ignored(self, store):
        """Superseding an already-superseded obs is ignored."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["version 1"])
        old_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]

        # First supersede
        store.add_observations(eid, ["version 2"], supersedes=old_id)
        # Second supersede of already-superseded obs — should be ignored
        store.add_observations(eid, ["version 3"], supersedes=old_id)

        # Only version 2 should have supersedes set (version 3's supersedes was ignored)
        rows = store.db.execute(
            "SELECT content, supersedes FROM observations WHERE entity_id = ? ORDER BY id",
            (eid,),
        ).fetchall()
        assert rows[0]["content"] == "version 1"
        assert rows[1]["content"] == "version 2"
        assert rows[1]["supersedes"] == old_id
        # version 3 should NOT have supersedes set
        assert rows[2]["supersedes"] is None

    def test_get_observations_excludes_superseded(self, store):
        """Default behavior: superseded observations are excluded."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["old obs"])
        old_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]
        store.add_observations(eid, ["new obs"], supersedes=old_id)

        obs = store.get_observations(eid)
        assert obs == ["new obs"]

    def test_get_observations_includes_superseded(self, store):
        """With exclude_superseded=False, superseded observations are included."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["old obs"])
        old_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]
        store.add_observations(eid, ["new obs"], supersedes=old_id)

        obs = store.get_observations(eid, exclude_superseded=False)
        assert "old obs" in obs
        assert "new obs" in obs

    def test_superseded_excluded_from_fts(self, store):
        """Verify FTS sync uses exclude_superseded — superseded obs not in FTS."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["unique_old_content_xyz"])
        old_id = store.db.execute(
            "SELECT id FROM observations WHERE entity_id = ?", (eid,)
        ).fetchone()["id"]
        store.add_observations(eid, ["new_content_abc"], supersedes=old_id)

        # Force FTS resync
        store._sync_fts(eid)

        # If FTS is available, superseded obs should not be searchable
        if store._fts_available:
            fts_results = store.search_fts("unique_old_content_xyz")
            assert len(fts_results) == 0
            fts_results = store.search_fts("new_content_abc")
            assert len(fts_results) == 1


# ==============================================================
# Server tool tests
# ==============================================================


class TestServerTools:
    """Tests for server.py tool functions using mock."""

    def _setup_store_mock(self, mock_store):
        """Configure common mock behaviors."""
        mock_store.get_entity_by_name.return_value = {
            "id": 1,
            "name": "TestEntity",
            "entity_type": "Test",
        }
        mock_store.get_observations.return_value = ["obs1", "obs2"]
        mock_store.get_observations_with_ids.return_value = [
            {
                "id": 1,
                "content": "obs1",
                "similarity_flag": 0,
                "kind": "generic",
                "supersedes": None,
                "superseded_at": None,
            },
            {
                "id": 2,
                "content": "obs2",
                "similarity_flag": 0,
                "kind": "generic",
                "supersedes": None,
                "superseded_at": None,
            },
        ]

    @patch("mcp_memory.server.store")
    def test_open_nodes_default_excludes_superseded(self, mock_store):
        """open_nodes by default calls get_observations_with_ids with exclude_superseded=True."""
        self._setup_store_mock(mock_store)

        from mcp_memory.server import open_nodes

        result = open_nodes(names=["TestEntity"])

        mock_store.get_observations_with_ids.assert_called_with(
            1, exclude_superseded=True
        )
        assert "entities" in result

    @patch("mcp_memory.server.store")
    def test_open_nodes_include_superseded(self, mock_store):
        """open_nodes with include_superseded=True passes exclude_superseded=False."""
        self._setup_store_mock(mock_store)

        from mcp_memory.server import open_nodes

        result = open_nodes(names=["TestEntity"], include_superseded=True)

        mock_store.get_observations_with_ids.assert_called_with(
            1, exclude_superseded=False
        )

    @patch("mcp_memory.server.store")
    def test_open_nodes_filter_by_kinds(self, mock_store):
        """open_nodes with kinds filter only includes matching observations."""
        mock_store.get_entity_by_name.return_value = {
            "id": 1,
            "name": "TestEntity",
            "entity_type": "Test",
        }
        mock_store.get_observations_with_ids.return_value = [
            {
                "id": 1,
                "content": "obs1",
                "similarity_flag": 0,
                "kind": "hallazgo",
                "supersedes": None,
                "superseded_at": None,
            },
            {
                "id": 2,
                "content": "obs2",
                "similarity_flag": 0,
                "kind": "generic",
                "supersedes": None,
                "superseded_at": None,
            },
        ]

        from mcp_memory.server import open_nodes

        result = open_nodes(names=["TestEntity"], kinds=["hallazgo"])

        # Only hallazgo observation should be in output
        obs = result["entities"][0]["observations"]
        assert len(obs) == 1
        assert obs[0]["kind"] == "hallazgo"

    @patch("mcp_memory.server.store")
    def test_add_observations_tool_accepts_kind(self, mock_store):
        """add_observations tool accepts kind parameter and passes to store."""
        mock_store.get_entity_by_name.return_value = {
            "id": 1,
            "name": "TestEntity",
            "entity_type": "Test",
        }
        mock_store.get_observations.return_value = ["new obs"]

        from mcp_memory.server import add_observations

        result = add_observations(
            name="TestEntity", observations=["new obs"], kind="metadata"
        )

        mock_store.add_observations.assert_called_once_with(
            1, ["new obs"], kind="metadata", supersedes=None
        )
        assert "entity" in result

    @patch("mcp_memory.server.store")
    def test_search_semantic_metadata_deboost(self, mock_store):
        """Entities with majority metadata observations get de-boosted limbic score."""
        from mcp_memory.server import search_semantic
        from mcp_memory.scoring import RoutingStrategy

        # Mock the engine
        with patch("mcp_memory.server._get_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_engine.available = True
            mock_engine.encode.return_value = MagicMock()  # query vector
            mock_engine_fn.return_value = mock_engine

            # Mock storage calls
            mock_store.search_embeddings.return_value = [
                {"entity_id": 1, "distance": 0.3}
            ]
            mock_store.search_fts.return_value = []
            mock_store.get_access_data.return_value = {
                1: {"access_count": 5, "last_access": "2025-01-01"}
            }
            mock_store.get_entity_degrees.return_value = {1: 3}
            mock_store.get_co_occurrences.return_value = {}
            mock_store.get_access_days.return_value = {1: 10}
            mock_store.get_entity_by_id.return_value = {
                "id": 1,
                "name": "MetaEntity",
                "entity_type": "Test",
                "created_at": "2024-01-01",
            }
            mock_store.get_entities_batch.return_value = {
                1: {
                    "id": 1,
                    "name": "MetaEntity",
                    "entity_type": "Test",
                    "created_at": "2024-01-01",
                }
            }
            mock_store.get_observations.return_value = ["meta1", "meta2", "meta3"]
            mock_store.get_observations_batch.return_value = {
                1: ["meta1", "meta2", "meta3"]
            }
            # 4 metadata, 1 generic → 80% metadata → de-boosted
            mock_store.get_observations_with_ids.return_value = [
                {
                    "id": 1,
                    "content": "m1",
                    "similarity_flag": 0,
                    "kind": "metadata",
                    "supersedes": None,
                    "superseded_at": None,
                },
                {
                    "id": 2,
                    "content": "m2",
                    "similarity_flag": 0,
                    "kind": "metadata",
                    "supersedes": None,
                    "superseded_at": None,
                },
                {
                    "id": 3,
                    "content": "m3",
                    "similarity_flag": 0,
                    "kind": "metadata",
                    "supersedes": None,
                    "superseded_at": None,
                },
                {
                    "id": 4,
                    "content": "m4",
                    "similarity_flag": 0,
                    "kind": "metadata",
                    "supersedes": None,
                    "superseded_at": None,
                },
                {
                    "id": 5,
                    "content": "g1",
                    "similarity_flag": 0,
                    "kind": "generic",
                    "supersedes": None,
                    "superseded_at": None,
                },
            ]
            mock_store.get_observations_with_ids_batch.return_value = {
                1: [
                    {
                        "id": 1,
                        "content": "m1",
                        "similarity_flag": 0,
                        "kind": "metadata",
                        "supersedes": None,
                        "superseded_at": None,
                    },
                    {
                        "id": 2,
                        "content": "m2",
                        "similarity_flag": 0,
                        "kind": "metadata",
                        "supersedes": None,
                        "superseded_at": None,
                    },
                    {
                        "id": 3,
                        "content": "m3",
                        "similarity_flag": 0,
                        "kind": "metadata",
                        "supersedes": None,
                        "superseded_at": None,
                    },
                    {
                        "id": 4,
                        "content": "m4",
                        "similarity_flag": 0,
                        "kind": "metadata",
                        "supersedes": None,
                        "superseded_at": None,
                    },
                    {
                        "id": 5,
                        "content": "g1",
                        "similarity_flag": 0,
                        "kind": "generic",
                        "supersedes": None,
                        "superseded_at": None,
                    },
                ]
            }
            mock_store.log_search_event.return_value = 1

            # Patch scoring imports that happen inside search_semantic
            with (
                patch("mcp_memory.scoring.reciprocal_rank_fusion") as mock_rrf,
                patch("mcp_memory.scoring.rank_candidates") as mock_rank,
                patch("mcp_memory.scoring.detect_query_type") as mock_detect,
                patch("mcp_memory.server._get_treatment", return_value=1),
            ):
                mock_detect.return_value = RoutingStrategy.HYBRID_BALANCED
                mock_rank.return_value = [
                    {
                        "entity_id": 1,
                        "limbic_score": 1.0,
                        "importance": 0.5,
                        "temporal_factor": 1.0,
                        "cooc_boost": 0.0,
                    }
                ]

                result = search_semantic("test query", limit=5)

                if "results" in result and result["results"]:
                    # The limbic_score should be de-boosted (1.0 * 0.7 = 0.7)
                    assert result["results"][0]["limbic_score"] == 0.7


# ==============================================================
# Backward compatibility
# ==============================================================


class TestBackwardCompatibility:
    def test_existing_get_observations_calls_still_work(self, store):
        """Calling get_observations without new params works as before."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["obs1", "obs2"])
        obs = store.get_observations(eid)
        assert obs == ["obs1", "obs2"]

    def test_existing_add_observations_calls_still_work(self, store):
        """Calling add_observations without kind/supersedes works as before."""
        eid = store.upsert_entity("TestEntity", "Test")
        count = store.add_observations(eid, ["obs1", "obs2"])
        assert count == 2
        obs = store.get_observations(eid)
        assert len(obs) == 2

    def test_multiple_kinds_same_entity(self, store):
        """Entity can have observations with different kinds."""
        eid = store.upsert_entity("TestEntity", "Test")
        store.add_observations(eid, ["generic obs"], kind="generic")
        store.add_observations(eid, ["hallazgo obs"], kind="hallazgo")
        store.add_observations(eid, ["metadata obs"], kind="metadata")

        obs_data = store.get_observations_with_ids(eid)
        assert len(obs_data) == 3
        kinds = {o["kind"] for o in obs_data}
        assert kinds == {"generic", "hallazgo", "metadata"}
