"""Tests for semantic deduplication in add_observations and find_duplicate_observations."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore
from mcp_memory.embeddings import EmbeddingEngine, DIMENSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_pair(cos_sim: float) -> tuple[np.ndarray, np.ndarray]:
    """Create two L2-normalised vectors with a specific cosine similarity."""
    rng = np.random.RandomState(42)
    v1 = rng.randn(DIMENSION).astype(np.float32)
    v1 = v1 / np.linalg.norm(v1)

    noise = rng.randn(DIMENSION).astype(np.float32)
    # Orthogonalise noise w.r.t. v1
    noise = noise - np.dot(noise, v1) * v1
    noise = noise / np.linalg.norm(noise)

    sin_sim = np.sqrt(max(0.0, 1.0 - cos_sim**2))
    v2 = cos_sim * v1 + sin_sim * noise
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2


class FakeEmbeddingEngine:
    """Controllable mock embedding engine for testing."""

    def __init__(self):
        self.available = True
        self._vectors: dict[str, np.ndarray] = {}

    def set_vector(self, text: str, vector: np.ndarray) -> None:
        self._vectors[text] = vector

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        result = []
        for t in texts:
            if t in self._vectors:
                result.append(self._vectors[t])
            else:
                # Random unique vector
                rng = np.random.RandomState(hash(t) % (2**31))
                v = rng.randn(DIMENSION).astype(np.float32)
                v = v / np.linalg.norm(v)
                self._vectors[t] = v
                result.append(v)
        return np.array(result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_semantic.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """Reset EmbeddingEngine singleton before/after each test."""
    EmbeddingEngine.reset()
    yield
    EmbeddingEngine.reset()


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestMigration:
    def test_similarity_flag_column_added(self, store):
        """similarity_flag column exists after init_db()."""
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "similarity_flag" in col_names

    def test_migration_is_idempotent(self, store):
        """Calling _add_similarity_flag_column() twice does not raise."""
        store._add_similarity_flag_column()
        store._add_similarity_flag_column()
        # Column should still be there
        cols = store.db.execute("PRAGMA table_info(observations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "similarity_flag" in col_names

    def test_default_flag_is_zero(self, store):
        """Existing rows get similarity_flag = 0 by default."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["some observation"])
        rows = store.db.execute(
            "SELECT similarity_flag FROM observations WHERE entity_id = ?", (eid,)
        ).fetchall()
        assert rows[0]["similarity_flag"] == 0


# ---------------------------------------------------------------------------
# add_observations semantic dedup
# ---------------------------------------------------------------------------


class TestAddObservationsSemanticDedup:
    def test_add_obs_no_flag_when_unique(self, store):
        """Observation new and different → flag = 0."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["existing observation"])

        # Create engine that returns orthogonal vectors (cosine ≈ 0)
        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.1)  # Very different
        fake.set_vector("existing observation", v1)
        fake.set_vector("brand new unique observation", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, ["brand new unique observation"])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "brand new unique observation"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 0

    def test_add_obs_flagged_when_similar(self, store):
        """Observation similar to existing → flag = 1."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["The cat sat on the mat"])

        # Create engine that returns vectors with cosine = 0.90
        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.90)
        fake.set_vector("The cat sat on the mat", v1)
        fake.set_vector("A cat was sitting on a mat", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, ["A cat was sitting on a mat"])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "A cat was sitting on a mat"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 1

    def test_add_obs_no_engine_fallback(self, store):
        """Without embedding engine → flag = 0, no crash."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["existing observation"])

        # Engine returns None (unavailable)
        with patch.object(store, "_get_embedding_engine", return_value=None):
            count = store.add_observations(eid, ["new observation without engine"])

        assert count == 1
        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "new observation without engine"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["similarity_flag"] == 0

    def test_add_obs_first_observation_no_flag(self, store):
        """First observation for entity → flag = 0 (nothing to compare)."""
        eid = store.upsert_entity("Test", "Type")

        fake = FakeEmbeddingEngine()
        with patch.object(store, "_get_embedding_engine", return_value=fake):
            count = store.add_observations(eid, ["first observation ever"])

        assert count == 1
        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "first observation ever"),
        ).fetchall()
        assert rows[0]["similarity_flag"] == 0

    def test_add_obs_exact_duplicate_still_skipped(self, store):
        """Exact duplicates still skipped even with engine available."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["exact text"])

        fake = FakeEmbeddingEngine()
        with patch.object(store, "_get_embedding_engine", return_value=fake):
            count = store.add_observations(eid, ["exact text"])

        assert count == 0  # Skipped

    def test_add_obs_boundary_threshold(self, store):
        """Observation at exactly 0.85 threshold → flag = 1."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["boundary test"])

        fake = FakeEmbeddingEngine()
        # Use 0.851 to avoid float32 precision issues at exact 0.85 boundary
        v1, v2 = _make_vector_pair(0.851)
        fake.set_vector("boundary test", v1)
        fake.set_vector("similar at threshold", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, ["similar at threshold"])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "similar at threshold"),
        ).fetchall()
        assert rows[0]["similarity_flag"] == 1

    def test_add_obs_below_threshold(self, store):
        """Observation at 0.84 (below 0.85) → flag = 0."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["below threshold"])

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.84)  # Just below threshold
        fake.set_vector("below threshold", v1)
        fake.set_vector("not quite similar", v2)

        with patch.object(store, "_get_embedding_engine", return_value=fake):
            store.add_observations(eid, ["not quite similar"])

        rows = store.db.execute(
            "SELECT similarity_flag FROM observations "
            "WHERE entity_id = ? AND content = ?",
            (eid, "not quite similar"),
        ).fetchall()
        assert rows[0]["similarity_flag"] == 0


# ---------------------------------------------------------------------------
# get_observations_with_ids
# ---------------------------------------------------------------------------


class TestGetObservationsWithIds:
    def test_returns_all_fields(self, store):
        """Returns id, content, and similarity_flag."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs1", "obs2"])

        result = store.get_observations_with_ids(eid)
        assert len(result) == 2
        for obs in result:
            assert "id" in obs
            assert "content" in obs
            assert "similarity_flag" in obs
            assert obs["similarity_flag"] == 0  # Default

    def test_returns_empty_for_no_observations(self, store):
        """Returns empty list when entity has no observations."""
        eid = store.upsert_entity("Test", "Type")
        result = store.get_observations_with_ids(eid)
        assert result == []


# ---------------------------------------------------------------------------
# find_duplicate_observations tool
# ---------------------------------------------------------------------------


class TestFindDuplicateObservations:
    def _import_tool(self):
        """Lazy import the tool function."""
        from mcp_memory.server import find_duplicate_observations

        return find_duplicate_observations

    def test_returns_pairs_for_similar_obs(self, store):
        """Tool returns pairs with similarity >= threshold."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["Python programming language"])

        # Add a flagged similar observation directly
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 1)",
            (eid, "Python is a programming language"),
        )
        store.db.commit()

        fake = FakeEmbeddingEngine()
        v1, v2 = _make_vector_pair(0.92)
        fake.set_vector("Python programming language", v1)
        fake.set_vector("Python is a programming language", v2)

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85)

        assert "clusters" in result
        assert result["total_observations"] == 2
        assert result["duplicates_found"] >= 1

    def test_empty_entity_returns_empty_clusters(self, store):
        """Entity with < 2 observations → empty clusters."""
        store.upsert_entity("Test", "Type")

        fake = FakeEmbeddingEngine()
        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85)

        assert result["clusters"] == []
        assert result["total_observations"] == 0

    def test_nonexistent_entity_returns_error(self, store):
        """Non-existent entity → error dict."""
        with patch("mcp_memory.server.store", store):
            tool = self._import_tool()
            result = tool(entity_name="NonExistent", threshold=0.85)
        assert "error" in result

    def test_no_engine_returns_error(self, store):
        """Without embedding engine → error dict."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs1", "obs2", "obs3"])

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=None),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85)

        assert "error" in result
        assert "Embedding model not available" in result["error"]

    def test_cluster_groups_transitive_similar(self, store):
        """A~B and B~C should cluster together even if A~C < threshold."""
        eid = store.upsert_entity("Test", "Type")
        store.add_observations(eid, ["obs A"])
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 0)",
            (eid, "obs B"),
        )
        store.db.execute(
            "INSERT INTO observations (entity_id, content, similarity_flag) VALUES (?, ?, 0)",
            (eid, "obs C"),
        )
        store.db.commit()

        fake = FakeEmbeddingEngine()
        # A~B = 0.90, B~C = 0.90, but A~C will be lower
        vA, vB = _make_vector_pair(0.90)
        vB2, vC = _make_vector_pair(0.90)
        fake.set_vector("obs A", vA)
        fake.set_vector("obs B", vB)  # B is similar to both but different vectors
        fake.set_vector("obs C", vC)

        with (
            patch("mcp_memory.server.store", store),
            patch("mcp_memory.server._get_engine", return_value=fake),
        ):
            tool = self._import_tool()
            result = tool(entity_name="Test", threshold=0.85)

        # At minimum, A~B and B~C should be pairs
        assert result["duplicates_found"] >= 2
