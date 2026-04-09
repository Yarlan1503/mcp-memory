"""Tests for Phase 4: Reflections — narrative layer for MCP Memory."""

import pytest

from mcp_memory.storage import MemoryStore


# ============================================================
# 1. Schema — reflections, FTS, and embeddings tables exist
# ============================================================


class TestReflectionsSchema:
    """Verify reflections-related tables are created by init_db."""

    def test_reflections_table_exists(self, store_with_schema):
        """reflections table exists after init_db."""
        rows = store_with_schema.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reflections'"
        ).fetchall()
        assert len(rows) == 1

    def test_reflection_fts_exists(self, store_with_schema):
        """reflection_fts virtual table exists."""
        rows = store_with_schema.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reflection_fts'"
        ).fetchall()
        assert len(rows) == 1

    def test_reflection_embeddings_exists(self, store_with_schema):
        """reflection_embeddings virtual table exists."""
        rows = store_with_schema.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reflection_embeddings'"
        ).fetchall()
        assert len(rows) == 1


# ============================================================
# 2. add_reflection storage
# ============================================================


class TestAddReflection:
    """Tests for MemoryStore.add_reflection."""

    def test_add_reflection_entity(self, store_with_schema):
        """Create reflection on entity — all fields saved."""
        eid = store_with_schema.upsert_entity("TestEnt", "Testing")
        result = store_with_schema.add_reflection(
            "entity", eid, "nolan", "Este proyecto es fascinante", "satisfaccion"
        )
        assert result is not None
        assert result["target_type"] == "entity"
        assert result["target_id"] == eid
        assert result["author"] == "nolan"
        assert result["content"] == "Este proyecto es fascinante"
        assert result["mood"] == "satisfaccion"
        assert result["id"] is not None

    def test_add_reflection_session(self, store_with_schema):
        """Create reflection on session."""
        sid = store_with_schema.upsert_entity("Sesion 2026-01-01", "Sesion")
        result = store_with_schema.add_reflection(
            "session", sid, "sofia", "Sesion productiva"
        )
        assert result is not None
        assert result["target_type"] == "session"

    def test_add_reflection_global(self, store_with_schema):
        """Create global reflection (target_id=NULL)."""
        result = store_with_schema.add_reflection(
            "global", None, "nolan", "Reflexión general sobre el sistema"
        )
        assert result is not None
        assert result["target_type"] == "global"
        assert result["target_id"] is None

    def test_add_reflection_with_mood(self, store_with_schema):
        """Create reflection with mood."""
        eid = store_with_schema.upsert_entity("TestEnt2", "Testing")
        result = store_with_schema.add_reflection(
            "entity", eid, "sofia", "No funciona bien", "frustracion"
        )
        assert result is not None
        assert result["mood"] == "frustracion"

    def test_add_reflection_without_mood(self, store_with_schema):
        """Create reflection without mood — mood should be None."""
        eid = store_with_schema.upsert_entity("TestEnt3", "Testing")
        result = store_with_schema.add_reflection(
            "entity", eid, "nolan", "Algo interesante"
        )
        assert result is not None
        assert result["mood"] is None

    def test_add_reflection_invalid_target_type(self, store_with_schema):
        """Invalid target_type returns None."""
        result = store_with_schema.add_reflection("invalid_type", 1, "nolan", "test")
        assert result is None

    def test_add_reflection_missing_target_id(self, store_with_schema):
        """Entity reflection without target_id returns None."""
        result = store_with_schema.add_reflection("entity", None, "nolan", "test")
        assert result is None


# ============================================================
# 3. Embeddings and FTS
# ============================================================


class TestReflectionEmbeddingsAndFTS:
    """Tests for embedding storage and FTS sync on reflections."""

    def test_reflection_embedding_stored(self, store_with_schema):
        """Creating a reflection generates embedding in reflection_embeddings."""
        if not store_with_schema._vec_loaded:
            pytest.skip("sqlite-vec not loaded")
        eid = store_with_schema.upsert_entity("EmbedTest", "Testing")
        result = store_with_schema.add_reflection(
            "entity", eid, "nolan", "Contenido para embedding"
        )
        assert result is not None
        # Verify embedding exists
        row = store_with_schema.db.execute(
            "SELECT rowid FROM reflection_embeddings WHERE rowid = ?",
            (result["id"],),
        ).fetchone()
        assert row is not None

    def test_reflection_fts_synced(self, store_with_schema):
        """Creating a reflection syncs to reflection_fts."""
        if not store_with_schema._fts_available:
            pytest.skip("FTS5 not available")
        eid = store_with_schema.upsert_entity("FTSTest", "Testing")
        result = store_with_schema.add_reflection(
            "entity", eid, "nolan", "Reflexión única sobre testing"
        )
        assert result is not None
        # Search in FTS
        fts_results = store_with_schema.search_reflection_fts("reflexión testing")
        assert len(fts_results) > 0
        assert any(r["id"] == result["id"] for r in fts_results)

    def test_reflection_fts_text_search(self, store_with_schema):
        """Reflection appears in FTS text search."""
        if not store_with_schema._fts_available:
            pytest.skip("FTS5 not available")
        store_with_schema.add_reflection(
            "global", None, "sofia", "El conocimiento crece con cada sesión"
        )
        store_with_schema.add_reflection(
            "global", None, "nolan", "El debugging es un arte"
        )
        results = store_with_schema.search_reflection_fts("conocimiento")
        assert len(results) > 0


# ============================================================
# 4. get_reflections_for_target
# ============================================================


class TestGetReflectionsForTarget:
    """Tests for MemoryStore.get_reflections_for_target."""

    def test_get_reflections_for_entity(self, store_with_schema):
        """Get reflections for a specific entity."""
        e1 = store_with_schema.upsert_entity("Ent1", "Testing")
        e2 = store_with_schema.upsert_entity("Ent2", "Testing")
        store_with_schema.add_reflection("entity", e1, "nolan", "Reflexión sobre Ent1")
        store_with_schema.add_reflection("entity", e2, "nolan", "Reflexión sobre Ent2")
        store_with_schema.add_reflection(
            "entity", e1, "sofia", "Otra reflexión sobre Ent1"
        )

        refs_e1 = store_with_schema.get_reflections_for_target("entity", e1)
        assert len(refs_e1) == 2

        refs_e2 = store_with_schema.get_reflections_for_target("entity", e2)
        assert len(refs_e2) == 1

    def test_get_reflections_global(self, store_with_schema):
        """Get global reflections."""
        store_with_schema.add_reflection("global", None, "nolan", "Global 1")
        store_with_schema.add_reflection("global", None, "sofia", "Global 2")
        store_with_schema.add_reflection("entity", 999, "nolan", "No global")

        refs = store_with_schema.get_reflections_for_target("global")
        assert len(refs) == 2


# ============================================================
# 5. search_reflection_embeddings — storage layer KNN
# ============================================================


class TestSearchReflectionEmbeddings:
    """Tests for storage-level reflection embedding search."""

    def test_search_reflection_embeddings_knn(self, store_with_schema):
        """KNN search on reflection_embeddings returns relevant results."""
        if not store_with_schema._vec_loaded:
            pytest.skip("sqlite-vec not loaded")

        from mcp_memory.embeddings import EmbeddingEngine, serialize_f32

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        store_with_schema.add_reflection(
            "global",
            None,
            "nolan",
            "El modelo de embeddings funciona correctamente",
        )
        store_with_schema.add_reflection(
            "global",
            None,
            "sofia",
            "La base de datos tiene muchos registros",
        )

        query_vector = engine.encode(["funcionamiento de embeddings"], task="query")
        query_bytes = serialize_f32(query_vector[0])
        results = store_with_schema.search_reflection_embeddings(query_bytes, limit=5)
        assert len(results) > 0

    def test_search_reflections_no_results(self, store_with_schema):
        """Search with no matching reflections returns empty list."""
        if not store_with_schema._vec_loaded:
            pytest.skip("sqlite-vec not loaded")

        from mcp_memory.embeddings import EmbeddingEngine, serialize_f32

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        query_vector = engine.encode(["xyznonexistent"], task="query")
        query_bytes = serialize_f32(query_vector[0])
        results = store_with_schema.search_reflection_embeddings(query_bytes, limit=5)
        assert isinstance(results, list)


# ============================================================
# 6. open_nodes integration — reflections section
# ============================================================


class TestOpenNodesReflections:
    """Tests that open_nodes includes reflections section."""

    def test_open_nodes_includes_reflections(self, store_with_schema, monkeypatch):
        """open_nodes returns reflections section for entities with reflections."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store_with_schema)

        eid = store_with_schema.upsert_entity("ReflectionEntity", "Testing")
        store_with_schema.add_reflection(
            "entity", eid, "nolan", "Una reflexión importante", "insight"
        )

        result = open_nodes(["ReflectionEntity"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 1
        assert "reflections" in entities[0]
        assert len(entities[0]["reflections"]) == 1
        assert entities[0]["reflections"][0]["content"] == "Una reflexión importante"

    def test_open_nodes_no_reflections(self, store_with_schema, monkeypatch):
        """open_nodes for entity without reflections has empty reflections list."""
        import mcp_memory.server as server_module
        from mcp_memory.server import open_nodes

        monkeypatch.setattr(server_module, "store", store_with_schema)

        store_with_schema.upsert_entity("NoRefEntity", "Testing")

        result = open_nodes(["NoRefEntity"])
        assert "error" not in result
        entities = result["entities"]
        assert len(entities) == 1
        assert "reflections" in entities[0]
        assert len(entities[0]["reflections"]) == 0
