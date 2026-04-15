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

    def test_add_reflection_session_without_id(self, store_with_schema):
        """Session reflection without target_id is valid (like global)."""
        result = store_with_schema.add_reflection(
            "session", None, "sofia", "Sesión productiva", "satisfaccion"
        )
        assert result is not None
        assert result["target_type"] == "session"
        assert result["target_id"] is None
        assert result["author"] == "sofia"
        assert result["content"] == "Sesión productiva"
        assert result["mood"] == "satisfaccion"

    def test_add_reflection_session_with_id(self, store_with_schema):
        """Session reflection with specific target_id also works."""
        sid = store_with_schema.upsert_entity("Sesion 2026-01-01", "Sesion")
        result = store_with_schema.add_reflection(
            "session", sid, "sofia", "Sesión con ID específico"
        )
        assert result is not None
        assert result["target_type"] == "session"
        assert result["target_id"] == sid

    def test_add_reflection_entity_still_requires_id(self, store_with_schema):
        """Entity reflection still requires target_id (no regression)."""
        result = store_with_schema.add_reflection("entity", None, "nolan", "test")
        assert result is None

    def test_add_reflection_relation_still_requires_id(self, store_with_schema):
        """Relation reflection still requires target_id (no regression)."""
        result = store_with_schema.add_reflection("relation", None, "nolan", "test")
        assert result is None

    def test_add_reflection_global_rejects_target_id(self, store_with_schema):
        """Global reflection with target_id should return None."""
        result = store_with_schema.add_reflection("global", 123, "nolan", "test")
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


# ============================================================
# 7. end_relation MCP tool tests
# ============================================================


class TestEndRelationTool:
    """Tests for the end_relation MCP tool."""

    def test_end_relation_active(self, store_with_schema, monkeypatch):
        """Expiring an active relation sets active=0 and ended_at."""
        import mcp_memory.server as server_module
        from mcp_memory.server import end_relation

        monkeypatch.setattr(server_module, "store", store_with_schema)

        e1 = store_with_schema.upsert_entity("Alpha", "T")
        e2 = store_with_schema.upsert_entity("Beta", "T")
        store_with_schema.create_relation(e1, e2, "producido_por")

        rel_row = store_with_schema.db.execute(
            "SELECT id FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        rel_id = rel_row["id"]

        result = end_relation(rel_id)
        assert "error" not in result
        assert "relation" in result
        assert result["relation"]["active"] is False
        assert result["relation"]["ended_at"] is not None
        assert result["relation"]["from"] == "Alpha"
        assert result["relation"]["to"] == "Beta"
        assert result["relation"]["relation_type"] == "producido_por"

    def test_end_relation_already_inactive(self, store_with_schema, monkeypatch):
        """Expiring an already-inactive relation returns notice, not error."""
        import mcp_memory.server as server_module
        from mcp_memory.server import end_relation

        monkeypatch.setattr(server_module, "store", store_with_schema)

        e1 = store_with_schema.upsert_entity("X", "T")
        e2 = store_with_schema.upsert_entity("Y", "T")
        store_with_schema.create_relation(e1, e2, "test_type")

        rel_row = store_with_schema.db.execute(
            "SELECT id FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        rel_id = rel_row["id"]

        # Expire once
        result1 = end_relation(rel_id)
        assert "error" not in result1

        # Expire again — should get notice
        result2 = end_relation(rel_id)
        assert "error" not in result2
        assert "notice" in result2
        assert "already inactive" in result2["notice"].lower()

    def test_end_relation_with_inverse(self, store_with_schema, monkeypatch):
        """Expiring contiene also expires the parte_de inverse."""
        import mcp_memory.server as server_module
        from mcp_memory.server import end_relation

        monkeypatch.setattr(server_module, "store", store_with_schema)

        e1 = store_with_schema.upsert_entity("Parent", "T")
        e2 = store_with_schema.upsert_entity("Child", "T")
        store_with_schema.create_relation(e1, e2, "contiene")

        # Find the contiene relation
        contiene_row = store_with_schema.db.execute(
            "SELECT id FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = 'contiene'",
            (e1, e2),
        ).fetchone()
        contiene_id = contiene_row["id"]

        # Find the parte_de inverse
        parte_de_row = store_with_schema.db.execute(
            "SELECT id, active FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = 'parte_de'",
            (e2, e1),
        ).fetchone()
        assert parte_de_row is not None
        parte_de_id = parte_de_row["id"]

        # Expire the contiene
        result = end_relation(contiene_id)
        assert "error" not in result
        assert "inverse_expired_id" in result
        assert result["inverse_expired_id"] == parte_de_id

        # Verify both are now inactive
        check_contiene = store_with_schema.db.execute(
            "SELECT active, ended_at FROM relations WHERE id = ?", (contiene_id,)
        ).fetchone()
        assert check_contiene["active"] == 0
        assert check_contiene["ended_at"] is not None

        check_parte = store_with_schema.db.execute(
            "SELECT active, ended_at FROM relations WHERE id = ?", (parte_de_id,)
        ).fetchone()
        assert check_parte["active"] == 0
        assert check_parte["ended_at"] is not None

    def test_end_relation_not_found(self, store_with_schema, monkeypatch):
        """Expiring a non-existent relation ID returns error."""
        import mcp_memory.server as server_module
        from mcp_memory.server import end_relation

        monkeypatch.setattr(server_module, "store", store_with_schema)

        result = end_relation(99999)
        assert "error" in result
        assert "not found" in result["error"].lower()


# ============================================================
# 8. search_reflections recency scoring
# ============================================================


class TestSearchReflectionsRecency:
    """Tests for recency scoring in search_reflections."""

    def test_recent_reflection_ranks_higher(self, store_with_schema, monkeypatch):
        """A newer reflection ranks higher than an older one for the same query."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections
        from mcp_memory.embeddings import EmbeddingEngine

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        monkeypatch.setattr(server_module, "store", store_with_schema)

        # Create two reflections with similar content but different timestamps
        # Old reflection — set created_at to 1 year ago
        old = store_with_schema.add_reflection(
            "global",
            None,
            "nolan",
            "El sistema de memoria artificial es fundamental para la inteligencia",
            "insight",
        )
        # Manually set created_at to 1 year ago
        store_with_schema.db.execute(
            "UPDATE reflections SET created_at = datetime('now', '-365 days') WHERE id = ?",
            (old["id"],),
        )
        store_with_schema.db.commit()

        # Recent reflection — just created (now)
        recent = store_with_schema.add_reflection(
            "global",
            None,
            "sofia",
            "El sistema de memoria artificial es clave para la inteligencia",
        )

        result = search_reflections("memoria artificial inteligencia")
        assert "error" not in result
        assert len(result["results"]) >= 2

        # The recent one should rank first (higher score due to recency)
        ids = [r["id"] for r in result["results"]]
        assert recent["id"] in ids
        assert old["id"] in ids

        # Find scores
        scores = {r["id"]: r["score"] for r in result["results"]}
        assert scores[recent["id"]] > scores[old["id"]]

    def test_same_day_reflections_similar_score(self, store_with_schema, monkeypatch):
        """Reflections created on the same day have very similar recency scores."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections
        from mcp_memory.embeddings import EmbeddingEngine

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        monkeypatch.setattr(server_module, "store", store_with_schema)

        # Create two reflections with identical timestamps
        r1 = store_with_schema.add_reflection(
            "global", None, "nolan", "La paciencia es una virtud del programador"
        )
        r2 = store_with_schema.add_reflection(
            "global", None, "sofia", "La paciencia del programador es admirable"
        )

        result = search_reflections("paciencia programador")
        assert "error" not in result
        assert len(result["results"]) >= 2

        # Both should be present and have similar scores
        scores = {r["id"]: r["score"] for r in result["results"]}
        if r1["id"] in scores and r2["id"] in scores:
            # Scores should be within 1% of each other (same hour essentially)
            ratio = min(scores[r1["id"]], scores[r2["id"]]) / max(
                scores[r1["id"]], scores[r2["id"]]
            )
            assert ratio > 0.99, f"Same-day scores too different: {scores}"

    def test_search_reflections_no_results(self, store_with_schema, monkeypatch):
        """Search with no matching reflections returns empty results."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections

        monkeypatch.setattr(server_module, "store", store_with_schema)

        result = search_reflections("xyznonexistent12345")
        assert "error" not in result
        assert result["results"] == []

    def test_search_reflections_finds_session_without_id(
        self, store_with_schema, monkeypatch
    ):
        """search_reflections can find session reflections without target_id."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections

        monkeypatch.setattr(server_module, "store", store_with_schema)

        store_with_schema.add_reflection(
            "session", None, "sofia", "Reflexión sobre la sesión de hoy", "satisfaccion"
        )
        result = search_reflections("sesión de hoy")
        assert "error" not in result
        assert len(result["results"]) >= 1
        found = result["results"][0]
        assert found["target_type"] == "session"
        assert found["target_id"] is None


# ============================================================
# 9. Backward compatibility
# ============================================================


class TestBackwardCompatibility:
    """Ensure existing functionality still works after changes."""

    def test_search_reflections_basic_functionality(
        self, store_with_schema, monkeypatch
    ):
        """search_reflections still works with basic query after changes."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections
        from mcp_memory.embeddings import EmbeddingEngine

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        monkeypatch.setattr(server_module, "store", store_with_schema)

        # Add a reflection
        store_with_schema.add_reflection(
            "global", None, "nolan", "Las pruebas son la base del software confiable"
        )

        result = search_reflections("pruebas software")
        assert "error" not in result
        assert len(result["results"]) >= 1
        assert "score" in result["results"][0]
        assert "created_at" in result["results"][0]

    def test_preexisting_reflections_searchable(self, store_with_schema, monkeypatch):
        """Reflections created before the recency change are still searchable."""
        import mcp_memory.server as server_module
        from mcp_memory.server import search_reflections
        from mcp_memory.embeddings import EmbeddingEngine

        engine = EmbeddingEngine.get_instance()
        if not engine or not engine.available:
            pytest.skip("Embedding model not available")

        monkeypatch.setattr(server_module, "store", store_with_schema)

        # Create a reflection and set its timestamp to long ago
        ref = store_with_schema.add_reflection(
            "global", None, "sofia", "La retrocompatibilidad es esencial"
        )
        store_with_schema.db.execute(
            "UPDATE reflections SET created_at = datetime('now', '-180 days') WHERE id = ?",
            (ref["id"],),
        )
        store_with_schema.db.commit()

        result = search_reflections("retrocompatibilidad")
        assert "error" not in result
        assert len(result["results"]) >= 1
        assert any(r["id"] == ref["id"] for r in result["results"])
