"""Tests for Phase 3: relations vigency, context, inverses, and normalization."""

import pytest

from mcp_memory.storage import MemoryStore, INVERSE_RELATIONS, LEGACY_RELATION_TYPES


@pytest.fixture
def store(tmp_path):
    """MemoryStore with schema initialized."""
    db_path = str(tmp_path / "test_v3.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


# ------------------------------------------------------------------ #
# 1. Migration tests
# ------------------------------------------------------------------ #


class TestMigration:
    def test_relation_fields_exist(self, store):
        """Columns context, active, ended_at exist in relations table."""
        cols = store.db.execute("PRAGMA table_info(relations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "context" in col_names
        assert "active" in col_names
        assert "ended_at" in col_names

    def test_relation_migration_idempotent(self, store):
        """Running migration twice does not raise errors."""
        store._migrate_relation_fields()
        store._migrate_relation_fields()
        # Should still have the columns
        cols = store.db.execute("PRAGMA table_info(relations)").fetchall()
        col_names = {c["name"] for c in cols}
        assert "context" in col_names

    def test_relation_defaults(self, store):
        """New relation has active=1, context=NULL, ended_at=NULL."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type")
        row = store.db.execute(
            "SELECT context, active, ended_at FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["context"] is None
        assert row["active"] == 1
        assert row["ended_at"] is None


# ------------------------------------------------------------------ #
# 2. Context tests
# ------------------------------------------------------------------ #


class TestContext:
    def test_create_relation_with_context(self, store):
        """Context is saved when provided."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type", context="some reason")
        row = store.db.execute(
            "SELECT context FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["context"] == "some reason"

    def test_create_relation_without_context(self, store):
        """Context is NULL when not provided."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type")
        row = store.db.execute(
            "SELECT context FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["context"] is None


# ------------------------------------------------------------------ #
# 3. Vigency tests
# ------------------------------------------------------------------ #


class TestVigency:
    def test_new_relation_active_by_default(self, store):
        """Newly created relation has active=1."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type")
        row = store.db.execute(
            "SELECT active FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["active"] == 1

    def test_end_relation(self, store):
        """_end_relation sets active=0 and ended_at."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type")
        rel_id = store.db.execute(
            "SELECT id FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()["id"]

        result = store._end_relation(rel_id)
        assert result is True

        row = store.db.execute(
            "SELECT active, ended_at FROM relations WHERE id = ?", (rel_id,)
        ).fetchone()
        assert row["active"] == 0
        assert row["ended_at"] is not None

    def test_end_relation_not_found(self, store):
        """_end_relation returns False for non-existent relation."""
        result = store._end_relation(99999)
        assert result is False

    def test_get_relations_includes_active_status(self, store):
        """get_all_relations and get_relations_for_entity include active field."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "test_type")

        # get_all_relations
        all_rels = store.get_all_relations()
        assert len(all_rels) >= 1
        rel = next(r for r in all_rels if r["from"] == "A" and r["to"] == "B")
        assert "active" in rel
        assert rel["active"] == 1
        assert "context" in rel
        assert "ended_at" in rel

        # get_relations_for_entity
        entity_rels = store.get_relations_for_entity(e1)
        assert len(entity_rels) >= 1
        erel = entity_rels[0]
        assert "active" in erel
        assert erel["active"] == 1
        assert "context" in erel
        assert "ended_at" in erel


# ------------------------------------------------------------------ #
# 4. Inverse relation tests
# ------------------------------------------------------------------ #


class TestInverseRelations:
    def test_contiene_creates_parte_de_inverse(self, store):
        """Creating contiene(A,B) auto-creates parte_de(B,A)."""
        e1 = store.upsert_entity("Parent", "T")
        e2 = store.upsert_entity("Child", "T")
        store.create_relation(e1, e2, "contiene")

        # Check inverse exists
        row = store.db.execute(
            "SELECT context FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
            (e2, e1, "parte_de"),
        ).fetchone()
        assert row is not None
        assert "inversa" in row["context"].lower() or "Inversa" in row["context"]

    def test_parte_de_creates_contiene_inverse(self, store):
        """Creating parte_de(B,A) auto-creates contiene(A,B)."""
        e1 = store.upsert_entity("Parent", "T")
        e2 = store.upsert_entity("Child", "T")
        store.create_relation(e2, e1, "parte_de")

        # Check inverse exists
        row = store.db.execute(
            "SELECT context FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
            (e1, e2, "contiene"),
        ).fetchone()
        assert row is not None
        assert "inversa" in row["context"].lower() or "Inversa" in row["context"]

    def test_no_inverse_for_other_types(self, store):
        """Non-inverse types don't auto-create anything."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.create_relation(e1, e2, "producido_por")

        # Should only be 1 relation
        count = store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert count == 1

    def test_duplicate_inverse_ignored(self, store):
        """Creating an inverse when it already exists does not duplicate."""
        e1 = store.upsert_entity("Parent", "T")
        e2 = store.upsert_entity("Child", "T")
        store.create_relation(e1, e2, "contiene")

        # Now explicitly try to create the inverse that was auto-created
        result = store.create_relation(e2, e1, "parte_de")
        assert result is False  # Already exists

        # Verify only 2 relations total (original + auto-inverse)
        count = store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert count == 2


# ------------------------------------------------------------------ #
# 5. Relation type normalization tests
# ------------------------------------------------------------------ #


class TestRelationTypeNormalization:
    def test_continua_migrated_to_contribuye_a(self, store):
        """Legacy 'continua' type is migrated to 'contribuye_a' with context."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        # Insert a legacy relation directly (bypassing create_relation)
        store.db.execute(
            "INSERT INTO relations (from_entity, to_entity, relation_type) VALUES (?, ?, 'continua')",
            (e1, e2),
        )
        store.db.commit()

        # Run migration
        store._migrate_relation_types()

        # Verify migrated
        row = store.db.execute(
            "SELECT relation_type, context FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["relation_type"] == "contribuye_a"
        assert row["context"] == "sesión continuación"

    def test_documentado_en_migrated_to_producido_por(self, store):
        """Legacy 'documentado_en' type is migrated to 'producido_por' with context."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        store.db.execute(
            "INSERT INTO relations (from_entity, to_entity, relation_type) VALUES (?, ?, 'documentado_en')",
            (e1, e2),
        )
        store.db.commit()

        store._migrate_relation_types()

        row = store.db.execute(
            "SELECT relation_type, context FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        assert row["relation_type"] == "producido_por"
        assert row["context"] == "documentado en"

    def test_existing_context_preserved(self, store):
        """If context already exists, migration does not overwrite it."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        # Insert with existing context
        store.db.execute(
            "INSERT INTO relations (from_entity, to_entity, relation_type, context) VALUES (?, ?, 'continua', 'original context')",
            (e1, e2),
        )
        store.db.commit()

        store._migrate_relation_types()

        row = store.db.execute(
            "SELECT relation_type, context FROM relations WHERE from_entity = ? AND to_entity = ?",
            (e1, e2),
        ).fetchone()
        # Type should still be migrated
        assert row["relation_type"] == "contribuye_a"
        # But context should be preserved
        assert row["context"] == "original context"


# ------------------------------------------------------------------ #
# 6. Backward compatibility tests
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    def test_create_relations_without_context(self, store):
        """create_relation works without context (backward compatible)."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        result = store.create_relation(e1, e2, "related_to")
        assert result is True
        # Verify in get_all_relations
        rels = store.get_all_relations()
        assert any(r["from"] == "A" and r["to"] == "B" for r in rels)

    def test_get_relations_legacy_format(self, store):
        """get_all_relations works with relations created before migration fields."""
        e1 = store.upsert_entity("A", "T")
        e2 = store.upsert_entity("B", "T")
        # Insert directly without context/active/ended_at (simulating pre-migration)
        store.db.execute(
            "INSERT INTO relations (from_entity, to_entity, relation_type) VALUES (?, ?, 'legacy_type')",
            (e1, e2),
        )
        store.db.commit()

        rels = store.get_all_relations()
        legacy = next(r for r in rels if r["relationType"] == "legacy_type")
        assert legacy["from"] == "A"
        assert legacy["to"] == "B"
        # New fields should be present with defaults
        assert legacy["context"] is None
        assert legacy["active"] == 1
        assert legacy["ended_at"] is None
