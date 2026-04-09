"""Tests for re-embedding entities — validates the DELETE+INSERT fix in store_embedding().

The original bug: store_embedding() used INSERT OR REPLACE against vec0 tables
(sqlite-vec), which is unsupported. The fix changed it to DELETE + INSERT.

These tests verify:
1. Re-embedding an entity replaces the old vector (not duplicates)
2. Storing the same embedding multiple times is idempotent
3. Reflection embeddings INSERT correctly
4. KNN search returns updated results after re-embedding
"""

import numpy as np
import pytest

from mcp_memory.embeddings import (
    DIMENSION,
    EmbeddingEngine,
    deserialize_f32,
    serialize_f32,
)
from mcp_memory.storage import MemoryStore


def test_store_embedding_overwrite(tmp_path):
    """Re-embedding an entity replaces the old vector in entity_embeddings."""
    store = MemoryStore(str(tmp_path / "test.db"))
    store.init_db()

    if not store._vec_loaded:
        store.close()
        pytest.skip("sqlite-vec not loaded")

    # Create entity
    eid = store.upsert_entity("TestEnt", "Testing")

    # Create two different embeddings
    vec1 = np.ones(DIMENSION, dtype=np.float32) * 0.5
    vec2 = np.ones(DIMENSION, dtype=np.float32) * 0.9

    # Store first embedding
    store.store_embedding(eid, serialize_f32(vec1))

    # Verify first embedding stored
    row = store.db.execute(
        "SELECT embedding FROM entity_embeddings WHERE rowid = ?", (eid,)
    ).fetchone()
    assert row is not None
    stored1 = deserialize_f32(row["embedding"])
    np.testing.assert_array_almost_equal(stored1, vec1)

    # Re-store with different embedding (THIS is what the bug prevented)
    store.store_embedding(eid, serialize_f32(vec2))

    # Verify second embedding replaced the first
    row2 = store.db.execute(
        "SELECT embedding FROM entity_embeddings WHERE rowid = ?", (eid,)
    ).fetchone()
    assert row2 is not None
    stored2 = deserialize_f32(row2["embedding"])
    np.testing.assert_array_almost_equal(stored2, vec2)

    # Verify it's a replacement, not a duplicate
    count = store.db.execute(
        "SELECT COUNT(*) FROM entity_embeddings WHERE rowid = ?", (eid,)
    ).fetchone()[0]
    assert count == 1  # Only one row, not two

    store.close()


def test_store_embedding_idempotent(tmp_path):
    """Storing the same embedding twice doesn't duplicate rows."""
    store = MemoryStore(str(tmp_path / "test.db"))
    store.init_db()

    if not store._vec_loaded:
        store.close()
        pytest.skip("sqlite-vec not loaded")

    eid = store.upsert_entity("IdempotentEnt", "Testing")
    vec = np.random.randn(DIMENSION).astype(np.float32)
    emb = serialize_f32(vec)

    store.store_embedding(eid, emb)
    store.store_embedding(eid, emb)
    store.store_embedding(eid, emb)

    count = store.db.execute(
        "SELECT COUNT(*) FROM entity_embeddings WHERE rowid = ?", (eid,)
    ).fetchone()[0]
    assert count == 1

    store.close()


def test_reflection_embedding_insert_works(tmp_path):
    """Creating a reflection stores its embedding in reflection_embeddings."""
    store = MemoryStore(str(tmp_path / "test.db"))
    store.init_db()

    if not store._vec_loaded:
        store.close()
        pytest.skip("sqlite-vec not loaded")

    engine = EmbeddingEngine.get_instance()
    if not engine or not engine.available:
        store.close()
        pytest.skip("Embedding model not available")

    eid = store.upsert_entity("RefTest", "Testing")
    result = store.add_reflection("entity", eid, "nolan", "Contenido de prueba")
    assert result is not None

    # Verify embedding exists
    row = store.db.execute(
        "SELECT embedding FROM reflection_embeddings WHERE rowid = ?",
        (result["id"],),
    ).fetchone()
    assert row is not None
    stored = deserialize_f32(row["embedding"])
    assert len(stored) == DIMENSION

    store.close()


def test_search_after_re_embedding(tmp_path):
    """After re-embedding, KNN search returns the updated vector."""
    store = MemoryStore(str(tmp_path / "test.db"))
    store.init_db()

    if not store._vec_loaded:
        store.close()
        pytest.skip("sqlite-vec not loaded")

    e1 = store.upsert_entity("Entity1", "Testing")
    e2 = store.upsert_entity("Entity2", "Testing")

    # Store initial embeddings
    vec1 = np.zeros(DIMENSION, dtype=np.float32)
    vec1[0] = 1.0  # pointing in dimension 0
    vec2 = np.zeros(DIMENSION, dtype=np.float32)
    vec2[1] = 1.0  # pointing in dimension 1

    store.store_embedding(e1, serialize_f32(vec1))
    store.store_embedding(e2, serialize_f32(vec2))

    # Query near e1 — should return e1 first
    query = np.zeros(DIMENSION, dtype=np.float32)
    query[0] = 1.0
    results = store.search_embeddings(serialize_f32(query), limit=2)
    assert len(results) >= 2
    assert results[0]["entity_id"] == e1

    # Now re-embed e1 to point near e2
    new_vec1 = np.zeros(DIMENSION, dtype=np.float32)
    new_vec1[1] = 1.0  # now pointing in dimension 1, same as e2
    store.store_embedding(e1, serialize_f32(new_vec1))

    # Query again — e1 should now be close to e2
    results2 = store.search_embeddings(serialize_f32(query), limit=2)
    # Both should appear in results
    ids = [r["entity_id"] for r in results2]
    assert e1 in ids
    assert e2 in ids

    store.close()
