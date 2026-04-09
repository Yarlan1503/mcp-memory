#!/usr/bin/env python3
"""Re-embed migrated entities after entity_type rename.

Normalizes tildes (Decisión → Decision), re-embeds entities of types
Proyecto, Sistema, Decision, Recurso, and verifies final state.

Usage::

    .venv/bin/python scripts/reembed_migrated.py
"""

import logging
import sys

from mcp_memory.embeddings import EmbeddingEngine, serialize_f32
from mcp_memory.storage import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Types that were migrated and need re-embedding
MIGRATED_TYPES = ("Proyecto", "Sistema", "Decision", "Recurso")


def main() -> None:
    print("=" * 60)
    print("  Re-embed migrated entities")
    print("=" * 60)

    # --- Init store ---
    store = MemoryStore()
    store.init_db()
    # Override busy_timeout for concurrent MCP server access
    store.db.execute("PRAGMA busy_timeout=30000;")

    # --- Init embedding engine ---
    engine = EmbeddingEngine.get_instance()
    if not engine.available:
        print("\n  ERROR: Embedding model not available.")
        print("  Run: .venv/bin/python scripts/download_model.py")
        sys.exit(1)

    print(f"  Engine: loaded ({engine.dimension}d)")

    # ================================================================
    # STEP 1: Normalize tilde — Decisión → Decision
    # ================================================================
    print("\n--- Step 1: Normalize tilde ---")
    cursor = store.db.execute(
        "UPDATE entities SET entity_type = 'Decision' WHERE entity_type = 'Decisión'"
    )
    tilde_count = cursor.rowcount
    store.db.commit()
    print(f"  Normalized {tilde_count} entities: Decisión → Decision")

    # ================================================================
    # STEP 2: Re-embed migrated entities
    # ================================================================
    print("\n--- Step 2: Re-embed migrated entities ---")

    # Fetch entities of migrated types
    placeholders = ",".join("?" for _ in MIGRATED_TYPES)
    rows = store.db.execute(
        f"SELECT id, name, entity_type, status FROM entities "
        f"WHERE entity_type IN ({placeholders})",
        list(MIGRATED_TYPES),
    ).fetchall()
    entities = [dict(r) for r in rows]

    total = len(entities)
    print(f"  Found {total} entities to re-embed\n")

    success = 0
    failed = 0

    for idx, entity in enumerate(entities, start=1):
        eid = entity["id"]
        name = entity["name"]
        etype = entity["entity_type"]
        status = entity.get("status", "activo")

        try:
            # Get observations (as dicts to preserve kind info)
            observations = store.get_observations_with_ids(eid, exclude_superseded=True)

            # Get relations
            relations = store.get_relations_for_entity(eid)

            # Prepare embedding text
            text = EmbeddingEngine.prepare_entity_text(
                name, etype, observations, relations, status
            )

            # Encode
            vectors = engine.encode([text])
            embedding_bytes = serialize_f32(vectors[0])

            # vec0 doesn't support INSERT OR REPLACE — must DELETE first
            try:
                store.db.execute(
                    "DELETE FROM entity_embeddings WHERE rowid = ?", (eid,)
                )
            except Exception:
                pass  # May not exist yet

            store.store_embedding(eid, embedding_bytes)
            store._sync_fts(eid)

            success += 1
            print(f"  Re-embedding {idx}/{total}: {name} ({etype})")

        except Exception as exc:
            failed += 1
            logger.error(
                "  FAILED %s/%s: %s (%s) — %s",
                idx,
                total,
                name,
                etype,
                exc,
            )

    # ================================================================
    # STEP 3: Verify final state
    # ================================================================
    print("\n--- Step 3: Verify final state ---")

    # Entity type distribution
    print("\n  Entity type distribution:")
    type_rows = store.db.execute(
        "SELECT entity_type, COUNT(*) as count "
        "FROM entities GROUP BY entity_type ORDER BY count DESC"
    ).fetchall()
    for r in type_rows:
        print(f"    {r['entity_type']:20s} {r['count']:>4d}")

    # Embedding coverage
    emb_count = store.db.execute("SELECT COUNT(*) FROM entity_embeddings").fetchone()[0]
    ent_count = store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"\n  Embeddings: {emb_count} / Entities: {ent_count}")
    if emb_count < ent_count:
        print(f"  ⚠  {ent_count - emb_count} entities without embeddings")

    # Check no Decisión remains
    tilde_left = store.db.execute(
        "SELECT COUNT(*) FROM entities WHERE entity_type = 'Decisión'"
    ).fetchone()[0]
    if tilde_left > 0:
        print(f"  ⚠  {tilde_left} entities still have 'Decisión' type!")
    else:
        print("  ✓ No 'Decisión' entities remaining — tilde normalized")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"    Tilde normalized:    {tilde_count} entities")
    print(f"    Re-embedded:         {success}/{total}")
    print(f"    Failed:              {failed}")
    print(f"    Embedding coverage:  {emb_count}/{ent_count}")
    print("=" * 60)

    store.close()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
