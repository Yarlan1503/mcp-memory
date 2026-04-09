#!/usr/bin/env python3
"""Migrate obsolete entity types in the knowledge graph.

Standalone script — run manually when Nolan/sofia decide to migrate.
Not called automatically by the system.

Usage:
    python scripts/migrate_entity_types.py [--dry-run] [--db-path PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project source is importable
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mcp_memory.storage import MemoryStore

logger = logging.getLogger(__name__)

ENTITY_TYPE_MIGRATIONS = {
    "Subproyecto": "Proyecto",
    "Mejora": "Proyecto",
    "Investigacion": "Proyecto",
    "Articulo": "Proyecto",
    "Tarea": "Proyecto",
    "Componente": "Sistema",
    "Diseno": "Decision",
    "ConocimientoTecnico": "Recurso",
}


def migrate_entity_types(store: MemoryStore, dry_run: bool = False) -> dict:
    """Migrate entity types and return summary.

    Returns a dict with:
        - type_counts: {old_type: count_migrated}
        - names_renamed: int
        - embeddings_ok: int
        - embeddings_total: int
        - fts_synced: int
        - affected_ids: list[int]
    """
    summary: dict = {
        "type_counts": {},
        "names_renamed": 0,
        "embeddings_ok": 0,
        "embeddings_total": 0,
        "fts_synced": 0,
        "affected_ids": [],
    }

    # Attempt to load the embedding engine (optional)
    engine = None
    try:
        from mcp_memory.embeddings import EmbeddingEngine, serialize_f32

        engine = EmbeddingEngine()
        if not engine.available:
            logger.warning(
                "Embedding model not available — skipping embedding recomputation"
            )
            engine = None
    except Exception as exc:
        logger.warning(
            "Could not load EmbeddingEngine: %s — skipping embedding recomputation", exc
        )

    for old_type, new_type in ENTITY_TYPE_MIGRATIONS.items():
        # Count entities to migrate
        row = store.db.execute(
            "SELECT COUNT(*) as cnt FROM entities WHERE entity_type = ?",
            (old_type,),
        ).fetchone()
        count = row["cnt"]

        if count == 0:
            if not dry_run:
                print(f"  {old_type} → {new_type}: 0 entities (skipped)")
            summary["type_counts"][old_type] = 0
            continue

        if dry_run:
            print(
                f"  [DRY RUN] {old_type} → {new_type}: {count} entities would be migrated"
            )
            summary["type_counts"][old_type] = count
            # Still collect affected IDs for reporting
            rows = store.db.execute(
                "SELECT id, name FROM entities WHERE entity_type = ?",
                (old_type,),
            ).fetchall()
            summary["affected_ids"].extend([r["id"] for r in rows])
            continue

        # Collect entity IDs and names before updating
        rows = store.db.execute(
            "SELECT id, name FROM entities WHERE entity_type = ?",
            (old_type,),
        ).fetchall()
        affected_ids = [r["id"] for r in rows]
        summary["affected_ids"].extend(affected_ids)

        # 1. Update entity_type
        store.db.execute(
            "UPDATE entities SET entity_type = ? WHERE entity_type = ?",
            (new_type, old_type),
        )
        store.db.commit()
        summary["type_counts"][old_type] = count

        # 2. Rename names with old type prefix
        prefix = f"{old_type}: "
        for r in rows:
            name = r["name"]
            if name.startswith(prefix):
                new_name = (
                    new_type + name[len(old_type) :]
                )  # Replace old_type prefix with new_type
                store.db.execute(
                    "UPDATE entities SET name = ? WHERE id = ?",
                    (new_name, r["id"]),
                )
                summary["names_renamed"] += 1
        store.db.commit()

        # 3. Re-compute embeddings & re-sync FTS for affected entities
        for eid in affected_ids:
            entity = store.get_entity_by_id(eid)
            if not entity:
                continue

            # Embedding recomputation
            if engine is not None:
                summary["embeddings_total"] += 1
                try:
                    observations = store.get_observations(eid, exclude_superseded=True)
                    relations = store.get_relations_for_entity(eid)
                    text = engine.prepare_entity_text(
                        entity["name"],
                        entity["entity_type"],
                        observations,
                        relations,
                    )
                    vector = engine.encode([text])
                    store.store_embedding(eid, serialize_f32(vector[0]))
                    summary["embeddings_ok"] += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to recompute embedding for entity %s (%s): %s",
                        eid,
                        entity["name"],
                        exc,
                    )

            # FTS re-sync
            store._sync_fts(eid)
            summary["fts_synced"] += 1

        print(f"  {old_type} → {new_type}: {count} entities migrated")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate obsolete entity types in the knowledge graph.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes.",
    )
    parser.add_argument(
        "--db-path",
        default="~/.config/opencode/mcp-memory/memory.db",
        help="Path to the SQLite database (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Entity type migration")
    if args.dry_run:
        print("  [DRY RUN — no changes will be made]")
    print("=" * 60)
    print()

    store = MemoryStore(db_path=args.db_path)
    store.init_db()

    summary = migrate_entity_types(store, dry_run=args.dry_run)

    # Print summary
    print()
    print("=" * 60)
    print("  Entity type migration summary:")
    for old_type, count in summary["type_counts"].items():
        new_type = ENTITY_TYPE_MIGRATIONS[old_type]
        print(f"    {old_type} → {new_type}: {count} entities migrated")
    print(f"  Names renamed: {summary['names_renamed']}")
    if summary["embeddings_total"] > 0:
        print(
            f"  Embeddings recomputed: {summary['embeddings_ok']}/{summary['embeddings_total']}"
        )
    else:
        total_affected = len(summary["affected_ids"])
        if total_affected > 0 and not args.dry_run:
            print("  Embeddings recomputed: model not available, skipped")
    print(f"  FTS re-synced: {summary['fts_synced']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
