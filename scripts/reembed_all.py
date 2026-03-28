#!/usr/bin/env python3
"""Re-embed all entities with the current embedding model.

Usage::

    uv run python scripts/reembed_all.py

This is necessary after switching to a different embedding model
(e.g., paraphrase-multilingual → multilingual-e5-small) because
embeddings from different models are not comparable.

The script:
    1. Loads all entities with their observations and relations
    2. Prepares entity text using prepare_entity_text()
    3. Encodes with passage prefix (default for entities)
    4. Stores updated embeddings in sqlite-vec
    5. Reports statistics

NOTE: This script must be run AFTER applying the code changes to
embeddings.py (QUERY_PREFIX, PASSAGE_PREFIX, task parameter in encode()).
Otherwise entities will be embedded without the required "passage: " prefix.
"""

import sys
from pathlib import Path

# Ensure project source is importable
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mcp_memory.embeddings import EmbeddingEngine, serialize_f32
from mcp_memory.storage import MemoryStore

BATCH_SIZE = 8  # Encode entities in small batches


def main() -> None:
    print("=" * 60)
    print("  Re-embedding all entities")
    print("=" * 60)

    # --- Init ---
    engine = EmbeddingEngine()
    if not engine.available:
        print("\n  ERROR: Embedding model not available.")
        print("  Make sure model files exist in ~/.cache/mcp-memory-v2/models/")
        sys.exit(1)

    print(f"  Model: {engine.dimension}d")
    print(f"  Engine: loaded")

    store = MemoryStore()
    store.init_db()

    # --- Load all entities ---
    entities = store.get_all_entities()
    if not entities:
        print("\n  No entities found in database.")
        return

    print(f"  Found {len(entities)} entities to re-embed\n")

    # --- Process in batches ---
    success = 0
    failed = 0

    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]
        texts: list[str] = []
        entity_ids: list[int] = []

        for entity in batch:
            # Get relations for context
            relations = store.get_relations_for_entity(entity["id"])
            text = engine.prepare_entity_text(
                entity["name"],
                entity["entity_type"],
                entity["observations"],
                relations,
            )
            texts.append(text)
            entity_ids.append(entity["id"])

        try:
            # Encode batch (default task="passage" for entities)
            vectors = engine.encode(texts)

            for eid, vector in zip(entity_ids, vectors):
                store.store_embedding(eid, serialize_f32(vector))

            success += len(batch)
            print(
                f"  [{i + len(batch):>{len(str(len(entities)))}d}/{len(entities)}] "
                f"batch OK"
            )
        except Exception as e:
            failed += len(batch)
            print(
                f"  [{i + len(batch):>{len(str(len(entities)))}d}/{len(entities)}] "
                f"BATCH FAILED: {e}"
            )

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  Re-embedding complete:")
    print(f"    Success: {success}")
    print(f"    Failed:  {failed}")
    print(f"    Total:   {len(entities)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
