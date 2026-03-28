import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_memory.embeddings import EmbeddingEngine
    from mcp_memory.storage import MemoryStore

logger = logging.getLogger(__name__)


def migrate_jsonl(
    store: "MemoryStore",
    source_path: str,
    engine: "EmbeddingEngine | None" = None,
) -> dict[str, int]:
    """
    Migrate JSONL Anthropic format to SQLite.

    - Parse line by line (json.loads)
    - Ignore corrupt lines (try/except, log warning)
    - Entities: upsert_entity + merge observations
    - Relations: create if both entities exist (skip if either missing)
    - Batch embeddings at the end (if engine available)

    Returns: {"entities_imported": N, "relations_imported": N, "errors": N, "skipped": N}
    """
    source_path = str(Path(source_path).expanduser())
    entities_imported = 0
    relations_imported = 0
    errors = 0
    skipped = 0

    # Collect all entity names for later embedding generation
    entity_names_for_embedding: list[str] = []

    with open(source_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: corrupt JSON — %s", line_num, e)
                errors += 1
                continue

            record_type = record.get("type")

            if record_type == "entity":
                try:
                    name = record.get("name", "")
                    entity_type = record.get("entityType", "Generic")
                    observations = record.get("observations", [])

                    if not name:
                        logger.warning(
                            "Line %d: entity without name, skipping", line_num
                        )
                        skipped += 1
                        continue

                    # Upsert entity
                    entity_id = store.upsert_entity(name, entity_type)

                    # Merge observations (skip duplicates)
                    if observations:
                        existing = store.get_observations(entity_id)
                        existing_set = set(existing)
                        new_obs = [o for o in observations if o not in existing_set]
                        if new_obs:
                            store.add_observations(entity_id, new_obs)

                    entities_imported += 1
                    entity_names_for_embedding.append(name)

                except Exception as e:
                    logger.error("Line %d: error importing entity — %s", line_num, e)
                    errors += 1

            elif record_type == "relation":
                try:
                    from_name = record.get("from", "")
                    to_name = record.get("to", "")
                    relation_type = record.get("relationType", "")

                    if not from_name or not to_name or not relation_type:
                        logger.warning(
                            "Line %d: relation missing fields, skipping", line_num
                        )
                        skipped += 1
                        continue

                    # Check both entities exist
                    from_ent = store.get_entity_by_name(from_name)
                    to_ent = store.get_entity_by_name(to_name)

                    if not from_ent or not to_ent:
                        logger.debug(
                            "Line %d: skipping relation %s → %s (entity not found)",
                            line_num,
                            from_name,
                            to_name,
                        )
                        skipped += 1
                        continue

                    # Create relation (idempotent — skip if exists)
                    created = store.create_relation(
                        from_ent["id"], to_ent["id"], relation_type
                    )
                    if created:
                        relations_imported += 1

                except Exception as e:
                    logger.error("Line %d: error importing relation — %s", line_num, e)
                    errors += 1

            else:
                logger.warning(
                    "Line %d: unknown record type '%s', skipping", line_num, record_type
                )
                skipped += 1

    # Batch embeddings at the end
    if engine and engine.available and entity_names_for_embedding:
        logger.info(
            "Generating embeddings for %d entities...", len(entity_names_for_embedding)
        )
        embed_count = 0
        for name in entity_names_for_embedding:
            try:
                entity = store.get_entity_by_name(name)
                if not entity:
                    continue

                observations = store.get_observations(entity["id"])
                relations = store.get_relations_for_entity(entity["id"])
                text = engine.prepare_entity_text(
                    entity["name"], entity["entity_type"], observations, relations
                )
                vector = engine.encode([text])

                from mcp_memory.embeddings import serialize_f32

                store.store_embedding(entity["id"], serialize_f32(vector[0]))
                embed_count += 1

            except Exception as e:
                logger.warning("Failed to generate embedding for '%s': %s", name, e)

        logger.info(
            "Generated %d/%d embeddings", embed_count, len(entity_names_for_embedding)
        )

    logger.info(
        "Migration complete: %d entities, %d relations, %d errors, %d skipped",
        entities_imported,
        relations_imported,
        errors,
        skipped,
    )

    return {
        "entities_imported": entities_imported,
        "relations_imported": relations_imported,
        "errors": errors,
        "skipped": skipped,
    }
