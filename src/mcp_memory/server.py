import logging
import sys
from typing import Any

from fastmcp import FastMCP

from mcp_memory.models import EntityInput, EntityOutput, RelationInput, RelationOutput
from mcp_memory.storage import MemoryStore

logger = logging.getLogger(__name__)

mcp = FastMCP("memory")

# Instantiate store — default path
store = MemoryStore()
store.init_db()


def _get_engine() -> Any:
    """Get embedding engine instance (may be unavailable).

    Lazy import to allow the server to start even when
    ``mcp_memory.embeddings`` hasn't been created yet (T6).
    """
    try:
        from mcp_memory.embeddings import EmbeddingEngine

        return EmbeddingEngine.get_instance()
    except Exception:
        return None


def _entity_to_output(row: dict, observations: list[str]) -> dict:
    """Convert DB row + observations to EntityOutput dict."""
    return {
        "name": row["name"],
        "entityType": row["entity_type"],
        "observations": observations,
    }


def _recompute_embedding(
    entity_id: int, name: str, entity_type: str, observations: list[str]
) -> None:
    """Recompute and store embedding for an entity (if engine available)."""
    engine = _get_engine()
    if not engine or not getattr(engine, "available", False):
        return
    try:
        from mcp_memory.embeddings import serialize_f32

        text = engine.prepare_entity_text(name, entity_type, observations)
        vector = engine.encode([text])
        store.store_embedding(entity_id, serialize_f32(vector[0]))
    except Exception as e:
        logger.warning("Failed to recompute embedding for %s: %s", name, e)


# ============================================================
# TOOL 1: create_entities
# ============================================================
@mcp.tool
def create_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Create or update entities in the knowledge graph.
    If an entity already exists, merge observations (don't overwrite).
    Returns the created/updated entities."""
    try:
        results = []
        for entity_dict in entities:
            parsed = EntityInput.model_validate(entity_dict)

            # Upsert entity
            entity_id = store.upsert_entity(parsed.name, parsed.entityType)

            # Get existing observations
            existing_obs = store.get_observations(entity_id)
            existing_set = set(existing_obs)

            # Add new observations (skip duplicates)
            new_obs = [o for o in parsed.observations if o not in existing_set]
            if new_obs:
                store.add_observations(entity_id, new_obs)

            # Get final observations (existing + new)
            all_obs = store.get_observations(entity_id)

            # Recompute embedding
            _recompute_embedding(entity_id, parsed.name, parsed.entityType, all_obs)

            # Limbic: initialize access tracking for new/updated entity
            store.init_access(entity_id)

            results.append(
                {
                    "name": parsed.name,
                    "entityType": parsed.entityType,
                    "observations": all_obs,
                }
            )

        return {"entities": results}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 2: create_relations
# ============================================================
@mcp.tool
def create_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """Create relations between entities. Both entities must exist.
    Returns created relations or errors for missing entities."""
    try:
        results = []
        errors = []

        for rel_dict in relations:
            parsed = RelationInput.model_validate(rel_dict)

            from_ent = store.get_entity_by_name(parsed.from_entity)
            to_ent = store.get_entity_by_name(parsed.to_entity)

            if not from_ent:
                errors.append(f"Entity not found: {parsed.from_entity}")
                results.append({"error": f"Entity not found: {parsed.from_entity}"})
                continue
            if not to_ent:
                errors.append(f"Entity not found: {parsed.to_entity}")
                results.append({"error": f"Entity not found: {parsed.to_entity}"})
                continue

            created = store.create_relation(
                from_ent["id"], to_ent["id"], parsed.relationType
            )
            if created:
                results.append(
                    {
                        "from": parsed.from_entity,
                        "to": parsed.to_entity,
                        "relationType": parsed.relationType,
                    }
                )
            else:
                results.append(
                    {
                        "from": parsed.from_entity,
                        "to": parsed.to_entity,
                        "relationType": parsed.relationType,
                        "error": "Relation already exists",
                    }
                )

        if errors:
            return {"relations": results, "errors": errors}
        return {"relations": results}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 3: add_observations
# ============================================================
@mcp.tool
def add_observations(name: str, observations: list[str]) -> dict[str, Any]:
    """Add observations to an existing entity."""
    try:
        entity = store.get_entity_by_name(name)
        if not entity:
            return {"error": f"Entity not found: {name}"}

        store.add_observations(entity["id"], observations)
        all_obs = store.get_observations(entity["id"])

        # Recompute embedding
        _recompute_embedding(
            entity["id"], entity["name"], entity["entity_type"], all_obs
        )

        # Limbic: record access on observation add
        store.record_access(entity["id"])

        return {
            "entity": {
                "name": entity["name"],
                "entityType": entity["entity_type"],
                "observations": all_obs,
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 4: delete_entities
# ============================================================
@mcp.tool
def delete_entities(entityNames: list[str]) -> dict[str, Any]:
    """Delete entities and all their relations/observations."""
    try:
        deleted = []
        errors = []

        for name in entityNames:
            entity = store.get_entity_by_name(name)
            if not entity:
                errors.append(f"Entity not found: {name}")
                continue
            count = store.delete_entities_by_names([name])
            if count > 0:
                deleted.append(name)

        result: dict[str, Any] = {"deleted": deleted}
        if errors:
            result["errors"] = errors
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 5: delete_observations
# ============================================================
@mcp.tool
def delete_observations(name: str, observations: list[str]) -> dict[str, Any]:
    """Delete specific observations from an entity."""
    try:
        entity = store.get_entity_by_name(name)
        if not entity:
            return {"error": f"Entity not found: {name}"}

        store.delete_observations(entity["id"], observations)
        all_obs = store.get_observations(entity["id"])

        # Recompute embedding
        _recompute_embedding(
            entity["id"], entity["name"], entity["entity_type"], all_obs
        )

        return {
            "entity": {
                "name": entity["name"],
                "entityType": entity["entity_type"],
                "observations": all_obs,
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 6: delete_relations
# ============================================================
@mcp.tool
def delete_relations(relations: list[dict[str, Any]]) -> dict[str, Any]:
    """Delete relations between entities."""
    try:
        deleted = []
        errors = []

        for rel_dict in relations:
            parsed = RelationInput.model_validate(rel_dict)

            from_ent = store.get_entity_by_name(parsed.from_entity)
            to_ent = store.get_entity_by_name(parsed.to_entity)

            if not from_ent or not to_ent:
                errors.append(
                    f"Entity not found: {parsed.from_entity} or {parsed.to_entity}"
                )
                continue

            was_deleted = store.delete_relation(
                from_ent["id"], to_ent["id"], parsed.relationType
            )
            if was_deleted:
                deleted.append(
                    {
                        "from": parsed.from_entity,
                        "to": parsed.to_entity,
                        "relationType": parsed.relationType,
                    }
                )
            else:
                errors.append(
                    f"Relation not found: {parsed.from_entity} -> "
                    f"{parsed.to_entity} ({parsed.relationType})"
                )

        result: dict[str, Any] = {"deleted": deleted}
        if errors:
            result["errors"] = errors
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 7: search_nodes
# ============================================================
@mcp.tool
def search_nodes(query: str) -> dict[str, Any]:
    """Search for nodes in the knowledge graph by name, type, or observation content."""
    try:
        entities = store.search_entities(query)
        results = []
        for entity in entities:
            obs = store.get_observations(entity["id"])
            results.append(_entity_to_output(entity, obs))
        return {"entities": results}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 8: open_nodes
# ============================================================
@mcp.tool
def open_nodes(names: list[str]) -> dict[str, Any]:
    """Open specific nodes by name. Returns full entity data with observations."""
    try:
        results = []
        for name in names:
            entity = store.get_entity_by_name(name)
            if entity:
                obs = store.get_observations(entity["id"])
                # Limbic: record access on node open
                store.record_access(entity["id"])
                results.append(_entity_to_output(entity, obs))
        return {"entities": results}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 9: read_graph
# ============================================================
@mcp.tool
def read_graph() -> dict[str, Any]:
    """Read the entire knowledge graph. Returns all entities with observations and all relations."""
    try:
        return store.read_graph()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 10: search_semantic
# ============================================================
@mcp.tool
def search_semantic(query: str, limit: int = 10) -> dict[str, Any]:
    """Semantic search using vector embeddings. Finds entities most similar to the query.
    Requires the embedding model to be downloaded (run download_model.py first)."""
    try:
        engine = _get_engine()
        if not engine or not engine.available:
            return {
                "error": "Embedding model not available. Run 'python scripts/download_model.py' to download the model first.",
            }

        from mcp_memory.embeddings import serialize_f32
        from mcp_memory.scoring import EXPANSION_FACTOR, rank_candidates

        # Encode query
        query_vector = engine.encode([query])
        query_bytes = serialize_f32(query_vector[0])

        # KNN search with expansion factor (over-retrieve candidates)
        expanded_limit = limit * EXPANSION_FACTOR
        knn_results = store.search_embeddings(query_bytes, limit=expanded_limit)

        if not knn_results:
            return {"results": []}

        # Collect candidate IDs
        candidate_ids = [r["entity_id"] for r in knn_results]

        # Fetch limbic signals for all candidates
        access_data = store.get_access_data(candidate_ids)
        degree_data = store.get_entity_degrees(candidate_ids)
        cooc_data = store.get_co_occurrences(candidate_ids)

        # Fetch created_at for temporal factor
        entity_created: dict[int, str] = {}
        for eid in candidate_ids:
            row = store.get_entity_by_id(eid)
            if row:
                entity_created[eid] = row["created_at"]

        # Re-rank with limbic scoring (Opción B)
        ranked = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=limit,
        )

        # Build output (same format as before)
        output = []
        top_k_ids = []
        for item in ranked:
            eid = item["entity_id"]
            row = store.get_entity_by_id(eid)
            if not row:
                continue

            obs = store.get_observations(eid)
            output.append(
                {
                    "name": row["name"],
                    "entityType": row["entity_type"],
                    "observations": obs,
                    "distance": round(item["distance"], 4),
                }
            )
            top_k_ids.append(eid)

        # Limbic: record access + co-occurrences for top-K (post-response, best-effort)
        try:
            for eid in top_k_ids:
                store.record_access(eid)
            if len(top_k_ids) >= 2:
                store.record_co_occurrences(top_k_ids)
        except Exception as e:
            logger.warning("Limbic tracking failed: %s", e)

        return {"results": output}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TOOL 11: migrate
# ============================================================
@mcp.tool
def migrate(
    source_path: str = "",
) -> dict[str, Any]:
    """Migrate data from Anthropic MCP Memory JSONL format to SQLite.
    This is idempotent — running it multiple times won't duplicate data.
    Requires a valid source_path to an existing JSONL file."""
    if not source_path:
        return {
            "error": "source_path is required — provide the path to an Anthropic MCP Memory JSONL file"
        }
    try:
        from mcp_memory.migrate import migrate_jsonl

        engine = _get_engine()
        result = migrate_jsonl(store, source_path, engine)
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Main entry point
# ============================================================
def main() -> None:
    """Start the MCP server (stdio transport)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Log to stderr, not stdout (MCP uses stdout)
    )
    logger.info("Starting MCP Memory v2 server...")
    mcp.run()
