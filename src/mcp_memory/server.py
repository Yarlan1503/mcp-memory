import hashlib
import logging
import random
import sys
import time
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

        relations = store.get_relations_for_entity(entity_id)
        text = engine.prepare_entity_text(name, entity_type, observations, relations)
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
        logger.error("Error in create_entities: %s", e)
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
        logger.error("Error in create_relations: %s", e)
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
        logger.error("Error in add_observations: %s", e)
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
        logger.error("Error in delete_entities: %s", e)
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
        logger.error("Error in delete_observations: %s", e)
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
        logger.error("Error in delete_relations: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 7: search_nodes
# ============================================================
@mcp.tool
def search_nodes(query: str) -> dict[str, Any]:
    """Search for nodes in the knowledge graph by name, type, or observation content."""
    try:
        if not query.strip():
            return {"entities": []}
        entities = store.search_entities(query)
        results = []
        for entity in entities:
            obs = store.get_observations(entity["id"])
            results.append(_entity_to_output(entity, obs))
        return {"entities": results}
    except Exception as e:
        logger.error("Error in search_nodes: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 8: open_nodes
# ============================================================
@mcp.tool
def open_nodes(names: list[str]) -> dict[str, Any]:
    """Open specific nodes by name. Returns full entity data with observations."""
    try:
        results = []
        opened_ids = []
        for name in names:
            entity = store.get_entity_by_name(name)
            if entity:
                obs = store.get_observations(entity["id"])
                # Limbic: record access on node open
                store.record_access(entity["id"])
                opened_ids.append(entity["id"])
                results.append(_entity_to_output(entity, obs))

        # Limbic: record co-occurrences for entities opened together (best-effort)
        try:
            if len(opened_ids) >= 2:
                store.record_co_occurrences(opened_ids)
        except Exception as e:
            logger.warning("Limbic co-occurrence tracking failed: %s", e)

        return {"entities": results}
    except Exception as e:
        logger.error("Error in open_nodes: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 9: search_semantic
# ============================================================

# --- A/B Testing Configuration ---
USE_AB_TESTING = True
BASELINE_PROBABILITY = 0.1  # 10% of queries are baseline (treatment=0)


def _compute_baseline_ranking(
    knn_results: list[dict],
    candidate_ids: list[int],
) -> list[dict]:
    """Compute baseline ranking (cosine-only) for A/B comparison.

    Returns list sorted by cosine_sim descending with baseline_rank assigned.
    Only includes entities present in knn_results (has distance).
    """
    baseline = []
    for item in knn_results:
        eid = item["entity_id"]
        distance = item.get("distance", 1.0)
        cosine_sim = max(0.0, 1.0 - distance)
        baseline.append(
            {
                "entity_id": eid,
                "cosine_sim": cosine_sim,
            }
        )

    # Sort by cosine_sim descending
    baseline.sort(key=lambda x: x["cosine_sim"], reverse=True)

    # Assign baseline_rank (1-based)
    for rank, item in enumerate(baseline, start=1):
        item["baseline_rank"] = rank

    return baseline


def _get_treatment(query: str) -> int:
    """Determine A/B treatment for a query.

    Uses hash-based determinism if random is unavailable.
    Returns 0 for baseline (cosine-only), 1 for limbic scoring.
    """
    if not USE_AB_TESTING:
        return 1  # Default to limbic

    # Try random first
    try:
        return 0 if random.random() < BASELINE_PROBABILITY else 1
    except Exception:
        # Fallback: hash-based determinism for reproducibility
        query_hash = hashlib.md5(query.encode()).hexdigest()
        hash_value = int(query_hash, 16) % 100
        return 0 if hash_value < (BASELINE_PROBABILITY * 100) else 1


@mcp.tool
def search_semantic(query: str, limit: int = 10) -> dict[str, Any]:
    """Semantic search using vector embeddings with optional full-text hybrid search.
    Combines semantic (KNN) and text (FTS5) results via Reciprocal Rank Fusion,
    then applies limbic re-ranking based on access patterns and co-occurrence.
    Requires the embedding model to be downloaded (run download_model.py first)."""
    try:
        engine = _get_engine()
        if not engine or not engine.available:
            return {
                "error": "Embedding model not available. Run 'python scripts/download_model.py' to download the model first.",
            }

        # --- A/B Testing: Determine treatment ---
        treatment = _get_treatment(query)

        from mcp_memory.embeddings import serialize_f32
        from mcp_memory.scoring import (
            EXPANSION_FACTOR,
            rank_candidates,
            rank_hybrid_candidates,
            reciprocal_rank_fusion,
        )

        # --- Step 1: Semantic KNN search ---
        query_vector = engine.encode([query], task="query")
        query_bytes = serialize_f32(query_vector[0])
        expanded_limit = limit * EXPANSION_FACTOR
        knn_results = store.search_embeddings(query_bytes, limit=expanded_limit)

        if not knn_results:
            return {"results": []}

        # --- Step 2: FTS5 full-text search (best-effort) ---
        fts_results = store.search_fts(query, limit=expanded_limit)

        # --- Step 3: Merge via RRF (hybrid) or pure semantic (fallback) ---
        use_hybrid = len(fts_results) > 0

        if use_hybrid:
            # RRF merge of semantic + FTS5 rankings
            merged = reciprocal_rank_fusion(knn_results, fts_results)
            # Collect all candidate IDs from merged results
            candidate_ids = [r["entity_id"] for r in merged]
        else:
            # Pure semantic — use original KNN results
            merged = None
            candidate_ids = [r["entity_id"] for r in knn_results]

        # --- SHADOW MODE: Compute baseline ranking for A/B comparison ---
        baseline_ranked = _compute_baseline_ranking(knn_results, candidate_ids)

        # Prepare limbic data for logging (needed for both treatments)
        access_data = store.get_access_data(candidate_ids)
        degree_data = store.get_entity_degrees(candidate_ids)
        cooc_data = store.get_co_occurrences(candidate_ids)

        entity_created: dict[int, str] = {}
        for eid in candidate_ids:
            row = store.get_entity_by_id(eid)
            if row:
                entity_created[eid] = row["created_at"]

        # --- SHADOW MODE: Log to database (best-effort, non-blocking) ---
        shadow_start_time = time.perf_counter()

        try:
            # Log event
            event_id = store.log_search_event(
                query_text=query,
                treatment=treatment,
                k_limit=limit,
                num_results=None,  # Will update after ranking
                duration_ms=None,
                engine_used="limbic" if treatment == 1 else "baseline",
            )
        except Exception as e:
            logger.warning("Shadow mode event logging failed: %s", e)
            event_id = None

        # --- Step 4: Re-rank based on treatment ---
        if treatment == 0:
            # BASELINE: Use cosine-only ranking (no limbic scoring)
            ranked = baseline_ranked[:limit]
        else:
            # LIMBIC: Apply limbic re-ranking
            if use_hybrid:
                ranked = rank_hybrid_candidates(
                    merged_results=merged,
                    access_data=access_data,
                    degree_data=degree_data,
                    cooc_data=cooc_data,
                    entity_created=entity_created,
                    limit=limit,
                )
            else:
                ranked = rank_candidates(
                    knn_results=knn_results,
                    access_data=access_data,
                    degree_data=degree_data,
                    cooc_data=cooc_data,
                    entity_created=entity_created,
                    limit=limit,
                )

        # --- Build output based on treatment ---
        output = []
        top_k_ids = []

        if treatment == 0:
            # BASELINE output: cosine-only ranking, no limbic scores
            for rank_pos, item in enumerate(ranked):
                eid = item["entity_id"]
                row = store.get_entity_by_id(eid)
                if not row:
                    continue

                obs = store.get_observations(eid)
                result_item = {
                    "name": row["name"],
                    "entityType": row["entity_type"],
                    "observations": obs,
                    "distance": round(1.0 - item.get("cosine_sim", 0.0), 4),
                    "treatment": "baseline",
                }

                # Add cosine_sim for baseline
                result_item["cosine_sim"] = round(item.get("cosine_sim", 0.0), 4)

                output.append(result_item)
                top_k_ids.append(eid)
        else:
            # LIMBIC output: full scoring details
            for item in ranked:
                eid = item["entity_id"]
                row = store.get_entity_by_id(eid)
                if not row:
                    continue

                obs = store.get_observations(eid)
                result_item = {
                    "name": row["name"],
                    "entityType": row["entity_type"],
                    "observations": obs,
                    "limbic_score": round(item.get("limbic_score", 0.0), 4),
                    "scoring": {
                        "importance": round(item.get("importance", 0.0), 4),
                        "temporal_factor": round(item.get("temporal_factor", 1.0), 4),
                        "cooc_boost": round(item.get("cooc_boost", 0.0), 4),
                    },
                }

                # Add distance if available (semantic results have it, FTS-only don't)
                if item.get("distance") is not None:
                    result_item["distance"] = round(item["distance"], 4)

                # Add rrf_score if hybrid mode
                if use_hybrid and item.get("rrf_score") is not None:
                    result_item["rrf_score"] = round(item["rrf_score"], 6)

                output.append(result_item)
                top_k_ids.append(eid)

        # --- Update shadow mode logging with results ---
        if event_id is not None:
            try:
                # Map entity_id -> baseline_rank for logging
                baseline_map = {
                    r["entity_id"]: r["baseline_rank"] for r in baseline_ranked
                }

                # Prepare results for logging (top-k only)
                results_to_log = []
                for rank_pos, rank_item in enumerate(ranked):
                    eid = rank_item["entity_id"]
                    row = store.get_entity_by_id(eid)
                    if not row:
                        continue

                    log_entry = {
                        "entity_id": eid,
                        "entity_name": row["name"],
                        "rank": rank_pos + 1,
                        "baseline_rank": baseline_map.get(eid),
                    }

                    if treatment == 1:
                        # LIMBIC: include full scoring details
                        log_entry["limbic_score"] = rank_item.get("limbic_score", 0.0)
                        log_entry["cosine_sim"] = max(
                            0.0, 1.0 - rank_item.get("distance", 1.0)
                        )
                        log_entry["importance"] = rank_item.get("importance", 0.0)
                        log_entry["temporal"] = rank_item.get("temporal_factor", 1.0)
                        log_entry["cooc_boost"] = rank_item.get("cooc_boost", 0.0)
                    else:
                        # BASELINE: only cosine similarity
                        log_entry["cosine_sim"] = rank_item.get("cosine_sim", 0.0)

                    results_to_log.append(log_entry)

                store.log_search_results(event_id, results_to_log)

                # Update event with final count and duration
                duration_ms = (time.perf_counter() - shadow_start_time) * 1000
                try:
                    store.db.execute(
                        "UPDATE search_events SET num_results = ?, duration_ms = ? WHERE event_id = ?",
                        (len(output), duration_ms, event_id),
                    )
                    store.db.commit()
                except Exception:
                    pass  # Non-critical

            except Exception as e:
                logger.warning("Shadow mode result logging failed: %s", e)

        # --- Limbic: record access + co-occurrences for top-K (post-response) ---
        if treatment == 1:
            # Only record limbic tracking for limbic queries
            try:
                for eid in top_k_ids:
                    store.record_access(eid)
                if len(top_k_ids) >= 2:
                    store.record_co_occurrences(top_k_ids)
            except Exception as e:
                logger.warning("Limbic tracking failed: %s", e)

        return {"results": output}

    except Exception as e:
        logger.error("Error in search_semantic: %s", e)
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
        logger.error("Error in migrate: %s", e)
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
