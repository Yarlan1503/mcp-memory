import hashlib
import logging
import math
import random
import sys
import time
from typing import Any

from fastmcp import FastMCP

from mcp_memory.entity_splitter import (
    analyze_entity_for_split,
    propose_entity_split,
    execute_entity_split,
    find_all_split_candidates,
    _get_threshold,
)
from mcp_memory.models import EntityInput, RelationInput
from mcp_memory.storage import MemoryStore

logger = logging.getLogger(__name__)

mcp = FastMCP("memory")

# Instantiate store — default path
store = MemoryStore()
store.init_db()

# ============================================================
# Input size limits
# ============================================================
MAX_OBSERVATIONS_PER_CALL = 100
MAX_ENTITIES_PER_CALL = 50
MAX_OBSERVATION_LENGTH = 2000
MAX_QUERY_LENGTH = 500


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


def _entity_to_output(
    row: dict,
    observations: list[str] | list[dict],
    relations: list[dict] | None = None,
) -> dict:
    """Convert DB row + observations to EntityOutput dict.
    Optionally includes relations with context, active, and ended_at."""
    obs_list = observations
    output = {
        "name": row["name"],
        "entityType": row["entity_type"],
        "status": row.get("status", "activo"),
        "observations": obs_list,
    }
    if relations is not None:
        output["relations"] = relations
    return output


def _recompute_embedding(entity_id: int, name: str, entity_type: str) -> None:
    """Recompute and store embedding for an entity (if engine available)."""
    engine = _get_engine()
    if not engine or not getattr(engine, "available", False):
        return
    try:
        from mcp_memory.embeddings import serialize_f32

        # Get entity status
        entity_data = store.get_entity_by_id(entity_id)
        status = entity_data.get("status", "activo") if entity_data else "activo"

        # Get observations with IDs (for kind info)
        obs_data = store.get_observations_with_ids(entity_id, exclude_superseded=True)
        relations = store.get_relations_for_entity(entity_id)
        text = engine.prepare_entity_text(
            name, entity_type, obs_data, relations, status=status
        )
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
        # Input validation
        if len(entities) > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Too many entities: {len(entities)} > {MAX_ENTITIES_PER_CALL}. Create fewer entities at a time."
            }
        for i, entity_dict in enumerate(entities):
            obs_list = entity_dict.get("observations", [])
            if len(obs_list) > MAX_OBSERVATIONS_PER_CALL:
                return {
                    "error": f"Entity {i} has too many observations: {len(obs_list)} > {MAX_OBSERVATIONS_PER_CALL}. Split into smaller batches."
                }
            for j, obs in enumerate(obs_list):
                if len(obs) > MAX_OBSERVATION_LENGTH:
                    return {
                        "error": f"Entity {i}, observation {j} too long: {len(obs)} > {MAX_OBSERVATION_LENGTH} characters. Shorten or split the observation."
                    }

        results = []
        for entity_dict in entities:
            parsed = EntityInput.model_validate(entity_dict)

            # Upsert entity
            entity_id = store.upsert_entity(
                parsed.name, parsed.entityType, parsed.status
            )

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
            _recompute_embedding(entity_id, parsed.name, parsed.entityType)

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
                from_ent["id"], to_ent["id"], parsed.relationType, parsed.context
            )
            if created:
                result_item = {
                    "from": parsed.from_entity,
                    "to": parsed.to_entity,
                    "relationType": parsed.relationType,
                }
                if parsed.context:
                    result_item["context"] = parsed.context
                results.append(result_item)
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
def add_observations(
    name: str,
    observations: list[str],
    kind: str = "generic",
    supersedes: int | None = None,
) -> dict[str, Any]:
    """Add observations to an existing entity.

    Args:
        name: Entity name.
        observations: List of observation strings to add.
        kind: The kind/type of observations (default 'generic').
        supersedes: Optional observation ID to supersede. The referenced obs
            will be marked as superseded, and the new obs will reference it."""
    try:
        # Input validation
        if len(observations) > MAX_OBSERVATIONS_PER_CALL:
            return {
                "error": f"Too many observations: {len(observations)} > {MAX_OBSERVATIONS_PER_CALL}. Split into smaller batches."
            }
        for i, obs in enumerate(observations):
            if len(obs) > MAX_OBSERVATION_LENGTH:
                return {
                    "error": f"Observation {i} too long: {len(obs)} > {MAX_OBSERVATION_LENGTH} characters. Shorten or split the observation."
                }

        entity = store.get_entity_by_name(name)
        if not entity:
            return {"error": f"Entity not found: {name}"}

        store.add_observations(
            entity["id"], observations, kind=kind, supersedes=supersedes
        )
        all_obs = store.get_observations(entity["id"])

        # Recompute embedding
        _recompute_embedding(entity["id"], entity["name"], entity["entity_type"])

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
        if len(entityNames) > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Too many entity names: {len(entityNames)} > {MAX_ENTITIES_PER_CALL}. Delete fewer entities at a time."
            }

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
        _recompute_embedding(entity["id"], entity["name"], entity["entity_type"])

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
    start_time = time.perf_counter()
    event_id = None
    try:
        if len(query) > MAX_QUERY_LENGTH:
            return {
                "error": f"Query too long: {len(query)} > {MAX_QUERY_LENGTH} characters. Use a more concise search query."
            }

        if not query.strip():
            return {"entities": []}

        # --- SHADOW MODE: Log to database (best-effort, non-blocking) ---
        try:
            event_id = store.log_search_event(
                query_text=query,
                treatment=-1,  # LIKE search, no A/B test
                k_limit=0,
                num_results=0,  # Will update after query
                duration_ms=None,
                engine_used="like",
            )
        except Exception as e:
            logger.warning("Shadow mode event logging failed: %s", e)
            event_id = None

        entities = store.search_entities(query)
        results = []
        for entity in entities:
            obs = store.get_observations(entity["id"])
            results.append(_entity_to_output(entity, obs))

        # Update event with final count and duration
        if event_id is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            try:
                store.update_search_event_completion(
                    event_id=event_id,
                    num_results=len(results),
                    duration_ms=duration_ms,
                    engine_used="like",
                )
            except Exception:
                pass  # Non-critical

        return {"entities": results}
    except Exception as e:
        logger.error("Error in search_nodes: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 8: open_nodes
# ============================================================
@mcp.tool
def open_nodes(
    names: list[str], kinds: list[str] | None = None, include_superseded: bool = False
) -> dict[str, Any]:
    """Open specific nodes by name. Returns full entity data with observations.

    Args:
        names: List of entity names to open.
        kinds: Optional filter — only include observations matching these kinds.
            If None or empty, include all kinds.
        include_superseded: If True, include superseded observations. Default False."""
    try:
        if len(names) > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Too many entity names: {len(names)} > {MAX_ENTITIES_PER_CALL}. Request fewer entities at a time."
            }

        results = []
        opened_ids = []
        for name in names:
            entity = store.get_entity_by_name(name)
            if entity:
                obs_data = store.get_observations_with_ids(
                    entity["id"], exclude_superseded=not include_superseded
                )
                if kinds:
                    obs_data = [o for o in obs_data if o["kind"] in kinds]

                # Get relations with context, active, ended_at
                entity_relations = store.get_relations_for_entity(entity["id"])
                rel_output = [
                    {
                        "relation_type": r["relation_type"],
                        "target_name": r["target_name"],
                        "direction": r["direction"],
                        "context": r["context"],
                        "active": bool(r["active"]),
                        "ended_at": r["ended_at"],
                    }
                    for r in entity_relations
                ]

                # Limbic: record access on node open
                store.record_access(entity["id"])
                opened_ids.append(entity["id"])
                result_dict = _entity_to_output(entity, obs_data, relations=rel_output)
                # Phase 4: Add reflections for this entity
                entity_reflections = store.get_reflections_for_target(
                    "entity", entity["id"]
                )
                result_dict["reflections"] = [
                    {
                        "id": r["id"],
                        "author": r["author"],
                        "content": r["content"],
                        "mood": r["mood"],
                        "created_at": r["created_at"],
                    }
                    for r in entity_reflections
                ]
                results.append(result_dict)

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


def _execute_hybrid_search(
    query: str, engine: Any, limit: int
) -> tuple[list[dict], list[dict], list[dict] | None, list[int], bool]:
    """Execute KNN + FTS5 search and merge via RRF.

    Returns: (knn_results, fts_results, merged, candidate_ids, use_hybrid)
    """
    from mcp_memory.embeddings import serialize_f32
    from mcp_memory.scoring import EXPANSION_FACTOR, reciprocal_rank_fusion

    query_vector = engine.encode([query], task="query")
    query_bytes = serialize_f32(query_vector[0])
    expanded_limit = limit * EXPANSION_FACTOR
    knn_results = store.search_embeddings(query_bytes, limit=expanded_limit)

    if not knn_results:
        return [], [], None, [], False

    fts_results = store.search_fts(query, limit=expanded_limit)

    use_hybrid = len(fts_results) > 0
    if use_hybrid:
        merged = reciprocal_rank_fusion(knn_results, fts_results)
        candidate_ids = [r["entity_id"] for r in merged]
    else:
        merged = None
        candidate_ids = [r["entity_id"] for r in knn_results]

    return knn_results, fts_results, merged, candidate_ids, use_hybrid


def _rank_candidates(
    treatment: int,
    routing_strategy: Any,
    knn_results: list[dict],
    merged: list[dict] | None,
    use_hybrid: bool,
    access_data: dict,
    degree_data: dict,
    cooc_data: dict,
    entity_created: dict[int, str],
    access_days_data: dict,
    limit: int,
    baseline_ranked: list[dict],
) -> list[dict]:
    """Rank candidates based on treatment and routing strategy.

    Returns: ranked list of dicts with entity_id, scores, etc.
    """
    from mcp_memory.scoring import (
        RoutingStrategy,
        rank_candidates,
        rank_hybrid_candidates,
        rank_with_routing_strategy,
    )

    if treatment == 0:
        return baseline_ranked[:limit]

    # LIMBIC: Apply limbic re-ranking with dynamic routing
    if routing_strategy == RoutingStrategy.HYBRID_BALANCED:
        if use_hybrid:
            return rank_hybrid_candidates(
                merged_results=merged,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                access_days_data=access_days_data,
            )
        else:
            return rank_candidates(
                knn_results=knn_results,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                access_days_data=access_days_data,
            )
    elif routing_strategy in (
        RoutingStrategy.COSINE_HEAVY,
        RoutingStrategy.LIMBIC_HEAVY,
    ):
        if use_hybrid:
            return rank_with_routing_strategy(
                merged_results=merged,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                strategy=routing_strategy,
                access_days_data=access_days_data,
            )
        else:
            knn_for_routing = [
                {
                    "entity_id": r["entity_id"],
                    "distance": r["distance"],
                    "rrf_score": 1.0,
                }
                for r in knn_results
            ]
            return rank_with_routing_strategy(
                merged_results=knn_for_routing,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                strategy=routing_strategy,
                access_days_data=access_days_data,
            )
    else:
        # Fallback (shouldn't happen): use original limbic scoring
        if use_hybrid:
            return rank_hybrid_candidates(
                merged_results=merged,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                access_days_data=access_days_data,
            )
        else:
            return rank_candidates(
                knn_results=knn_results,
                access_data=access_data,
                degree_data=degree_data,
                cooc_data=cooc_data,
                entity_created=entity_created,
                limit=limit,
                access_days_data=access_days_data,
            )


def _apply_deboosts(ranked: list[dict], treatment: int) -> list[dict]:
    """Apply status and metadata deboosts to ranked results.

    Modifies limbic_score in-place and returns ranked.
    """
    STATUS_MULTIPLIERS = {
        "activo": 1.0,
        "pausado": 0.85,
        "completado": 0.7,
        "archivado": 0.5,
    }

    if treatment == 1:
        # Batch prefetch: entities + obs_with_ids (2 queries total instead of 2N)
        entity_ids = [item["entity_id"] for item in ranked]
        entities_map = store.get_entities_batch(entity_ids)
        obs_map = store.get_observations_with_ids_batch(
            entity_ids, exclude_superseded=True
        )

        for item in ranked:
            eid = item["entity_id"]
            row = entities_map.get(eid)
            if row:
                status = row.get("status", "activo")
                multiplier = STATUS_MULTIPLIERS.get(status, 1.0)
                if multiplier < 1.0 and "limbic_score" in item:
                    item["limbic_score"] *= multiplier

            # De-boost metadata-heavy entities
            try:
                obs_data = obs_map.get(eid, [])
                if obs_data:
                    metadata_ratio = sum(
                        1 for o in obs_data if o.get("kind") == "metadata"
                    ) / len(obs_data)
                    if metadata_ratio > 0.5 and "limbic_score" in item:
                        item["limbic_score"] = round(
                            round(item["limbic_score"], 4) * 0.7, 4
                        )
            except Exception:
                pass  # Non-critical

    return ranked


def _build_search_output(
    ranked: list[dict],
    treatment: int,
    routing_strategy: Any,
    use_hybrid: bool,
) -> tuple[list[dict], list[int]]:
    """Build the output result list from ranked candidates.

    Returns: (output_list, top_k_entity_ids)
    """
    output = []
    top_k_ids = []

    # Batch prefetch: entities + observations (2 queries total instead of 2N)
    entity_ids = [item["entity_id"] for item in ranked]
    entities_map = store.get_entities_batch(entity_ids)
    obs_map = store.get_observations_batch(entity_ids)

    if treatment == 0:
        for item in ranked:
            eid = item["entity_id"]
            row = entities_map.get(eid)
            if not row:
                continue

            obs = obs_map.get(eid, [])
            result_item = {
                "name": row["name"],
                "entityType": row["entity_type"],
                "observations": obs,
                "distance": round(1.0 - item.get("cosine_sim", 0.0), 4),
                "treatment": "baseline",
            }

            result_item["cosine_sim"] = round(item.get("cosine_sim", 0.0), 4)

            output.append(result_item)
            top_k_ids.append(eid)
    else:
        for item in ranked:
            eid = item["entity_id"]
            row = entities_map.get(eid)
            if not row:
                continue

            obs = obs_map.get(eid, [])
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
                "routing_strategy": item.get(
                    "routing_strategy",
                    routing_strategy.value if routing_strategy else "hybrid_balanced",
                ),
            }

            if item.get("distance") is not None:
                result_item["distance"] = round(item["distance"], 4)

            if use_hybrid and item.get("rrf_score") is not None:
                result_item["rrf_score"] = round(item["rrf_score"], 6)

            output.append(result_item)
            top_k_ids.append(eid)

    return output, top_k_ids


def _log_shadow_and_track(
    event_id: int | None,
    ranked: list[dict],
    treatment: int,
    baseline_ranked: list[dict],
    top_k_ids: list[int],
    shadow_start_time: float,
) -> None:
    """Log shadow mode results and record limbic access tracking."""
    if event_id is not None:
        try:
            baseline_map = {r["entity_id"]: r["baseline_rank"] for r in baseline_ranked}

            # Batch prefetch entity names
            log_entity_ids = [r["entity_id"] for r in ranked]
            entities_map = store.get_entities_batch(log_entity_ids)

            results_to_log = []
            for rank_pos, rank_item in enumerate(ranked):
                eid = rank_item["entity_id"]
                row = entities_map.get(eid)
                if not row:
                    continue

                log_entry = {
                    "entity_id": eid,
                    "entity_name": row["name"],
                    "rank": rank_pos + 1,
                    "baseline_rank": baseline_map.get(eid),
                }

                if treatment == 1:
                    log_entry["limbic_score"] = rank_item.get("limbic_score", 0.0)
                    log_entry["cosine_sim"] = max(
                        0.0, 1.0 - rank_item.get("distance", 1.0)
                    )
                    log_entry["importance"] = rank_item.get("importance", 0.0)
                    log_entry["temporal"] = rank_item.get("temporal_factor", 1.0)
                    log_entry["cooc_boost"] = rank_item.get("cooc_boost", 0.0)
                else:
                    log_entry["cosine_sim"] = rank_item.get("cosine_sim", 0.0)

                results_to_log.append(log_entry)

            store.log_search_results(event_id, results_to_log)

            duration_ms = (time.perf_counter() - shadow_start_time) * 1000
            try:
                store.update_search_event_completion(
                    event_id=event_id,
                    num_results=len(top_k_ids),
                    duration_ms=duration_ms,
                    engine_used="limbic" if treatment == 1 else "baseline",
                )
            except Exception:
                pass  # Non-critical

        except Exception as e:
            logger.warning("Shadow mode result logging failed: %s", e)

    if treatment == 1:
        try:
            for eid in top_k_ids:
                store.record_access(eid)
            if len(top_k_ids) >= 2:
                store.record_co_occurrences(top_k_ids)
        except Exception as e:
            logger.warning("Limbic tracking failed: %s", e)


@mcp.tool
def search_semantic(query: str, limit: int = 10) -> dict[str, Any]:
    """Semantic search using vector embeddings with optional full-text hybrid search.
    Combines semantic (KNN) and text (FTS5) results via Reciprocal Rank Fusion,
    then applies limbic re-ranking based on access patterns and co-occurrence.
    Requires the embedding model to be downloaded (run download_model.py first)."""
    try:
        if len(query) > MAX_QUERY_LENGTH:
            return {
                "error": f"Query too long: {len(query)} > {MAX_QUERY_LENGTH} characters. Use a more concise search query."
            }
        if limit > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Limit too high: {limit} > {MAX_ENTITIES_PER_CALL}. Request fewer results."
            }

        engine = _get_engine()
        if not engine or not engine.available:
            return {
                "error": "Embedding model not available. Run 'python scripts/download_model.py' to download the model first.",
            }

        from mcp_memory.scoring import RoutingStrategy, detect_query_type

        # Determine treatment and routing
        treatment = _get_treatment(query)
        routing_strategy = None
        if treatment == 1:
            routing_strategy = detect_query_type(query, limit)
            logger.info(f"Query routing: '{query[:50]}...' -> {routing_strategy.value}")

        # Execute search
        knn_results, fts_results, merged, candidate_ids, use_hybrid = (
            _execute_hybrid_search(query, engine, limit)
        )
        if not knn_results:
            return {"results": []}

        # Collect data
        baseline_ranked = _compute_baseline_ranking(knn_results)
        access_data = store.get_access_data(candidate_ids)
        degree_data = store.get_entity_degrees(candidate_ids)
        cooc_data = store.get_co_occurrences(candidate_ids)
        access_days_data = store.get_access_days(candidate_ids)

        entity_created = {
            eid: row["created_at"]
            for eid, row in store.get_entities_batch(candidate_ids).items()
        }

        # Log shadow event
        shadow_start_time = time.perf_counter()
        try:
            event_id = store.log_search_event(
                query_text=query,
                treatment=treatment,
                k_limit=limit,
                num_results=None,
                duration_ms=None,
                engine_used="limbic" if treatment == 1 else "baseline",
            )
        except Exception as e:
            logger.warning("Shadow mode event logging failed: %s", e)
            event_id = None

        # Rank and apply deboosts
        ranked = _rank_candidates(
            treatment,
            routing_strategy,
            knn_results,
            merged,
            use_hybrid,
            access_data,
            degree_data,
            cooc_data,
            entity_created,
            access_days_data,
            limit,
            baseline_ranked,
        )
        ranked = _apply_deboosts(ranked, treatment)

        # Build output
        output, top_k_ids = _build_search_output(
            ranked, treatment, routing_strategy, use_hybrid
        )

        # Log and track
        _log_shadow_and_track(
            event_id, ranked, treatment, baseline_ranked, top_k_ids, shadow_start_time
        )

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
# TOOL 12: analyze_entity_split
# ============================================================
@mcp.tool
def analyze_entity_split(entity_name: str) -> dict[str, Any]:
    """Analyze an entity and determine if it needs to be split.

    An entity is a candidate for splitting when it exceeds the observation
    threshold for its type (Sesion=15, Proyecto=25, otras=20) and has
    sufficient topic diversity.

    Returns a dict with analysis results including entity metadata,
    observation count, threshold, detected topics, and split score."""
    try:
        analysis = analyze_entity_for_split(store, entity_name)
        if analysis is None:
            return {"error": f"Entity not found: {entity_name}"}
        return {"analysis": analysis}
    except Exception as e:
        logger.error("Error in analyze_entity_split: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 13: propose_entity_split
# ============================================================
@mcp.tool
def propose_entity_split_tool(entity_name: str) -> dict[str, Any]:
    """Analyze and propose a split for an entity if needed.

    Returns a split proposal with suggested new entity names,
    topic groupings, and relations to create.
    Returns None if the entity doesn't need splitting."""
    try:
        proposal = propose_entity_split(store, entity_name)
        if proposal is None:
            return {"proposal": None, "message": "Entity does not need splitting"}
        return {"proposal": proposal}
    except Exception as e:
        logger.error("Error in propose_entity_split_tool: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 14: execute_entity_split
# ============================================================
@mcp.tool
def execute_entity_split_tool(
    entity_name: str,
    approved_splits: list[dict[str, Any]],
    parent_entity_name: str | None = None,
) -> dict[str, Any]:
    """Execute an approved entity split.

    Creates new entities from the approved splits, moves observations,
    and establishes contiene/parte_de relations between parent and children.

    Args:
        entity_name: Name of the original entity to split
        approved_splits: List of approved split definitions, each with
            name, entity_type, and observations
        parent_entity_name: Optional explicit parent name (defaults to entity_name)"""
    try:
        result = execute_entity_split(
            store, entity_name, approved_splits, parent_entity_name
        )
        return {"result": result}
    except Exception as e:
        logger.error("Error in execute_entity_split_tool: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 15: find_split_candidates
# ============================================================
@mcp.tool
def find_split_candidates() -> dict[str, Any]:
    """Find all entities in the knowledge graph that are candidates for splitting.

    Scans all entities and returns those that exceed their type-specific
    observation threshold and have sufficient topic diversity."""
    try:
        candidates = find_all_split_candidates(store)
        return {"candidates": candidates}
    except Exception as e:
        logger.error("Error in find_split_candidates: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 16: find_duplicate_observations
# ============================================================
@mcp.tool
def find_duplicate_observations(
    entity_name: str, threshold: float = 0.85, containment_threshold: float = 0.7
) -> dict[str, Any]:
    """Find observations that may be semantically duplicated within an entity.
    Returns pairs of observations with similarity score above threshold.
    Use this to review and consolidate redundant observations.
    Uses combined similarity: cosine >= threshold OR containment >= containment_threshold
    when texts have asymmetric length (one is 2x+ longer than the other)."""
    try:
        # 1. Find entity
        entity = store.get_entity_by_name(entity_name)
        if not entity:
            return {"error": f"Entity not found: {entity_name}"}

        # 2. Get observations with metadata
        observations = store.get_observations_with_ids(entity["id"])
        if len(observations) < 2:
            return {
                "entity": entity_name,
                "total_observations": len(observations),
                "clusters": [],
            }

        # 3. Get embedding engine
        engine = _get_engine()
        if not engine or not engine.available:
            return {
                "error": "Embedding model not available. "
                "Run 'python scripts/download_model.py' to download the model first.",
            }

        # 4. Import combined_similarity once (not inside loop)
        from mcp_memory.scoring import combined_similarity

        # 5. Encode all observations (single batch)
        contents = [o["content"] for o in observations]
        embeddings = engine.encode(contents)  # (n, 384)

        # 6. Pairwise cosine similarities (L2-normalised → dot product)
        import numpy as np

        sim_matrix = embeddings @ embeddings.T  # (n, n)

        # 7. Union-Find for clustering above threshold
        n = len(observations)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        pairs: list[dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i][j])
                if combined_similarity(
                    sim,
                    contents[i],
                    contents[j],
                    cosine_threshold=threshold,
                    containment_threshold=containment_threshold,
                ):
                    # Determine effective similarity score for display
                    # If cosine alone didn't pass, show containment-based match
                    effective_score = sim
                    pairs.append(
                        {
                            "observation_a": {
                                "id": observations[i]["id"],
                                "content": observations[i]["content"],
                                "similarity_flag": observations[i]["similarity_flag"],
                            },
                            "observation_b": {
                                "id": observations[j]["id"],
                                "content": observations[j]["content"],
                                "similarity_flag": observations[j]["similarity_flag"],
                            },
                            "similarity_score": round(effective_score, 4),
                            "match_type": (
                                "cosine" if sim >= threshold else "containment"
                            ),
                        }
                    )
                    union(i, j)

        # 8. Build clusters from union-find
        clusters_map: dict[int, list[dict[str, Any]]] = {}
        for idx in range(n):
            root = find(idx)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(
                {
                    "observation_id": observations[idx]["id"],
                    "content": observations[idx]["content"],
                    "similarity_flag": observations[idx]["similarity_flag"],
                }
            )

        # Only keep clusters with 2+ members (actual duplicates)
        clusters = [
            {"observations": members, "size": len(members)}
            for members in clusters_map.values()
            if len(members) > 1
        ]

        return {
            "entity": entity_name,
            "total_observations": n,
            "duplicates_found": len(pairs),
            "clusters": clusters,
        }

    except Exception as e:
        logger.error("Error in find_duplicate_observations: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 17: consolidation_report
# ============================================================
def _preview_content(content: str, max_len: int = 80) -> str:
    """Truncate content for preview display."""
    if len(content) <= max_len:
        return content
    return content[:max_len] + "..."


def _days_since_access_sql(last_access: str | None) -> int | None:
    """Compute days since last access using SQLite julianday (avoids timezone issues)."""
    if last_access is None:
        return None
    try:
        row = store.db.execute(
            "SELECT CAST(julianday('now') - julianday(?) AS INTEGER)",
            (last_access,),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


@mcp.tool
def consolidation_report(stale_days: float = 90.0) -> dict[str, Any]:
    """Generate a memory consolidation report without making any changes.

    Analyzes the knowledge graph for:
    - Split candidates: entities exceeding observation thresholds with sufficient topic diversity
    - Flagged observations: observations marked as potential duplicates (similarity_flag=1)
    - Stale entities: entities not accessed in N days with low access count
    - Large entities: entities exceeding size thresholds (may need splitting)

    Use this report to decide what consolidation actions to take.
    The report is read-only — no changes are made to the knowledge graph."""
    try:
        # 1. Collect raw data from storage (read-only queries)
        data = store.get_consolidation_data(stale_days)

        # 2. Get split candidates from entity_splitter
        split_candidates_raw = find_all_split_candidates(store)
        split_candidate_names = {c["entity_name"] for c in split_candidates_raw}

        # 3. Build split_candidates with recommendations
        split_candidates: list[dict[str, Any]] = []
        for c in split_candidates_raw:
            ratio = (
                c["observation_count"] / c["threshold"] if c["threshold"] > 0 else 0.0
            )
            split_candidates.append(
                {
                    "entity_name": c["entity_name"],
                    "entity_type": c["entity_type"],
                    "observation_count": c["observation_count"],
                    "threshold": c["threshold"],
                    "topic_diversity": round(c["split_score"], 2),
                    "recommendation": (
                        f"Split recommended — exceeds {c['threshold']} "
                        f"by {ratio:.1f}x with topic diversity {c['split_score']:.2f}"
                    ),
                }
            )

        # 4. Build flagged_observations with content preview
        flagged: dict[str, list[dict[str, Any]]] = {}
        for entity_name, obs_list in data["flagged_observations"].items():
            flagged[entity_name] = [
                {
                    "id": o["id"],
                    "content_preview": _preview_content(o["content"]),
                    "content_length": len(o["content"]),
                }
                for o in obs_list
            ]

        # 5. Build stale_entities with recommendations and days_since_access
        stale_entities: list[dict[str, Any]] = []
        for s in data["stale_entities"]:
            days = _days_since_access_sql(s["last_access"])
            if days is None:
                recommendation = "Never accessed — consider archiving or deleting"
            else:
                recommendation = f"Not accessed in {days} days — consider archiving"
            stale_entities.append(
                {
                    "entity_name": s["entity_name"],
                    "entity_type": s["entity_type"],
                    "status": s.get("status", "activo"),
                    "observation_count": s["observation_count"],
                    "last_access": s["last_access"],
                    "access_count": s["access_count"],
                    "days_since_access": days,
                    "recommendation": recommendation,
                }
            )

        # 6. Build large_entities
        # Entities exceeding their type threshold but NOT identified as split
        # candidates by entity_splitter. With the current splitting algorithm,
        # all entities exceeding the threshold are split candidates, so this
        # list may be empty. Included for future algorithm changes.
        large_entities: list[dict[str, Any]] = []
        for entity_name, obs_count in data["entity_sizes"].items():
            if entity_name in split_candidate_names:
                continue
            entity = store.get_entity_by_name(entity_name)
            if entity is None:
                continue
            threshold = _get_threshold(entity["entity_type"])
            if obs_count > threshold:
                ratio = obs_count / threshold if threshold > 0 else 0.0
                if ratio < 1.5:
                    rec = "Near threshold — monitor growth"
                else:
                    rec = f"Exceeds threshold by {ratio:.1f}x — evaluate for split"
                large_entities.append(
                    {
                        "entity_name": entity_name,
                        "entity_type": entity["entity_type"],
                        "observation_count": obs_count,
                        "threshold": threshold,
                        "recommendation": rec,
                    }
                )

        # 7. Count total flagged observations
        total_flagged = sum(len(v) for v in data["flagged_observations"].values())

        return {
            "summary": {
                "total_entities": data["total_entities"],
                "total_observations": data["total_observations"],
                "split_candidates_count": len(split_candidates),
                "flagged_observations_count": total_flagged,
                "stale_entities_count": len(stale_entities),
                "large_entities_count": len(large_entities),
            },
            "split_candidates": split_candidates,
            "flagged_observations": flagged,
            "stale_entities": stale_entities,
            "large_entities": large_entities,
        }

    except Exception as e:
        logger.error("Error in consolidation_report: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 18: add_reflection
# ============================================================
@mcp.tool
def add_reflection(
    target_type: str,
    target_id: int | None = None,
    author: str = "sofia",
    content: str = "",
    mood: str | None = None,
) -> dict[str, Any]:
    """Add a narrative reflection to give context and meaning to a memory.

    Args:
        target_type: What this reflection targets — 'entity', 'session', 'relation', or 'global'.
        target_id: ID of the target entity/relation. Required for entity/session/relation. NULL for global.
        author: Who wrote this — 'nolan' or 'sofia'.
        content: The reflection text (free prose, no prefixes).
        mood: Optional mood — 'frustracion', 'satisfaccion', 'curiosidad', 'duda', 'insight'.
    """
    try:
        if not content.strip():
            return {"error": "content cannot be empty"}

        result = store.add_reflection(
            target_type=target_type,
            target_id=target_id,
            author=author,
            content=content,
            mood=mood,
        )
        if result is None:
            return {
                "error": "Invalid parameters — check target_type, author, and mood values"
            }
        return {"reflection": result}
    except Exception as e:
        logger.error("Error in add_reflection: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 19: search_reflections
# ============================================================
@mcp.tool
def search_reflections(
    query: str,
    author: str | None = None,
    mood: str | None = None,
    target_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search reflections by semantic similarity and optional filters.

    Combines semantic (KNN) and text (FTS5) results via Reciprocal Rank Fusion.

    Args:
        query: Search text.
        author: Filter by author ('nolan' or 'sofia').
        mood: Filter by mood.
        target_type: Filter by target type.
        limit: Max results (default 10).
    """
    try:
        if len(query) > MAX_QUERY_LENGTH:
            return {
                "error": f"Query too long: {len(query)} > {MAX_QUERY_LENGTH} characters. Use a more concise search query."
            }
        if limit > MAX_ENTITIES_PER_CALL:
            return {
                "error": f"Limit too high: {limit} > {MAX_ENTITIES_PER_CALL}. Request fewer results."
            }

        engine = _get_engine()
        if not engine or not engine.available:
            return {
                "error": "Embedding model not available. Run 'python scripts/download_model.py' to download the model first.",
            }

        from mcp_memory.embeddings import serialize_f32
        from mcp_memory.scoring import reciprocal_rank_fusion

        # Step 1: Semantic KNN search on reflection_embeddings
        query_vector = engine.encode([query], task="query")
        query_bytes = serialize_f32(query_vector[0])
        expanded_limit = limit * 3  # Over-retrieve for filtering
        knn_results = store.search_reflection_embeddings(
            query_bytes, limit=expanded_limit
        )

        # Step 2: FTS5 search on reflection_fts
        fts_results = store.search_reflection_fts(query, limit=expanded_limit)

        # Step 3: RRF merge (adapt format for RRF function)
        # reciprocal_rank_fusion expects "entity_id" but we have "id" — rename
        knn_for_rrf = [
            {"entity_id": r["id"], "distance": r["distance"]} for r in knn_results
        ]
        fts_for_rrf = [{"entity_id": r["id"], "rank": r["rank"]} for r in fts_results]

        use_hybrid = len(fts_results) > 0
        if use_hybrid:
            merged = reciprocal_rank_fusion(knn_for_rrf, fts_for_rrf)
            candidate_ids = [r["entity_id"] for r in merged]
        else:
            merged = None
            candidate_ids = [r["entity_id"] for r in knn_for_rrf]

        if not candidate_ids:
            return {"results": []}

        # Step 4: Fetch reflection data and apply filters
        placeholders = ",".join("?" for _ in candidate_ids)

        query_sql = f"""
            SELECT r.id, r.target_type, r.target_id, r.author, r.content, r.mood, r.created_at
            FROM reflections r
            WHERE r.id IN ({placeholders})
        """
        params: list[Any] = list(candidate_ids)

        # Apply filters as SQL WHERE clauses
        conditions = []
        if author:
            conditions.append("r.author = ?")
            params.append(author)
        if mood:
            conditions.append("r.mood = ?")
            params.append(mood)
        if target_type:
            conditions.append("r.target_type = ?")
            params.append(target_type)

        if conditions:
            query_sql += " AND " + " AND ".join(conditions)

        rows = store.db.execute(query_sql, params).fetchall()

        # Build result map for scoring
        row_map = {r["id"]: dict(r) for r in rows}

        # Step 5: Apply recency scoring and sort
        # Recency factor: max(0.1, exp(-0.0001 * hours_since_creation))
        # Recent reflections rank higher than old ones
        REFLECTION_TEMPORAL_FLOOR = 0.1
        REFLECTION_TEMPORAL_LAMBDA = 0.0001  # Same half-life as entity scoring

        from datetime import datetime, timezone

        results = []
        if merged:
            for item in merged:
                rid = item["entity_id"]
                if rid in row_map:
                    result_item = row_map[rid]
                    base_score = item["rrf_score"]
                    # Compute recency factor from created_at
                    created_at_str = result_item.get("created_at", "")
                    recency = 1.0
                    if created_at_str:
                        try:
                            dt = datetime.strptime(
                                created_at_str, "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=timezone.utc)
                            hours = max(
                                0.0,
                                (datetime.now(timezone.utc) - dt).total_seconds()
                                / 3600.0,
                            )
                            recency = max(
                                REFLECTION_TEMPORAL_FLOOR,
                                math.exp(-REFLECTION_TEMPORAL_LAMBDA * hours),
                            )
                        except (ValueError, TypeError):
                            pass
                    result_item["score"] = round(base_score * recency, 6)
                    results.append(result_item)
        else:
            # Pure semantic — sort by distance
            for item in knn_for_rrf:
                rid = item["entity_id"]
                if rid in row_map:
                    result_item = row_map[rid]
                    base_score = 1.0 - item["distance"]
                    # Compute recency factor from created_at
                    created_at_str = result_item.get("created_at", "")
                    recency = 1.0
                    if created_at_str:
                        try:
                            dt = datetime.strptime(
                                created_at_str, "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=timezone.utc)
                            hours = max(
                                0.0,
                                (datetime.now(timezone.utc) - dt).total_seconds()
                                / 3600.0,
                            )
                            recency = max(
                                REFLECTION_TEMPORAL_FLOOR,
                                math.exp(-REFLECTION_TEMPORAL_LAMBDA * hours),
                            )
                        except (ValueError, TypeError):
                            pass
                    result_item["score"] = round(base_score * recency, 4)
                    results.append(result_item)

        # Sort by score descending (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        return {"results": results}

    except Exception as e:
        logger.error("Error in search_reflections: %s", e)
        return {"error": str(e)}


# ============================================================
# TOOL 20: end_relation
# ============================================================
@mcp.tool
def end_relation(relation_id: int) -> dict[str, Any]:
    """Expire an active relation by setting active=0 and ended_at=now.

    For inverse pairs (contiene↔parte_de), also expires the inverse relation.

    Args:
        relation_id: The ID of the relation to expire.
    """
    try:
        from mcp_memory.storage import INVERSE_RELATIONS

        # 1. Look up the relation
        rel = store.get_relation_by_id(relation_id)
        if rel is None:
            return {"error": f"Relation with id={relation_id} not found"}

        # 2. Check if already inactive
        if rel["active"] == 0:
            # Already expired — return notice, not error
            return {
                "notice": f"Relation {relation_id} is already inactive",
                "relation": {
                    "id": rel["id"],
                    "from_entity": rel["from_entity"],
                    "to_entity": rel["to_entity"],
                    "relation_type": rel["relation_type"],
                    "active": bool(rel["active"]),
                    "ended_at": rel["ended_at"],
                },
            }

        # 3. Expire the relation
        success = store._end_relation(relation_id)
        if not success:
            return {"error": f"Failed to expire relation {relation_id}"}

        # 4. Check for inverse and expire it too
        inverse_expired = None
        inverse_type = INVERSE_RELATIONS.get(rel["relation_type"])
        if inverse_type:
            # Find the inverse: (to_entity, from_entity, inverse_type)
            inv_row = store.db.execute(
                "SELECT id, active FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
                (rel["to_entity"], rel["from_entity"], inverse_type),
            ).fetchone()
            if inv_row and inv_row["active"] == 1:
                store._end_relation(inv_row["id"])
                inverse_expired = inv_row["id"]

        # 5. Re-fetch to get updated ended_at
        updated = store.get_relation_by_id(relation_id)

        # Resolve entity names for the response
        from_ent = store.get_entity_by_id(rel["from_entity"])
        to_ent = store.get_entity_by_id(rel["to_entity"])

        result = {
            "relation": {
                "id": updated["id"],
                "from": from_ent["name"] if from_ent else str(rel["from_entity"]),
                "to": to_ent["name"] if to_ent else str(rel["to_entity"]),
                "relation_type": updated["relation_type"],
                "active": bool(updated["active"]),
                "ended_at": updated["ended_at"],
                "context": updated["context"],
            }
        }
        if inverse_expired is not None:
            result["inverse_expired_id"] = inverse_expired

        return result

    except Exception as e:
        logger.error("Error in end_relation: %s", e)
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
