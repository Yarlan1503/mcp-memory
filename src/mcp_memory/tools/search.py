"""Search tools for MCP Memory knowledge graph."""

import hashlib
import logging
import random
import time
from typing import Any

from mcp_memory.config import (
    BASELINE_PROBABILITY,
    MAX_ENTITIES_PER_CALL,
    MAX_QUERY_LENGTH,
    USE_AB_TESTING,
)
from mcp_memory._helpers import _entity_to_output, _get_engine, tool_error_handler

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================


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

    import mcp_memory.server as _server_mod

    store = _server_mod.store
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
    import mcp_memory.server as _server_mod

    store = _server_mod.store

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
    import mcp_memory.server as _server_mod

    store = _server_mod.store

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


def _cosine_sim_from_distance(distance: float | None) -> float | None:
    """Convert vector distance to cosine similarity when distance exists.

    Hybrid search can include FTS-only results whose distance is intentionally
    None. Preserve that semantic instead of treating None as a numeric value.
    """
    if distance is None:
        return None
    return max(0.0, 1.0 - distance)


def _log_shadow_and_track(
    event_id: int | None,
    ranked: list[dict],
    treatment: int,
    baseline_ranked: list[dict],
    top_k_ids: list[int],
    shadow_start_time: float,
) -> None:
    """Log shadow mode results and record limbic access tracking."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store

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
                    log_entry["cosine_sim"] = _cosine_sim_from_distance(
                        rank_item.get("distance")
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


# ============================================================
# TOOL FUNCTIONS
# ============================================================


@tool_error_handler
def search_nodes(query: str) -> dict[str, Any]:
    """Search for nodes in the knowledge graph by name, type, or observation content."""
    start_time = time.perf_counter()
    event_id = None
    import mcp_memory.server as _server_mod

    store = _server_mod.store

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


@tool_error_handler
def open_nodes(
    names: list[str], kinds: list[str] | None = None, include_superseded: bool = False
) -> dict[str, Any]:
    """Open specific nodes by name. Returns full entity data with observations.

    Args:
        names: List of entity names to open.
        kinds: Optional filter — only include observations matching these kinds.
            If None or empty, include all kinds.
        include_superseded: If True, include superseded observations. Default False."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store

    if len(names) > MAX_ENTITIES_PER_CALL:
        return {
            "error": f"Too many entity names: {len(names)} > {MAX_ENTITIES_PER_CALL}. Request fewer entities at a time."
        }

    # 1. Resolve names to entities (preserve input order)
    entities_list = []
    for name in names:
        entity = store.get_entity_by_name(name)
        if entity:
            entities_list.append(entity)

    if not entities_list:
        return {"entities": []}

    entity_ids = [e["id"] for e in entities_list]

    # 2. Batch prefetch (4 queries total)
    obs_map = store.get_observations_with_ids_batch(
        entity_ids, exclude_superseded=not include_superseded
    )
    rel_map = store.get_relations_for_entity_batch(entity_ids)
    ref_map = store.get_reflections_for_target_batch("entity", entity_ids)

    # 3. Reconstruct results preserving input order
    results = []
    opened_ids = []
    for entity in entities_list:
        eid = entity["id"]
        obs_data = obs_map.get(eid, [])
        if kinds:
            obs_data = [o for o in obs_data if o["kind"] in kinds]

        rel_output = [
            {
                "relation_type": r["relation_type"],
                "target_name": r["target_name"],
                "direction": r["direction"],
                "context": r["context"],
                "active": bool(r["active"]),
                "ended_at": r["ended_at"],
            }
            for r in rel_map.get(eid, [])
        ]

        entity_reflections = ref_map.get(eid, [])

        store.record_access(eid)
        opened_ids.append(eid)

        result_dict = _entity_to_output(entity, obs_data, relations=rel_output)
        result_dict["reflections"] = entity_reflections
        results.append(result_dict)

    # Limbic: record co-occurrences for entities opened together (best-effort)
    try:
        if len(opened_ids) >= 2:
            store.record_co_occurrences(opened_ids)
    except Exception as e:
        logger.warning("Limbic co-occurrence tracking failed: %s", e)

    return {"entities": results}


@tool_error_handler
def search_semantic(query: str, limit: int = 10) -> dict[str, Any]:
    """Semantic search using vector embeddings with optional full-text hybrid search.
    Combines semantic (KNN) and text (FTS5) results via Reciprocal Rank Fusion,
    then applies limbic re-ranking based on access patterns and co-occurrence.
    Requires the embedding model to be downloaded (run download_model.py first)."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store

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

    from mcp_memory.scoring import detect_query_type

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
            num_results=0,  # Will update after ranking/logging
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
