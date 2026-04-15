"""Reflection tools for MCP Memory knowledge graph."""

import logging
import math
from datetime import datetime, timezone
from typing import Any

import mcp_memory.server as _server_mod
from mcp_memory.config import MAX_ENTITIES_PER_CALL, MAX_QUERY_LENGTH
from mcp_memory._helpers import _get_engine

logger = logging.getLogger(__name__)


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
        store = _server_mod.store
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
