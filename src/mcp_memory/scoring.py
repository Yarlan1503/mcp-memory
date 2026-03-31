"""Limbic scoring engine for MCP Memory — Opción B (Producto).

Reranks KNN candidates using importance, temporal decay, and co-occurrence signals.
Transparent to the external API — same output format as plain KNN."""

import math
from datetime import datetime, timezone
from typing import Any

# ------------------------------------------------------------------
# Tuneable constants
# ------------------------------------------------------------------

BETA_SAL = 0.5  # Salience boost factor
BETA_DEG = 0.15  # Degree weight
D_MAX = 15  # Degree cap
LAMBDA_HOURLY = 0.0001  # Decay rate per hour (half-life ~290 days)
COOC_TEMPORAL_FLOOR = (
    0.1  # Floor for co-occurrence decay (knowledge degrades but isn't destroyed)
)
GAMMA = (
    0.01  # Co-occurrence weight per pair (reduced from 0.1 — 0.1 dominated scoring 24x)
)
EXPANSION_FACTOR = 3  # KNN over-retrieval factor
TEMPORAL_FLOOR = 0.1  # Temporal decay never drops below this (knowledge degrades but isn't destroyed)
RRF_K = 60  # RRF constant (default from original paper)


# ------------------------------------------------------------------
# Component functions
# ------------------------------------------------------------------


def _parse_sqlite_datetime(dt_str: str) -> datetime:
    """Parse SQLite datetime string ('YYYY-MM-DD HH:MM:SS') to UTC datetime."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def compute_cooc_decay(last_co_str: str) -> float:
    """Compute temporal decay factor for co-occurrences.

    cooc_decay = max(COOC_TEMPORAL_FLOOR, exp(-LAMBDA_HOURLY × Δt_hours))

    Uses the same half-life (~290 days) as compute_temporal_factor.
    """
    try:
        dt = _parse_sqlite_datetime(last_co_str)
    except (ValueError, TypeError):
        return 1.0  # Can't parse → no decay (neutral)

    now = datetime.now(timezone.utc)
    delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    return max(COOC_TEMPORAL_FLOOR, math.exp(-LAMBDA_HOURLY * delta_hours))


def compute_importance(
    access_count: int,
    max_access: int,
    degree: int,
) -> float:
    """Compute structural importance (SIN temporal).

    importance(e) = [log2(1 + access_count) / log2(1 + max_access)]
                  × (1 + β_deg × min(degree, D_max) / D_max)

    When max_access == 0, returns 0.0 (no access data → neutral).
    """
    if max_access <= 0:
        return 0.0

    access_norm = math.log2(1 + access_count) / math.log2(1 + max_access)
    degree_norm = min(degree, D_MAX) / D_MAX
    return access_norm * (1 + BETA_DEG * degree_norm)


def compute_temporal_factor(last_access_str: str, created_at_str: str) -> float:
    """Compute temporal decay factor.

    temporal_factor(e) = max(TEMPORAL_FLOOR, exp(-LAMBDA_HOURLY × Δt_hours))

    Uses last_access if available, otherwise created_at.
    """
    try:
        dt = _parse_sqlite_datetime(last_access_str)
    except (ValueError, TypeError):
        try:
            dt = _parse_sqlite_datetime(created_at_str)
        except (ValueError, TypeError):
            return 1.0  # Can't parse → neutral

    now = datetime.now(timezone.utc)
    delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    return max(TEMPORAL_FLOOR, math.exp(-LAMBDA_HOURLY * delta_hours))


def compute_cooc_boost(
    entity_id: int,
    result_ids: list[int],
    cooc_map: dict[tuple[int, int], dict[str, Any]],
) -> float:
    """Compute co-occurrence boost with temporal decay.

    cooc_boost(e, R) = Σ_{r ∈ R, r ≠ e} log2(1 + co_count(e, r)) × decay(last_co(e, r))

    Decay uses the same half-life (~290 days) as compute_temporal_factor.
    """
    total = 0.0
    for other_id in result_ids:
        if other_id == entity_id:
            continue
        a, b = min(entity_id, other_id), max(entity_id, other_id)
        cooc_info = cooc_map.get((a, b))
        if cooc_info:
            co_count = cooc_info.get("co_count", 0)
            last_co_str = cooc_info.get("last_co", "")
            if co_count > 0:
                decay = compute_cooc_decay(last_co_str)
                total += math.log2(1 + co_count) * decay
    return total


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    fts_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge results from semantic (KNN) and FTS5 search using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    where rank_i(d) is the 1-based position of document d in ranking i.

    If a document appears in only one ranking, it only gets a score from that system.
    Documents present in both rankings get a boost.

    Args:
        semantic_results: [{"entity_id": int, "distance": float}] ordered by relevance
            (distance ascending = more similar first, as returned by sqlite-vec KNN).
        fts_results: [{"entity_id": int, "rank": float}] ordered by relevance
            (rank descending = more relevant first, as returned by search_fts).
        k: RRF smoothing constant (default 60). Higher = less weight on rank position.

    Returns:
        List of dicts sorted by rrf_score descending.
        Each dict: {"entity_id": int, "rrf_score": float, "distance": float | None}
        distance is populated for entities found in semantic_results, None otherwise.
    """
    rrf_scores: dict[int, float] = {}
    distances: dict[int, float] = {}

    # Semantic ranking: 1-based position by distance (ascending)
    for rank_pos, item in enumerate(semantic_results, start=1):
        eid = item["entity_id"]
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank_pos)
        distances[eid] = item["distance"]

    # FTS5 ranking: 1-based position by rank (descending = already ordered)
    for rank_pos, item in enumerate(fts_results, start=1):
        eid = item["entity_id"]
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank_pos)

    # Build sorted results
    merged = []
    for eid, score in rrf_scores.items():
        merged.append(
            {
                "entity_id": eid,
                "rrf_score": score,
                "distance": distances.get(eid),
            }
        )

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def rank_candidates(
    knn_results: list[dict],
    access_data: dict[int, dict],
    degree_data: dict[int, int],
    cooc_data: dict[tuple[int, int], dict[str, Any]],
    entity_created: dict[int, str],
    limit: int,
) -> list[dict]:
    """Re-rank KNN candidates using Opción B scoring.

    score(e, q) = cosine_sim(q, e)
                × (1 + β_sal × importance(e))
                × temporal_factor(e)
                × (1 + γ · cooc_boost(e, R))

    Args:
        knn_results: [{"entity_id": int, "distance": float}] from KNN.
        access_data: {entity_id: {"access_count": int, "last_access": str}}.
        degree_data: {entity_id: degree_count}.
        cooc_data: {(id_a, id_b): {"co_count": int, "last_co": str}}.
        entity_created: {entity_id: created_at_str}.
        limit: Number of results to return (top-K).

    Returns:
        Top-K candidates sorted by limbic score (descending).
        Each item includes "entity_id", "distance", and "limbic_score".
    """
    if not knn_results:
        return []

    # Compute max_access for normalization
    max_access = max(
        (
            access_data.get(r["entity_id"], {}).get("access_count", 0)
            for r in knn_results
        ),
        default=0,
    )

    # Collect entity IDs for co-occurrence boost
    all_ids = [r["entity_id"] for r in knn_results]

    scored = []
    for item in knn_results:
        eid = item["entity_id"]
        distance = item["distance"]

        # 1. Cosine similarity (distance from sqlite-vec cosine)
        cosine_sim = max(0.0, 1.0 - distance)

        # 2. Importance
        ad = access_data.get(eid, {})
        importance = compute_importance(
            access_count=ad.get("access_count", 0),
            max_access=max_access,
            degree=degree_data.get(eid, 0),
        )

        # 3. Temporal factor
        temporal = compute_temporal_factor(
            last_access_str=ad.get("last_access", ""),
            created_at_str=entity_created.get(eid, ""),
        )

        # 4. Co-occurrence boost
        cooc = compute_cooc_boost(eid, all_ids, cooc_data)

        # Opción B: product of all factors
        limbic_score = (
            cosine_sim * (1 + BETA_SAL * importance) * temporal * (1 + GAMMA * cooc)
        )

        scored.append(
            {
                "entity_id": eid,
                "distance": distance,
                "limbic_score": limbic_score,
                "importance": importance,
                "temporal_factor": temporal,
                "cooc_boost": cooc,
            }
        )

    # Sort by limbic_score descending
    scored.sort(key=lambda x: x["limbic_score"], reverse=True)

    return scored[:limit]


def rank_hybrid_candidates(
    merged_results: list[dict],
    access_data: dict[int, dict],
    degree_data: dict[int, int],
    cooc_data: dict[tuple[int, int], dict[str, Any]],
    entity_created: dict[int, str],
    limit: int,
) -> list[dict]:
    """Re-rank RRF-merged candidates using limbic scoring.

    Like rank_candidates, but uses rrf_score (normalized) as the base relevance
    instead of cosine similarity from KNN distance.

    For entities with KNN distance available:
        base_relevance = cosine_sim = max(0.0, 1.0 - distance)
    For FTS-only entities (distance is None):
        base_relevance = min-max normalized rrf_score across the batch

    score(e) = base_relevance(e)
              × (1 + β_sal × importance(e))
              × temporal_factor(e)
              × (1 + γ · cooc_boost(e, R))

    Args:
        merged_results: [{"entity_id": int, "rrf_score": float, "distance": float | None}]
            from reciprocal_rank_fusion(), ordered by rrf_score.
        access_data: {entity_id: {"access_count": int, "last_access": str}}.
        degree_data: {entity_id: degree_count}.
        cooc_data: {(id_a, id_b): {"co_count": int, "last_co": str}}.
        entity_created: {entity_id: created_at_str}.
        limit: Number of results to return (top-K).

    Returns:
        Top-K candidates sorted by limbic_score (descending).
        Each item: {"entity_id", "distance" (or None), "rrf_score", "limbic_score",
                     "importance", "temporal_factor", "cooc_boost"}.
    """
    if not merged_results:
        return []

    # Normalize RRF scores to [0, 1] for FTS-only entities
    rrf_values = [r["rrf_score"] for r in merged_results]
    rrf_min = min(rrf_values) if rrf_values else 0.0
    rrf_max = max(rrf_values) if rrf_values else 1.0
    rrf_range = rrf_max - rrf_min if rrf_max != rrf_min else 1.0

    # Compute max_access for importance normalization
    max_access = max(
        (
            access_data.get(r["entity_id"], {}).get("access_count", 0)
            for r in merged_results
        ),
        default=0,
    )

    all_ids = [r["entity_id"] for r in merged_results]

    scored = []
    for item in merged_results:
        eid = item["entity_id"]
        distance = item["distance"]
        rrf_score = item["rrf_score"]

        # 1. Base relevance
        if distance is not None:
            # Entity found by KNN — use cosine similarity
            base_relevance = max(0.0, 1.0 - distance)
        else:
            # FTS-only entity — use normalized RRF score
            # Map to [0.2, 0.8] range to avoid extremes
            norm_rrf = (rrf_score - rrf_min) / rrf_range
            base_relevance = 0.2 + 0.6 * norm_rrf

        # 2. Importance
        ad = access_data.get(eid, {})
        importance = compute_importance(
            access_count=ad.get("access_count", 0),
            max_access=max_access,
            degree=degree_data.get(eid, 0),
        )

        # 3. Temporal factor
        temporal = compute_temporal_factor(
            last_access_str=ad.get("last_access", ""),
            created_at_str=entity_created.get(eid, ""),
        )

        # 4. Co-occurrence boost
        cooc = compute_cooc_boost(eid, all_ids, cooc_data)

        # Product of all factors (same formula as rank_candidates)
        limbic_score = (
            base_relevance * (1 + BETA_SAL * importance) * temporal * (1 + GAMMA * cooc)
        )

        scored.append(
            {
                "entity_id": eid,
                "distance": distance,
                "rrf_score": rrf_score,
                "limbic_score": limbic_score,
                "importance": importance,
                "temporal_factor": temporal,
                "cooc_boost": cooc,
            }
        )

    scored.sort(key=lambda x: x["limbic_score"], reverse=True)
    return scored[:limit]
