"""Limbic scoring engine for MCP Memory — Opción B (Producto).

Reranks KNN candidates using importance, temporal decay, and co-occurrence signals.
Transparent to the external API — same output format as plain KNN."""

import math
from datetime import datetime, timezone

# ------------------------------------------------------------------
# Tuneable constants
# ------------------------------------------------------------------

BETA_SAL = 0.5  # Salience boost factor
BETA_DEG = 0.15  # Degree weight
D_MAX = 15  # Degree cap
LAMBDA_HOURLY = 0.001  # Decay rate per hour (half-life ~29 days)
GAMMA = 0.1  # Co-occurrence weight per pair
EXPANSION_FACTOR = 3  # KNN over-retrieval factor


# ------------------------------------------------------------------
# Component functions
# ------------------------------------------------------------------


def _parse_sqlite_datetime(dt_str: str) -> datetime:
    """Parse SQLite datetime string ('YYYY-MM-DD HH:MM:SS') to UTC datetime."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


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

    temporal_factor(e) = exp(-λ × Δt_hours)

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
    return math.exp(-LAMBDA_HOURLY * delta_hours)


def compute_cooc_boost(
    entity_id: int,
    result_ids: list[int],
    cooc_map: dict[tuple[int, int], int],
) -> float:
    """Compute co-occurrence boost.

    cooc_boost(e, R) = Σ_{r ∈ R, r ≠ e} log2(1 + co_count(e, r))
    """
    total = 0.0
    for other_id in result_ids:
        if other_id == entity_id:
            continue
        a, b = min(entity_id, other_id), max(entity_id, other_id)
        co_count = cooc_map.get((a, b), 0)
        if co_count > 0:
            total += math.log2(1 + co_count)
    return total


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def rank_candidates(
    knn_results: list[dict],
    access_data: dict[int, dict],
    degree_data: dict[int, int],
    cooc_data: dict[tuple[int, int], int],
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
        cooc_data: {(id_a, id_b): co_count}.
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
            }
        )

    # Sort by limbic_score descending
    scored.sort(key=lambda x: x["limbic_score"], reverse=True)

    return scored[:limit]
