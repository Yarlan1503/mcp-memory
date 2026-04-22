"""Limbic scoring engine for MCP Memory — Opción B (Producto).

Reranks KNN candidates using importance, temporal decay, and co-occurrence signals.
Transparent to the external API — same output format as plain KNN."""

import logging
import math
import string
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Estrategias de routing para search_semantic."""

    COSINE_HEAVY = "cosine_heavy"  # 70% cosine, 30% limbic — queries factuales
    LIMBIC_HEAVY = "limbic_heavy"  # 30% cosine, 70% limbic — queries exploratorias
    HYBRID_BALANCED = (
        "hybrid_balanced"  # 50% cosine, 50% limbic — queries de contexto amplio
    )


# ------------------------------------------------------------------
# Tuneable constants
# ------------------------------------------------------------------

BETA_SAL = 0.5  # Salience boost factor
BETA_DEG = 0.15  # Degree weight
ALPHA_CONS = 0.2  # Consolidation weight (multi-day access signal)
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

# Containment dedup constants
CONTAINMENT_THRESHOLD = 0.7  # % of tokens in shorter that must appear in longer
LENGTH_RATIO_THRESHOLD = (
    2.0  # One string must be 2x+ longer than other to activate containment
)


# ------------------------------------------------------------------
# Containment dedup helpers
# ------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase words, stripping punctuation."""
    return {
        word.strip(string.punctuation)
        for word in text.lower().split()
        if word and word.strip(string.punctuation)
    }


def compute_containment(shorter: str, longer: str) -> float:
    """Compute token containment ratio: % of tokens in shorter that appear in longer.

    containment(s, l) = |tokens(s) ∩ tokens(l)| / |tokens(s)|

    Returns 0.0 to 1.0. Returns 1.0 if shorter is empty or all tokens match.
    """
    if not shorter.strip():
        return 1.0

    shorter_tokens = _tokenize(shorter)
    longer_tokens = _tokenize(longer)

    if not shorter_tokens:
        return 1.0

    overlap = shorter_tokens & longer_tokens
    return len(overlap) / len(shorter_tokens)


def combined_similarity(
    cosine_sim: float,
    text_a: str,
    text_b: str,
    cosine_threshold: float = 0.85,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
    length_ratio_threshold: float = LENGTH_RATIO_THRESHOLD,
) -> bool:
    """Determine if two texts are duplicates using cosine + containment.

    Returns True if:
    - cosine_sim >= cosine_threshold (existing behavior), OR
    - containment_ratio >= containment_threshold AND length ratio is asymmetric (>= length_ratio_threshold)

    The containment signal only activates when one text is significantly longer
    than the other, avoiding false positives between texts of similar length.
    """
    # Path 1: existing cosine behavior
    if cosine_sim >= cosine_threshold:
        return True

    # Path 2: containment for asymmetric length pairs
    len_a = len(text_a)
    len_b = len(text_b)
    min_len = max(min(len_a, len_b), 1)
    max_len = max(len_a, len_b)
    length_ratio = max_len / min_len

    if length_ratio >= length_ratio_threshold:
        shorter, longer = (text_a, text_b) if len_a <= len_b else (text_b, text_a)
        containment = compute_containment(shorter, longer)
        if containment >= containment_threshold:
            return True

    return False


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
    access_days: int = 1,
    max_access_days: int = 1,
) -> float:
    """Compute structural importance with consolidation signal.

    importance(e) = [log2(1 + access_count) / log2(1 + max_access)]
                  × (1 + β_deg × min(degree, D_max) / D_max)
                  × (1 + α_cons × consolidation(e))

    where consolidation(e) = log2(1 + access_days) / log2(1 + max_access_days).

    When max_access_days <= 1, consolidation = 0 and the multiplier is 1.0
    (identical to pre-consolidation scoring — retrocompatible).

    When max_access == 0, returns 0.0 (no access data → neutral).
    """
    if max_access <= 0:
        return 0.0

    access_norm = math.log2(1 + access_count) / math.log2(1 + max_access)
    degree_norm = min(degree, D_MAX) / D_MAX

    # Consolidation factor: neutral when max_access_days <= 1
    if max_access_days > 1:
        consolidation = math.log2(1 + access_days) / math.log2(1 + max_access_days)
        cons_multiplier = 1 + ALPHA_CONS * consolidation
    else:
        cons_multiplier = 1.0

    return access_norm * (1 + BETA_DEG * degree_norm) * cons_multiplier


def _compute_recency_factor(created_at_str: str) -> float:
    """Compute recency factor from a SQLite datetime string.

    recency = max(TEMPORAL_FLOOR, exp(-LAMBDA_HOURLY × Δt_hours))

    Returns 1.0 if the string cannot be parsed.
    """
    try:
        dt = _parse_sqlite_datetime(created_at_str)
    except (ValueError, TypeError):
        return 1.0

    now = datetime.now(timezone.utc)
    delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    return max(TEMPORAL_FLOOR, math.exp(-LAMBDA_HOURLY * delta_hours))


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
    gamma: float | None = None,
    beta_sal: float | None = None,
    access_days_data: dict[int, int] | None = None,
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
        access_days_data: {entity_id: access_days_count}. Optional consolidation signal.
            When empty/None, scoring is identical to pre-consolidation behavior.

    Returns:
        Top-K candidates sorted by limbic score (descending).
        Each item includes "entity_id", "distance", and "limbic_score".
    """
    if not knn_results:
        return []

    _access_days_data = access_days_data or {}
    _max_access_days = max(_access_days_data.values(), default=1)

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

        # 2. Importance (with consolidation signal)
        ad = access_data.get(eid, {})
        importance = compute_importance(
            access_count=ad.get("access_count", 0),
            max_access=max_access,
            degree=degree_data.get(eid, 0),
            access_days=_access_days_data.get(eid, 1),
            max_access_days=_max_access_days,
        )

        # 3. Temporal factor
        temporal = compute_temporal_factor(
            last_access_str=ad.get("last_access", ""),
            created_at_str=entity_created.get(eid, ""),
        )

        # 4. Co-occurrence boost
        cooc = compute_cooc_boost(eid, all_ids, cooc_data)

        # Opción B: product of all factors
        # Use passed values or fall back to module constants
        _gamma = gamma if gamma is not None else GAMMA
        _beta_sal = beta_sal if beta_sal is not None else BETA_SAL
        limbic_score = (
            cosine_sim * (1 + _beta_sal * importance) * temporal * (1 + _gamma * cooc)
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
    gamma: float | None = None,
    beta_sal: float | None = None,
    access_days_data: dict[int, int] | None = None,
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
        access_days_data: {entity_id: access_days_count}. Optional consolidation signal.

    Returns:
        Top-K candidates sorted by limbic_score (descending).
        Each item: {"entity_id", "distance" (or None), "rrf_score", "limbic_score",
                     "importance", "temporal_factor", "cooc_boost"}.
    """
    if not merged_results:
        return []

    _access_days_data = access_days_data or {}
    _max_access_days = max(_access_days_data.values(), default=1)

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

        # 2. Importance (with consolidation signal)
        ad = access_data.get(eid, {})
        importance = compute_importance(
            access_count=ad.get("access_count", 0),
            max_access=max_access,
            degree=degree_data.get(eid, 0),
            access_days=_access_days_data.get(eid, 1),
            max_access_days=_max_access_days,
        )

        # 3. Temporal factor
        temporal = compute_temporal_factor(
            last_access_str=ad.get("last_access", ""),
            created_at_str=entity_created.get(eid, ""),
        )

        # 4. Co-occurrence boost
        cooc = compute_cooc_boost(eid, all_ids, cooc_data)

        # Product of all factors (same formula as rank_candidates)
        # Use passed values or fall back to module constants
        _gamma = gamma if gamma is not None else GAMMA
        _beta_sal = beta_sal if beta_sal is not None else BETA_SAL
        limbic_score = (
            base_relevance
            * (1 + _beta_sal * importance)
            * temporal
            * (1 + _gamma * cooc)
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


def detect_query_type(query_text: str, k_limit: int) -> RoutingStrategy:
    """Detect query type based on linguistic features and k_limit.

    Scoring rules:
        - Factual keywords ("qué es", "definición", "cómo funciona", "es un/una") → +2
        - Intermediate keywords ("relación", "diferencia", "ejemplos", "comparar") → +1
        - Exploratory keywords ("explícame", "relación entre", "dime todo", "cuentame", "qué piensas") → -2
        - Query length ≤3 words → +1 (factual)
        - Query length ≥10 words → -1 (exploratory)
        - K limit ≤3 → +1 (precise/factual)
        - K limit ≥10 → -1 (exploratory)

    Final routing:
        - Score ≥ 2 → COSINE_HEAVY
        - Score ≤ -2 → LIMBIC_HEAVY
        - Otherwise → HYBRID_BALANCED

    Args:
        query_text: The search query string.
        k_limit: The k limit parameter for the search.

    Returns:
        RoutingStrategy enum value.
    """
    query_lower = query_text.lower()
    word_count = len(query_text.split())

    score = 0

    # Factual keywords (+2 each)
    factual_keywords = ["qué es", "definición", "cómo funciona", "es un ", "es una "]
    for kw in factual_keywords:
        if kw in query_lower:
            score += 2

    # Intermediate keywords (+1 each)
    intermediate_keywords = ["relación", "diferencia", "ejemplos", "comparar"]
    for kw in intermediate_keywords:
        if kw in query_lower:
            score += 1

    # Exploratory keywords (-2 each)
    exploratory_keywords = [
        "explícame",
        "relación entre",
        "dime todo",
        "cuentame",
        "qué piensas",
    ]
    for kw in exploratory_keywords:
        if kw in query_lower:
            score -= 2

    # Query length scoring
    if word_count <= 3:
        score += 1  # Short query → factual
    elif word_count >= 10:
        score -= 1  # Long query → exploratory

    # K limit scoring
    if k_limit <= 3:
        score += 1  # Small k → precise/factual
    elif k_limit >= 10:
        score -= 1  # Large k → exploratory

    # Determine routing strategy
    if score >= 2:
        strategy = RoutingStrategy.COSINE_HEAVY
    elif score <= -2:
        strategy = RoutingStrategy.LIMBIC_HEAVY
    else:
        strategy = RoutingStrategy.HYBRID_BALANCED

    logger.debug(f"Query '{query_text}' scored {score} → {strategy.value}")

    return strategy


def rank_with_routing_strategy(
    merged_results: list[dict],
    access_data: dict[int, dict],
    degree_data: dict[int, int],
    cooc_data: dict[tuple[int, int], dict[str, Any]],
    entity_created: dict[int, str],
    limit: int,
    strategy: RoutingStrategy,
    gamma: float | None = None,
    beta_sal: float | None = None,
    access_days_data: dict[int, int] | None = None,
) -> list[dict]:
    """Re-rank merged candidates using a specified routing strategy.

    First calls rank_hybrid_candidates() to compute limbic_score for each entity,
    then blends cosine similarity with normalized limbic_score according to strategy.

    Blending formulas:
        - COSINE_HEAVY: 0.7 * cosine + 0.3 * limbic_norm
        - LIMBIC_HEAVY: 0.3 * cosine + 0.7 * limbic_norm
        - HYBRID_BALANCED: 0.5 * cosine + 0.5 * limbic_norm

    Args:
        merged_results: [{"entity_id": int, "rrf_score": float, "distance": float | None}]
            from reciprocal_rank_fusion().
        access_data: {entity_id: {"access_count": int, "last_access": str}}.
        degree_data: {entity_id: degree_count}.
        cooc_data: {(id_a, id_b): {"co_count": int, "last_co": str}}.
        entity_created: {entity_id: created_at_str}.
        limit: Number of results to return (top-K).
        strategy: RoutingStrategy enum value.
        access_days_data: {entity_id: access_days_count}. Optional consolidation signal.

    Returns:
        Top-K candidates sorted by final_score (descending).
        Each item includes original fields plus "final_score" and "routing_strategy".
    """
    if not merged_results:
        return []

    # Step 1: Get limbic scores via rank_hybrid_candidates
    limbic_results = rank_hybrid_candidates(
        merged_results=merged_results,
        access_data=access_data,
        degree_data=degree_data,
        cooc_data=cooc_data,
        entity_created=entity_created,
        limit=limit,
        gamma=gamma,
        beta_sal=beta_sal,
        access_days_data=access_days_data,
    )

    # Step 2: Collect cosine and limbic values for normalization
    cosine_values = []
    limbic_values = []

    for item in limbic_results:
        distance = item.get("distance")
        if distance is not None:
            cosine = max(0.0, 1.0 - distance)
        else:
            cosine = 0.0
        cosine_values.append(cosine)
        limbic_values.append(item["limbic_score"])

    # Min-max normalization for limbic_score
    limbic_min = min(limbic_values) if limbic_values else 0.0
    limbic_max = max(limbic_values) if limbic_values else 1.0
    limbic_range = limbic_max - limbic_min if limbic_max != limbic_min else 1.0

    # Step 3: Apply blending based on strategy
    if strategy == RoutingStrategy.COSINE_HEAVY:
        cosine_weight, limbic_weight = 0.7, 0.3
    elif strategy == RoutingStrategy.LIMBIC_HEAVY:
        cosine_weight, limbic_weight = 0.3, 0.7
    else:  # HYBRID_BALANCED
        cosine_weight, limbic_weight = 0.5, 0.5

    scored = []
    for item in limbic_results:
        distance = item.get("distance")
        cosine = max(0.0, 1.0 - distance) if distance is not None else 0.0
        limbic_norm = (item["limbic_score"] - limbic_min) / limbic_range

        final_score = cosine_weight * cosine + limbic_weight * limbic_norm

        result = dict(item)
        result["final_score"] = final_score
        result["routing_strategy"] = strategy.value
        scored.append(result)

    # Sort by final_score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    return scored[:limit]
