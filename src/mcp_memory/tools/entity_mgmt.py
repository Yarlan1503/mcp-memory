"""Entity management tools for MCP Memory knowledge graph."""

import logging
from typing import Any

from mcp_memory.backpressure import bounded_heavy_tool
from mcp_memory.entity_splitter import (
    analyze_entity_for_split,
    propose_entity_split,
    execute_entity_split,
    find_all_split_candidates,
    _get_threshold,
)
from mcp_memory._helpers import _get_engine, tool_error_handler

logger = logging.getLogger(__name__)


def _preview_content(content: str, max_len: int = 80) -> str:
    """Truncate content for preview display."""
    if len(content) <= max_len:
        return content
    return content[:max_len] + "..."


# ============================================================
# TOOL 12: analyze_entity_split
# ============================================================
@tool_error_handler
@bounded_heavy_tool
def analyze_entity_split(entity_name: str) -> dict[str, Any]:
    """Analyze an entity and determine if it needs to be split.

    An entity is a candidate for splitting when it exceeds the observation
    threshold for its type (Sesion=15, Proyecto=25, otras=20) and has
    sufficient topic diversity.

    Returns a dict with analysis results including entity metadata,
    observation count, threshold, detected topics, and split score."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store
    analysis = analyze_entity_for_split(store, entity_name)
    if analysis is None:
        return {"error": f"Entity not found: {entity_name}"}
    return {"analysis": analysis}


# ============================================================
# TOOL 13: propose_entity_split
# ============================================================
@tool_error_handler
@bounded_heavy_tool
def propose_entity_split_tool(entity_name: str) -> dict[str, Any]:
    """Analyze and propose a split for an entity if needed.

    Returns a split proposal with suggested new entity names,
    topic groupings, and relations to create.
    Returns None if the entity doesn't need splitting."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store
    proposal = propose_entity_split(store, entity_name)
    if proposal is None:
        return {"proposal": None, "message": "Entity does not need splitting"}
    return {"proposal": proposal}


# ============================================================
# TOOL 14: execute_entity_split
# ============================================================
@tool_error_handler
@bounded_heavy_tool
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
    import mcp_memory.server as _server_mod

    store = _server_mod.store
    result = execute_entity_split(
        store, entity_name, approved_splits, parent_entity_name
    )
    return {"result": result}


# ============================================================
# TOOL 15: find_split_candidates
# ============================================================
@tool_error_handler
@bounded_heavy_tool
def find_split_candidates() -> dict[str, Any]:
    """Find all entities in the knowledge graph that are candidates for splitting.

    Scans all entities and returns those that exceed their type-specific
    observation threshold and have sufficient topic diversity."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store
    candidates = find_all_split_candidates(store)
    return {"candidates": candidates}


# ============================================================
# TOOL 16: find_duplicate_observations
# ============================================================
@tool_error_handler
@bounded_heavy_tool
def find_duplicate_observations(
    entity_name: str, threshold: float = 0.85, containment_threshold: float = 0.7
) -> dict[str, Any]:
    """Find observations that may be semantically duplicated within an entity.
    Returns pairs of observations with similarity score above threshold.
    Use this to review and consolidate redundant observations.
    Uses combined similarity: cosine >= threshold OR containment >= containment_threshold
    when texts have asymmetric length (one is 2x+ longer than the other)."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store
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


# ============================================================
# TOOL 17: consolidation_report
# ============================================================
@tool_error_handler
@bounded_heavy_tool
def consolidation_report(stale_days: float = 90.0) -> dict[str, Any]:
    """Generate a memory consolidation report without making any changes.

    Analyzes the knowledge graph for:
    - Split candidates: entities exceeding observation thresholds with sufficient topic diversity
    - Flagged observations: observations marked as potential duplicates (similarity_flag=1)
    - Stale entities: entities not accessed in N days with low access count
    - Large entities: entities exceeding size thresholds (may need splitting)

    Use this report to decide what consolidation actions to take.
    The report is read-only — no changes are made to the knowledge graph."""
    import mcp_memory.server as _server_mod

    store = _server_mod.store
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
        days = store.days_since_access(s["last_access"])
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
