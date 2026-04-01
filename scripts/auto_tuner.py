"""Auto-tuner for MCP Memory limbic scoring hyperparameters.

Analyzes collected search data to find optimal GAMMA × BETA_SAL combinations
and applies them smoothly to avoid sudden degradations.

Usage:
    python scripts/auto_tuner.py --analyze
    python scripts/auto_tuner.py --tune
    python scripts/auto_tuner.py --set-gamma 0.05 --set-beta 0.75
"""

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default parameter values (backwards compatibility)
DEFAULT_GAMMA = 0.01
DEFAULT_BETA_SAL = 0.5
DEFAULT_MIN_EVENTS = 50


# ------------------------------------------------------------------
# Core tuning functions
# ------------------------------------------------------------------


def recompute_score(
    cosine_sim: float,
    importance: float,
    temporal: float,
    cooc_boost: float,
    gamma: float,
    beta_sal: float,
) -> float:
    """Recompute limbic score with given parameters.

    Formula:
        limbic_score = cosine_sim * (1 + beta_sal * importance) * temporal * (1 + gamma * cooc_boost)
    """
    return (
        cosine_sim * (1 + beta_sal * importance) * temporal * (1 + gamma * cooc_boost)
    )


def get_current_params(conn: sqlite3.Connection) -> dict[str, float]:
    """Get current GAMMA and BETA_SAL from db_metadata.

    Returns defaults if not set.
    """
    gamma = _get_metadata_float(conn, "gamma", DEFAULT_GAMMA)
    beta_sal = _get_metadata_float(conn, "beta_sal", DEFAULT_BETA_SAL)
    return {"gamma": gamma, "beta_sal": beta_sal}


def _get_metadata_float(conn: sqlite3.Connection, key: str, default: float) -> float:
    """Get a float value from db_metadata, returning default if not found."""
    row = conn.execute("SELECT value FROM db_metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return float(row["value"])
    except (ValueError, TypeError):
        return default


def analyze_current_performance(conn: sqlite3.Connection) -> dict[str, Any]:
    """Calculate aggregated metrics (NDCG@K, Lift@K) with CURRENT parameters.

    Uses treatment=1 events only (A/B test treatment group).

    Returns:
        {
            "ndcg@1": float,
            "ndcg@5": float,
            "lift@5": float,
            "num_events": int,
        }
    """
    cursor = conn.execute(
        """
        SELECT sr.event_id, sr.entity_id, sr.rank, sr.limbic_score,
               sr.cosine_sim, sr.importance, sr.temporal, sr.cooc_boost,
               if.implicit_feedback
        FROM search_results sr
        JOIN search_events se ON se.event_id = sr.event_id
        LEFT JOIN (
            SELECT event_id, entity_id, MAX(re_accessed) as implicit_feedback
            FROM implicit_feedback
            GROUP BY event_id, entity_id
        ) if ON if.event_id = sr.event_id AND if.entity_id = sr.entity_id
        WHERE se.treatment = 1
        ORDER BY sr.event_id, sr.limbic_score DESC
        """
    )

    # Group by event
    events: dict[int, list[dict]] = {}
    for row in cursor.fetchall():
        event_id = row[0]
        if event_id not in events:
            events[event_id] = []
        events[event_id].append(
            {
                "entity_id": row[1],
                "rank": row[2],
                "limbic_score": row[3],
                "cosine_sim": row[4],
                "importance": row[5],
                "temporal": row[6],
                "cooc_boost": row[7],
                "implicit_feedback": row[8] or 0,
            }
        )

    if not events:
        return {"ndcg@1": 0.0, "ndcg@5": 0.0, "lift@5": 0.0, "num_events": 0}

    # Calculate NDCG@K for each event
    ndcg_at_1_scores = []
    ndcg_at_5_scores = []
    lift_at_5_scores = []

    for event_id, items in events.items():
        # Sort by limbic_score (already should be, but ensure)
        items_sorted = sorted(items, key=lambda x: x["limbic_score"], reverse=True)

        # Relevance judgments: implicit_feedback > 0 means relevant
        relevance = [item["implicit_feedback"] for item in items_sorted]

        # NDCG@1
        ndcg_1 = _ndcg_at_k(relevance, k=1)
        ndcg_at_1_scores.append(ndcg_1)

        # NDCG@5
        ndcg_5 = _ndcg_at_k(relevance, k=5)
        ndcg_at_5_scores.append(ndcg_5)

        # Lift@5: proportion of relevant items in top-5 vs overall
        top5_relevant = sum(relevance[:5])
        total_relevant = sum(relevance)
        total_items = len(relevance)
        if total_relevant > 0 and total_items > 0:
            top5_proportion = top5_relevant / min(5, total_items)
            overall_proportion = total_relevant / total_items
            lift = (
                top5_proportion / overall_proportion if overall_proportion > 0 else 0.0
            )
        else:
            lift = 0.0
        lift_at_5_scores.append(lift)

    return {
        "ndcg@1": sum(ndcg_at_1_scores) / len(ndcg_at_1_scores)
        if ndcg_at_1_scores
        else 0.0,
        "ndcg@5": sum(ndcg_at_5_scores) / len(ndcg_at_5_scores)
        if ndcg_at_5_scores
        else 0.0,
        "lift@5": sum(lift_at_5_scores) / len(lift_at_5_scores)
        if lift_at_5_scores
        else 0.0,
        "num_events": len(events),
    }


def _ndcg_at_k(relevance: list[int], k: int) -> float:
    """Calculate NDCG@K for a relevance list.

    Uses log2-based DCG: DCG@k = SUM_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        relevance: List of relevance judgments (0 or 1 typically)
        k: Cutoff position

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    k = min(k, len(relevance))
    if k == 0:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i in range(k):
        rel = relevance[i] if i < len(relevance) else 0
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG@k (sorted descending)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i in range(k):
        rel = ideal_relevance[i] if i < len(ideal_relevance) else 0
        idcg += (2**rel - 1) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def find_optimal_params(
    conn: sqlite3.Connection,
    gamma_range: list[float],
    beta_sal_range: list[float],
) -> dict[str, Any]:
    """Execute grid search to find optimal GAMMA × BETA_SAL combination.

    Uses treatment=1 events and recomputes limbic_score for each combination,
    then calculates NDCG@5 as the optimization metric.

    Returns:
        {
            "best_gamma": float,
            "best_beta_sal": float,
            "best_ndcg@5": float,
            "all_results": [...],
        }
    """
    # Get all events with their search results
    cursor = conn.execute(
        """
        SELECT sr.event_id, sr.entity_id, sr.rank, sr.limbic_score,
               sr.cosine_sim, sr.importance, sr.temporal, sr.cooc_boost,
               if.implicit_feedback
        FROM search_results sr
        JOIN search_events se ON se.event_id = sr.event_id
        LEFT JOIN (
            SELECT event_id, entity_id, MAX(re_accessed) as implicit_feedback
            FROM implicit_feedback
            GROUP BY event_id, entity_id
        ) if ON if.event_id = sr.event_id AND if.entity_id = sr.entity_id
        WHERE se.treatment = 1
        ORDER BY sr.event_id, sr.rank
        """
    )

    # Group by event
    events: dict[int, list[dict]] = {}
    for row in cursor.fetchall():
        event_id = row[0]
        if event_id not in events:
            events[event_id] = []
        events[event_id].append(
            {
                "entity_id": row[1],
                "rank": row[2],
                "limbic_score": row[3],
                "cosine_sim": row[4],
                "importance": row[5],
                "temporal": row[6],
                "cooc_boost": row[7],
                "implicit_feedback": row[8] or 0,
            }
        )

    if not events:
        return {
            "best_gamma": DEFAULT_GAMMA,
            "best_beta_sal": DEFAULT_BETA_SAL,
            "best_ndcg@5": 0.0,
            "all_results": [],
        }

    results = []

    for gamma in gamma_range:
        for beta_sal in beta_sal_range:
            ndcg_scores = []

            for event_id, items in events.items():
                # Recompute scores with new parameters
                scored = []
                for item in items:
                    new_score = recompute_score(
                        cosine_sim=item["cosine_sim"],
                        importance=item["importance"],
                        temporal=item["temporal"],
                        cooc_boost=item["cooc_boost"],
                        gamma=gamma,
                        beta_sal=beta_sal,
                    )
                    scored.append({**item, "new_score": new_score})

                # Sort by new score
                scored.sort(key=lambda x: x["new_score"], reverse=True)

                # Build relevance list for NDCG
                relevance = [item["implicit_feedback"] for item in scored]
                ndcg = _ndcg_at_k(relevance, k=5)
                ndcg_scores.append(ndcg)

            avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
            results.append(
                {
                    "gamma": gamma,
                    "beta_sal": beta_sal,
                    "ndcg@5": avg_ndcg,
                }
            )

    # Sort by NDCG descending
    results.sort(key=lambda x: x["ndcg@5"], reverse=True)

    best = (
        results[0]
        if results
        else {"gamma": DEFAULT_GAMMA, "beta_sal": DEFAULT_BETA_SAL, "ndcg@5": 0.0}
    )

    return {
        "best_gamma": best["gamma"],
        "best_beta_sal": best["beta_sal"],
        "best_ndcg@5": best["ndcg@5"],
        "all_results": results,
    }


def compute_quality_gain(
    conn: sqlite3.Connection,
    gamma: float,
    beta_sal: float,
) -> dict[str, Any]:
    """Compare quality of a new parameter combination vs current.

    Uses NDCG@5 as the quality metric, calculated only over events
    with treatment=1 (A/B test treatment group).

    Returns:
        {
            "gamma": float,
            "beta_sal": float,
            "current_ndcg@5": float,
            "new_ndcg@5": float,
            "gain": float,
        }
    """
    # Get current params
    current = get_current_params(conn)

    # Get current performance
    current_metrics = analyze_current_performance(conn)
    current_ndcg = current_metrics["ndcg@5"]

    # Get new performance
    cursor = conn.execute(
        """
        SELECT sr.event_id, sr.entity_id, sr.cosine_sim, sr.importance,
               sr.temporal, sr.cooc_boost, if.implicit_feedback
        FROM search_results sr
        JOIN search_events se ON se.event_id = sr.event_id
        LEFT JOIN (
            SELECT event_id, entity_id, MAX(re_accessed) as implicit_feedback
            FROM implicit_feedback
            GROUP BY event_id, entity_id
        ) if ON if.event_id = sr.event_id AND if.entity_id = sr.entity_id
        WHERE se.treatment = 1
        """
    )

    # Group by event
    events: dict[int, list[dict]] = {}
    for row in cursor.fetchall():
        event_id = row[0]
        if event_id not in events:
            events[event_id] = []
        events[event_id].append(
            {
                "entity_id": row[1],
                "cosine_sim": row[2],
                "importance": row[3],
                "temporal": row[4],
                "cooc_boost": row[5],
                "implicit_feedback": row[6] or 0,
            }
        )

    new_ndcg_scores = []
    for event_id, items in events.items():
        scored = []
        for item in items:
            new_score = recompute_score(
                cosine_sim=item["cosine_sim"],
                importance=item["importance"],
                temporal=item["temporal"],
                cooc_boost=item["cooc_boost"],
                gamma=gamma,
                beta_sal=beta_sal,
            )
            scored.append({**item, "new_score": new_score})

        scored.sort(key=lambda x: x["new_score"], reverse=True)
        relevance = [item["implicit_feedback"] for item in scored]
        new_ndcg_scores.append(_ndcg_at_k(relevance, k=5))

    new_ndcg = sum(new_ndcg_scores) / len(new_ndcg_scores) if new_ndcg_scores else 0.0

    return {
        "gamma": gamma,
        "beta_sal": beta_sal,
        "current_ndcg@5": current_ndcg,
        "new_ndcg@5": new_ndcg,
        "gain": new_ndcg - current_ndcg,
    }


def smooth_apply(
    conn: sqlite3.Connection,
    gamma: float,
    beta_sal: float,
    blend_factor: float = 0.1,
) -> dict[str, float]:
    """Apply new parameter values smoothly using exponential moving average.

    new_value = current * (1 - blend) + new * blend

    This avoids sudden changes that could cause unexpected degradation.

    Args:
        conn: SQLite connection
        gamma: New GAMMA value
        beta_sal: New BETA_SAL value
        blend_factor: Blend factor (0.0 to 1.0). Default 0.1 means
            10% of the new value is applied each time.

    Returns:
        {
            "gamma": float,  # Actual gamma written to DB
            "beta_sal": float,  # Actual beta_sal written to DB
        }
    """
    # Get current values
    current = get_current_params(conn)

    # Apply smooth blend
    new_gamma = current["gamma"] * (1 - blend_factor) + gamma * blend_factor
    new_beta_sal = current["beta_sal"] * (1 - blend_factor) + beta_sal * blend_factor

    # Write to db_metadata
    _set_metadata_float(conn, "gamma", new_gamma)
    _set_metadata_float(conn, "beta_sal", new_beta_sal)
    _set_metadata_str(conn, "last_tuned_at", datetime.now(timezone.utc).isoformat())

    return {"gamma": new_gamma, "beta_sal": new_beta_sal}


def _set_metadata_float(conn: sqlite3.Connection, key: str, value: float) -> None:
    """Set a float value in db_metadata."""
    conn.execute(
        "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
        (key, str(value)),
    )
    conn.commit()


def _set_metadata_str(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a string value in db_metadata."""
    conn.execute(
        "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def apply_to_scoring_file(gamma: float, beta_sal: float) -> None:
    """Apply GAMMA and BETA_SAL values directly to scoring.py module constants.

    This updates the hardcoded defaults in scoring.py so that the scoring
    functions use the tuned values when called without explicit parameters.

    Args:
        gamma: New GAMMA value to write.
        beta_sal: New BETA_SAL value to write.

    Note:
        This uses the edit tool to modify scoring.py in place.
        It preserves existing comments and formatting.
    """
    import re

    scoring_path = (
        Path(__file__).resolve().parents[1] / "src" / "mcp_memory" / "scoring.py"
    )

    if not scoring_path.exists():
        raise FileNotFoundError(f"scoring.py not found at {scoring_path}")

    content = scoring_path.read_text()

    # Update GAMMA value (preserve comment on same line or next line)
    # Match GAMMA = <number> with optional comment
    gamma_pattern = re.compile(r"^(GAMMA\s*=\s*)\d+(\.\d+)?", re.MULTILINE)
    if gamma_pattern.search(content):
        content = gamma_pattern.sub(rf"\g<1>{gamma}", content)
    else:
        # Fallback: find line containing "GAMMA = " and replace number
        gamma_fallback = re.compile(r"^(.*?GAMMA\s*=\s*)\d+(\.\d+)?(.*)$", re.MULTILINE)
        content = gamma_fallback.sub(rf"\g<1>{gamma}\g<3>", content)

    # Update BETA_SAL value
    beta_pattern = re.compile(r"^(BETA_SAL\s*=\s*)\d+(\.\d+)?", re.MULTILINE)
    if beta_pattern.search(content):
        content = beta_pattern.sub(rf"\g<1>{beta_sal}", content)
    else:
        # Fallback
        beta_fallback = re.compile(
            r"^(.*?BETA_SAL\s*=\s*)\d+(\.\d+)?(.*)$", re.MULTILINE
        )
        content = beta_fallback.sub(rf"\g<1>{beta_sal}\g<3>", content)

    scoring_path.write_text(content)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_range(s: str) -> list[float]:
    """Parse comma-separated float values."""
    return [float(x.strip()) for x in s.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-tuner for MCP Memory limbic scoring hyperparameters"
    )
    parser.add_argument(
        "--db",
        default="~/.config/opencode/mcp-memory/memory.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze current performance, don't apply changes",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Analyze and apply best parameters found",
    )
    parser.add_argument(
        "--set-gamma",
        type=float,
        metavar="X",
        help="Force specific gamma value (with smooth blend)",
    )
    parser.add_argument(
        "--set-beta",
        type=float,
        dest="set_beta_sal",
        metavar="Y",
        help="Force specific beta_sal value (with smooth blend)",
    )
    parser.add_argument(
        "--gamma-range",
        default="0.001,0.005,0.01,0.05,0.1",
        help="Comma-separated GAMMA values to explore",
    )
    parser.add_argument(
        "--beta-sal-range",
        default="0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated BETA_SAL values to explore",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=DEFAULT_MIN_EVENTS,
        help=f"Minimum events for tuning (default: {DEFAULT_MIN_EVENTS})",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.1,
        help="Blend factor for smooth apply (default: 0.1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Expand path
    db_path = Path(args.db).expanduser()

    if not db_path.exists():
        error_msg = {"status": "error", "message": f"Database not found at {db_path}"}
        print(json.dumps(error_msg, indent=2))
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure db_metadata table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()

    try:
        # Get current parameters
        current_params = get_current_params(conn)
        current_metrics = analyze_current_performance(conn)

        result: dict[str, Any] = {
            "status": "ok",
            "current_gamma": current_params["gamma"],
            "current_beta_sal": current_params["beta_sal"],
            "current_metrics": {
                "ndcg@1": round(current_metrics["ndcg@1"], 4),
                "ndcg@5": round(current_metrics["ndcg@5"], 4),
                "lift@5": round(current_metrics["lift@5"], 4),
                "num_events": current_metrics["num_events"],
            },
            "last_tuned_at": _get_metadata_str(conn, "last_tuned_at", None),
        }

        # Handle --set-gamma and --set-beta
        if args.set_gamma is not None or args.set_beta_sal is not None:
            gamma = (
                args.set_gamma
                if args.set_gamma is not None
                else current_params["gamma"]
            )
            beta_sal = (
                args.set_beta_sal
                if args.set_beta_sal is not None
                else current_params["beta_sal"]
            )

            applied = smooth_apply(conn, gamma, beta_sal, blend_factor=args.blend)
            result["applied"] = {
                "gamma": round(applied["gamma"], 6),
                "beta_sal": round(applied["beta_sal"], 6),
                "blend_factor": args.blend,
            }
            result["status"] = "applied_smooth"

        # Handle --tune
        elif args.tune:
            num_events = current_metrics["num_events"]
            if num_events < args.min_events:
                result["status"] = "skipped"
                result["reason"] = (
                    f"Only {num_events} events, minimum {args.min_events} required"
                )
            else:
                gamma_range = _parse_range(args.gamma_range)
                beta_sal_range = _parse_range(args.beta_sal_range)

                optimal = find_optimal_params(conn, gamma_range, beta_sal_range)
                gain = compute_quality_gain(
                    conn, optimal["best_gamma"], optimal["best_beta_sal"]
                )

                result["optimal"] = {
                    "gamma": optimal["best_gamma"],
                    "beta_sal": optimal["best_beta_sal"],
                    "ndcg@5": round(optimal["best_ndcg@5"], 4),
                }
                result["projected_gain"] = {
                    "current_ndcg@5": round(gain["current_ndcg@5"], 4),
                    "new_ndcg@5": round(gain["new_ndcg@5"], 4),
                    "gain": round(gain["gain"], 4),
                }

                # Apply smoothly
                applied = smooth_apply(
                    conn,
                    optimal["best_gamma"],
                    optimal["best_beta_sal"],
                    blend_factor=args.blend,
                )
                result["applied"] = {
                    "gamma": round(applied["gamma"], 6),
                    "beta_sal": round(applied["beta_sal"], 6),
                    "blend_factor": args.blend,
                }
                # Also update scoring.py with the new values
                apply_to_scoring_file(applied["gamma"], applied["beta_sal"])
                result["status"] = "tuned"

        # Handle --analyze (default if no action specified)
        elif args.analyze or not any([args.tune, args.set_gamma is not None]):
            # Just analyze, no changes
            pass

        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        print(json.dumps(result, indent=2))
        return 0

    except Exception as exc:
        error_result = {"status": "error", "message": str(exc)}
        print(json.dumps(error_result, indent=2))
        return 1
    finally:
        conn.close()


def _get_metadata_str(
    conn: sqlite3.Connection, key: str, default: str | None
) -> str | None:
    """Get a string value from db_metadata, returning default if not found."""
    row = conn.execute("SELECT value FROM db_metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return row["value"]


if __name__ == "__main__":
    raise SystemExit(main())
