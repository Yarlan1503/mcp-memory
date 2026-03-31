"""A/B Testing Metrics Analysis for MCP Memory.

Mide la efectividad del scoring límbico vs baseline (cosine-only).
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Any


def compute_precision_at_k(
    results: list[dict],  # [{entity_id, rank, baseline_rank}]
    k: int,
) -> float:
    """Precision@K: proporción de resultados limbic en top-K que también estaban en baseline top-K."""
    if k <= 0:
        return 0.0

    limbic_top_k = {r["entity_id"] for r in results[:k]}
    baseline_top_k = {
        r["entity_id"]
        for r in sorted(results, key=lambda x: x.get("baseline_rank", 999))[:k]
    }

    if not baseline_top_k:
        return 0.0

    return len(limbic_top_k & baseline_top_k) / k


def compute_mrr_at_k(
    results: list[dict],  # [{entity_id, rank}]
    k: int,
) -> float:
    """Mean Reciprocal Rank@K: inversa del rank del primer resultado relevante."""
    for r in results[:k]:
        if r["rank"] <= k:
            return 1.0 / r["rank"]
    return 0.0


def compute_ndcg_at_k(
    results: list[dict],  # [{entity_id, rank, limbic_score}]
    k: int,
) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    # DCG@K = Σ (2^rel_i - 1) / log2(i+1)
    # IDCG@K = DCG de ranking perfecto
    # NDCG = DCG / IDCG

    def dcg(results_subset):
        total = 0.0
        for i, r in enumerate(results_subset[:k]):
            rel = r.get("limbic_score", 1.0)  # usar score como relevancia
            total += (2 ** min(rel, 1.0) - 1) / (i + 1) ** 0.5  # log2(i+2) simplificado
        return total

    dcg_val = dcg(results)

    # IDCG = ranking perfecto por limbic_score
    ideal = sorted(
        results[: k * 3], key=lambda x: x.get("limbic_score", 0), reverse=True
    )[:k]
    idcg_val = dcg(ideal)

    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def compute_lift_at_k(
    results: list[dict],
    k: int,
) -> float:
    """Lift@K: mejora promedio en scores de limbic vs baseline."""
    limbic_scores = [r.get("limbic_score", 0) for r in results[:k]]
    # Promedio de limbic_score en top-K
    avg_limbic = sum(limbic_scores) / len(limbic_scores) if limbic_scores else 0

    # Promedio de cosine_sim (baseline) en top-K
    baseline_scores = [r.get("cosine_sim", 0) for r in results[:k]]
    avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

    if avg_baseline == 0:
        return float("inf") if avg_limbic > 0 else 0.0

    return avg_limbic / avg_baseline


def compute_metrics_for_event(
    conn: sqlite3.Connection, event_id: int
) -> dict[str, Any]:
    """Computar todas las métricas para un evento específico."""
    cursor = conn.execute(
        """
        SELECT entity_id, entity_name, rank, limbic_score, cosine_sim, 
               importance, temporal, cooc_boost, baseline_rank
        FROM search_results
        WHERE event_id = ?
        ORDER BY rank
        """,
        (event_id,),
    )
    results = [dict(row) for row in cursor.fetchall()]

    if not results:
        return {}

    k_values = [1, 3, 5, 10]
    metrics = {"event_id": event_id}

    for k in k_values:
        metrics[f"precision@{k}"] = compute_precision_at_k(results, k)
        metrics[f"mrr@{k}"] = compute_mrr_at_k(results, k)
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(results, k)
        metrics[f"lift@{k}"] = compute_lift_at_k(results, k)

    return metrics


def compute_aggregate_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Computar métricas agregadas para todos los eventos."""
    cursor = conn.execute(
        "SELECT DISTINCT event_id FROM search_events ORDER BY event_id"
    )
    event_ids = [row[0] for row in cursor.fetchall()]

    if not event_ids:
        return {"error": "No events found"}

    all_metrics = []
    for eid in event_ids:
        m = compute_metrics_for_event(conn, eid)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        return {"error": "No results found"}

    # Promediar across events
    k_values = [1, 3, 5, 10]
    agg = {}
    for k in k_values:
        vals = [m[f"precision@{k}"] for m in all_metrics if f"precision@{k}" in m]
        agg[f"precision@{k}_mean"] = sum(vals) / len(vals) if vals else 0

        vals = [m[f"mrr@{k}"] for m in all_metrics if f"mrr@{k}" in m]
        agg[f"mrr@{k}_mean"] = sum(vals) / len(vals) if vals else 0

        vals = [m[f"ndcg@{k}"] for m in all_metrics if f"ndcg@{k}" in m]
        agg[f"ndcg@{k}_mean"] = sum(vals) / len(vals) if vals else 0

        vals = [m[f"lift@{k}"] for m in all_metrics if f"lift@{k}" in m]
        agg[f"lift@{k}_mean"] = sum(vals) / len(vals) if vals else 0

    agg["num_events"] = len(event_ids)
    return agg


def main():
    parser = argparse.ArgumentParser(description="A/B Testing Metrics Analysis")
    parser.add_argument(
        "--db",
        default="~/.config/opencode/mcp-memory/memory.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        help="Analyze specific event (optional)",
    )
    parser.add_argument(
        "--export-json",
        help="Export results to JSON file (optional)",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))

    if args.event_id:
        metrics = compute_metrics_for_event(conn, args.event_id)
    else:
        metrics = compute_aggregate_metrics(conn)

    import json

    print(json.dumps(metrics, indent=2))

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nExported to {args.export_json}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
