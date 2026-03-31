"""Grid Search for A/B Testing Hyperparameters.

Explora el espacio GAMMA × BETA_SAL para encontrar la mejor configuración.
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import Any


# Parámetros actuales (baseline)
BASE_GAMMA = 0.01
BASE_BETA_SAL = 0.5


def recompute_score(
    cosine_sim: float,
    importance: float,
    temporal: float,
    cooc_boost: float,
    gamma: float,
    beta_sal: float,
) -> float:
    """Recompute limbic score con parámetros dados."""
    return (
        cosine_sim * (1 + beta_sal * importance) * temporal * (1 + gamma * cooc_boost)
    )


def grid_search(
    conn: sqlite3.Connection,
    gamma_values: list[float],
    beta_sal_values: list[float],
) -> dict[str, Any]:
    """Realizar grid search sobre gamma y beta_sal."""
    cursor = conn.execute(
        """
        SELECT sr.event_id, sr.entity_id, sr.rank, sr.limbic_score, 
               sr.cosine_sim, sr.importance, sr.temporal, sr.cooc_boost,
               sr.baseline_rank
        FROM search_results sr
        JOIN search_events se ON se.event_id = sr.event_id
        WHERE se.treatment = 1
        ORDER BY sr.event_id, sr.rank
        """
    )

    # Agrupar por evento
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
                "baseline_rank": row[8],
            }
        )

    results = []

    for gamma in gamma_values:
        for beta_sal in beta_sal_values:
            event_scores = []

            for event_id, items in events.items():
                # Recomputar scores con nuevos parámetros
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

                # Ordenar por nuevo score
                scored.sort(key=lambda x: x["new_score"], reverse=True)

                # Calcular metrics para este evento
                # Comparar ranking nuevo vs baseline_rank
                correct_positions = 0
                for i, item in enumerate(scored[:10]):
                    if item["baseline_rank"] and item["baseline_rank"] <= 10:
                        correct_positions += 1

                event_scores.append(correct_positions / 10)

            avg_score = sum(event_scores) / len(event_scores) if event_scores else 0
            results.append(
                {
                    "gamma": gamma,
                    "beta_sal": beta_sal,
                    "avg_relevance": avg_score,
                }
            )

    # Ordenar por avg_relevance descending
    results.sort(key=lambda x: x["avg_relevance"], reverse=True)
    return {"grid_search_results": results}


def main():
    parser = argparse.ArgumentParser(description="Grid Search for A/B Hyperparameters")
    parser.add_argument(
        "--db",
        default="~/.config/opencode/mcp-memory/memory.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--gamma",
        default="0.001,0.005,0.01,0.05,0.1",
        help="Comma-separated gamma values",
    )
    parser.add_argument(
        "--beta-sal",
        default="0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated beta_sal values",
    )
    parser.add_argument(
        "--export-json",
        help="Export results to JSON file",
    )
    args = parser.parse_args()

    gamma_values = [float(x) for x in args.gamma.split(",")]
    beta_sal_values = [float(x) for x in args.beta_sal.split(",")]

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    results = grid_search(conn, gamma_values, beta_sal_values)

    print(json.dumps(results, indent=2))

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(results, f, indent=2)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
