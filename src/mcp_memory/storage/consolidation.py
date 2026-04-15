"""Consolidation report queries for the MCP Memory knowledge graph."""

from typing import Any

import logging

logger = logging.getLogger(__name__)


class ConsolidationMixin:
    """Consolidation report methods for MemoryStore."""

    def get_consolidation_data(self, stale_days: float = 90.0) -> dict[str, Any]:
        """Collect data for a consolidation report. All queries are read-only.

        Args:
            stale_days: Number of days threshold for considering an entity stale.

        Returns:
            Dict with total_entities, total_observations, flagged_observations,
            stale_entities, and entity_sizes.
        """
        data: dict[str, Any] = {}

        # --- 1. Totals ---
        data["total_entities"] = self.db.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        data["total_observations"] = self.db.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]

        # --- 2. Flagged observations (similarity_flag = 1) ---
        rows = self.db.execute(
            """
            SELECT e.name, o.id, o.content, o.similarity_flag
            FROM observations o
            JOIN entities e ON e.id = o.entity_id
            WHERE o.similarity_flag = 1
            ORDER BY e.name, o.id
            """
        ).fetchall()

        flagged: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            name = r["name"]
            if name not in flagged:
                flagged[name] = []
            flagged[name].append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "similarity_flag": r["similarity_flag"],
                }
            )
        data["flagged_observations"] = flagged

        # --- 3. Stale entities ---
        # Entities with observations but not accessed in N days and low access count.
        # LEFT JOIN entity_access so entities without access records are included.
        # COALESCE handles NULLs from missing entity_access rows.
        stale_param = str(int(stale_days))
        rows = self.db.execute(
            """
            SELECT e.name, e.entity_type, e.status, e.created_at,
                   COALESCE(ea.access_count, 0) AS access_count,
                   ea.last_access,
                   COUNT(o.id) AS obs_count
            FROM entities e
            LEFT JOIN entity_access ea ON ea.entity_id = e.id
            JOIN observations o ON o.entity_id = e.id
            GROUP BY e.id
            HAVING (ea.last_access IS NULL
                    OR datetime(ea.last_access) < datetime('now', '-' || ? || ' days'))
               AND COALESCE(ea.access_count, 0) <= 2
               AND e.status != 'archivado'
            ORDER BY CASE WHEN ea.last_access IS NULL THEN 0 ELSE 1 END,
                     ea.last_access ASC
            """,
            (stale_param,),
        ).fetchall()

        data["stale_entities"] = [
            {
                "entity_name": r["name"],
                "entity_type": r["entity_type"],
                "status": r["status"],
                "created_at": r["created_at"],
                "access_count": r["access_count"],
                "last_access": r["last_access"],
                "observation_count": r["obs_count"],
            }
            for r in rows
        ]

        # --- 4. Entity sizes ---
        rows = self.db.execute(
            """
            SELECT e.name, COUNT(o.id) AS obs_count
            FROM entities e
            LEFT JOIN observations o ON o.entity_id = e.id
            GROUP BY e.id
            ORDER BY obs_count DESC
            """
        ).fetchall()

        data["entity_sizes"] = {r["name"]: r["obs_count"] for r in rows}

        return data
