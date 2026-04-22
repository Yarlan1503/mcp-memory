"""Access tracking and co-occurrence scoring for the MCP Memory knowledge graph."""

from typing import Any

import logging

from mcp_memory.retry import retry_on_locked

logger = logging.getLogger(__name__)


class AccessMixin:
    """Access tracking and co-occurrence methods for MemoryStore."""

    # ------------------------------------------------------------------
    # Limbic: Access tracking
    # ------------------------------------------------------------------

    @retry_on_locked
    def init_access(self, entity_id: int) -> None:
        """Initialize access tracking for a new entity (access_count=1)."""
        self.db.execute(
            "INSERT OR IGNORE INTO entity_access (entity_id, access_count, last_access) VALUES (?, 1, datetime('now'))",
            (entity_id,),
        )
        self.db.commit()

    @retry_on_locked
    def record_access(self, entity_id: int) -> None:
        """Record an access event: increment count and update last_access.

        Also records a daily entry in entity_access_log for consolidation tracking.
        """
        self.db.execute(
            """
            INSERT INTO entity_access (entity_id, access_count, last_access)
            VALUES (?, 1, datetime('now'))
            ON CONFLICT(entity_id) DO UPDATE SET
                access_count = access_count + 1,
                last_access = datetime('now')
            """,
            (entity_id,),
        )
        self.db.execute(
            """
            INSERT INTO entity_access_log (entity_id, access_date, access_count)
            VALUES (?, DATE('now'), 1)
            ON CONFLICT(entity_id, access_date) DO UPDATE SET
                access_count = access_count + 1
            """,
            (entity_id,),
        )
        self.db.commit()

    def get_access_data(self, entity_ids: list[int]) -> dict[int, dict]:
        """Get access data for a list of entity IDs.
        Returns dict with access_count and last_access for each.
        Defaults to access_count=0, last_access=created_at if no record exists."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT e.id, e.created_at,
                   COALESCE(ea.access_count, 0) AS access_count,
                   COALESCE(ea.last_access, e.created_at) AS last_access
            FROM entities e
            LEFT JOIN entity_access ea ON ea.entity_id = e.id
            WHERE e.id IN ({placeholders})
            """,
            entity_ids,
        ).fetchall()
        return {
            r["id"]: {
                "access_count": r["access_count"],
                "last_access": r["last_access"],
            }
            for r in rows
        }

    def get_access_days(self, entity_ids: list[int]) -> dict[int, int]:
        """Get count of unique days with access for each entity.

        Returns {entity_id: access_days_count}.
        Entities with no rows in entity_access_log (pre-migration accesses)
        are not included in the result — callers should default to 1.
        """
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_id, COUNT(*) as access_days
            FROM entity_access_log
            WHERE entity_id IN ({placeholders})
            GROUP BY entity_id
            """,
            entity_ids,
        ).fetchall()
        return {r["entity_id"]: r["access_days"] for r in rows}

    def days_since_access(self, last_access: str | None) -> int | None:
        """Compute days since last access using SQLite julianday (avoids timezone issues)."""
        if last_access is None:
            return None
        try:
            row = self.db.execute(
                "SELECT CAST(julianday('now') - julianday(?) AS INTEGER)",
                (last_access,),
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def get_entity_degrees(self, entity_ids: list[int]) -> dict[int, int]:
        """Get relation degree count for a list of entity IDs.
        Counts relations where entity is either from_entity or to_entity."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_id, COUNT(*) AS degree FROM (
                SELECT from_entity AS entity_id FROM relations WHERE from_entity IN ({placeholders})
                UNION ALL
                SELECT to_entity AS entity_id FROM relations WHERE to_entity IN ({placeholders})
            ) GROUP BY entity_id
            """,
            entity_ids + entity_ids,
        ).fetchall()
        result = {eid: 0 for eid in entity_ids}
        for r in rows:
            result[r["entity_id"]] = r["degree"]
        return result

    # ------------------------------------------------------------------
    # Limbic: Co-occurrence tracking
    # ------------------------------------------------------------------

    @retry_on_locked
    def record_co_occurrences(self, entity_ids: list[int]) -> None:
        """Record co-occurrences for all unique pairs of entity IDs.
        Canonical ordering: entity_a_id < entity_b_id always."""
        if len(entity_ids) < 2:
            return
        sorted_ids = sorted(entity_ids)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                a, b = sorted_ids[i], sorted_ids[j]
                self.db.execute(
                    """
                    INSERT INTO co_occurrences (entity_a_id, entity_b_id, co_count, last_co)
                    VALUES (?, ?, 1, datetime('now'))
                    ON CONFLICT(entity_a_id, entity_b_id) DO UPDATE SET
                        co_count = co_count + 1,
                        last_co = datetime('now')
                    """,
                    (a, b),
                )
        self.db.commit()

    def get_co_occurrences(
        self, entity_ids: list[int]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Get co-occurrence counts for pairs within a set of entity IDs.
        Returns dict with (id_a, id_b) -> {"co_count": int, "last_co": str}, where id_a < id_b."""
        if len(entity_ids) < 2:
            return {}
        id_set = set(entity_ids)
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.db.execute(
            f"""
            SELECT entity_a_id, entity_b_id, co_count, last_co
            FROM co_occurrences
            WHERE entity_a_id IN ({placeholders}) AND entity_b_id IN ({placeholders})
            """,
            entity_ids + entity_ids,
        ).fetchall()
        return {
            (r["entity_a_id"], r["entity_b_id"]): {
                "co_count": r["co_count"],
                "last_co": r["last_co"],
            }
            for r in rows
            if r["entity_a_id"] in id_set and r["entity_b_id"] in id_set
        }
