"""Tests for multi-day consolidation signal in limbic scoring.

Tests cover:
- compute_importance() with consolidation parameters
- storage record_access() writing to entity_access_log
- get_access_days() retrieval
- rank_candidates() / rank_hybrid_candidates() with access_days_data
- Retrocompatibility when access_days_data is empty
"""

import math
from datetime import datetime, timezone

import pytest

from src.mcp_memory.scoring import (
    ALPHA_CONS,
    BETA_DEG,
    D_MAX,
    compute_importance,
    rank_candidates,
    rank_hybrid_candidates,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def format_dt(dt: datetime) -> str:
    """Format datetime as SQLite-compatible string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def now_str() -> str:
    return format_dt(datetime.now(timezone.utc))


# ------------------------------------------------------------------
# compute_importance consolidation tests
# ------------------------------------------------------------------


class TestImportanceConsolidation:
    """Tests for consolidation signal in compute_importance()."""

    def test_importance_with_zero_consolidation(self):
        """access_days=1, max_access_days=1 → same importance as before.

        When max_access_days <= 1, consolidation is neutral (multiplier = 1.0).
        """
        result = compute_importance(
            access_count=50,
            max_access=100,
            degree=5,
            access_days=1,
            max_access_days=1,
        )

        # Expected: same as old formula (no consolidation)
        access_norm = math.log2(1 + 50) / math.log2(1 + 100)
        degree_norm = min(5, D_MAX) / D_MAX
        expected = access_norm * (1 + BETA_DEG * degree_norm)

        assert result == pytest.approx(expected, abs=0.0001)

    def test_importance_with_high_consolidation(self):
        """Entity with more access_days gets higher importance than one with fewer."""
        base_importance = compute_importance(
            access_count=50,
            max_access=100,
            degree=5,
            access_days=1,
            max_access_days=10,
        )

        high_consolidation = compute_importance(
            access_count=50,
            max_access=100,
            degree=5,
            access_days=10,
            max_access_days=10,
        )

        assert high_consolidation > base_importance

    def test_importance_consolidation_factor_formula(self):
        """Verify formula (1 + α_cons × consolidation) with known values."""
        access_count = 10
        max_access = 10
        degree = 0  # No degree factor to isolate consolidation
        access_days = 5
        max_access_days = 10

        result = compute_importance(
            access_count=access_count,
            max_access=max_access,
            degree=degree,
            access_days=access_days,
            max_access_days=max_access_days,
        )

        access_norm = math.log2(1 + access_count) / math.log2(1 + max_access)
        consolidation = math.log2(1 + access_days) / math.log2(1 + max_access_days)
        expected = access_norm * (1 + ALPHA_CONS * consolidation)

        assert result == pytest.approx(expected, abs=0.0001)

    def test_importance_default_params_match_old_behavior(self):
        """Calling without access_days params produces same result as old formula."""
        result_with_defaults = compute_importance(
            access_count=50,
            max_access=100,
            degree=5,
        )

        # Old formula
        access_norm = math.log2(1 + 50) / math.log2(1 + 100)
        degree_norm = min(5, D_MAX) / D_MAX
        expected = access_norm * (1 + BETA_DEG * degree_norm)

        assert result_with_defaults == pytest.approx(expected, abs=0.0001)

    def test_importance_max_access_days_zero(self):
        """max_access_days=0 → consolidation neutral (same as 1)."""
        result = compute_importance(
            access_count=50,
            max_access=100,
            degree=5,
            access_days=1,
            max_access_days=0,
        )

        # Should be identical to no consolidation
        access_norm = math.log2(1 + 50) / math.log2(1 + 100)
        degree_norm = min(5, D_MAX) / D_MAX
        expected = access_norm * (1 + BETA_DEG * degree_norm)

        assert result == pytest.approx(expected, abs=0.0001)


# ------------------------------------------------------------------
# Storage: entity_access_log tests
# ------------------------------------------------------------------


class TestEntityAccessLog:
    """Tests for entity_access_log table and record_access()."""

    def test_init_db_creates_access_log_table(self, store_with_schema):
        """entity_access_log table exists after init_db()."""
        tables = [
            r[0]
            for r in store_with_schema.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "entity_access_log" in tables

    def test_record_access_updates_log(self, store_with_schema):
        """record_access creates a row in entity_access_log."""
        eid = store_with_schema.upsert_entity("TestLog", "Test")
        store_with_schema.record_access(eid)

        rows = store_with_schema.db.execute(
            "SELECT entity_id, access_date, access_count FROM entity_access_log"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["entity_id"] == eid
        assert rows[0]["access_count"] == 1

    def test_record_access_same_day_increments(self, store_with_schema):
        """Two accesses on the same day → access_count in log = 2, access_days = 1."""
        eid = store_with_schema.upsert_entity("TestSameDay", "Test")
        store_with_schema.record_access(eid)
        store_with_schema.record_access(eid)

        rows = store_with_schema.db.execute(
            "SELECT entity_id, access_date, access_count FROM entity_access_log"
        ).fetchall()
        assert len(rows) == 1  # Only one row (same date)
        assert rows[0]["access_count"] == 2

    def test_record_access_different_days(self, store_with_schema):
        """Two accesses on different days → access_days = 2."""
        eid = store_with_schema.upsert_entity("TestDiffDay", "Test")

        # First access (today)
        store_with_schema.record_access(eid)

        # Simulate second access on a different day
        store_with_schema.db.execute(
            """
            INSERT INTO entity_access_log (entity_id, access_date, access_count)
            VALUES (?, '2025-01-01', 1)
            ON CONFLICT(entity_id, access_date) DO UPDATE SET
                access_count = access_count + 1
            """,
            (eid,),
        )
        store_with_schema.db.commit()

        # get_access_days should return 2
        result = store_with_schema.get_access_days([eid])
        assert result[eid] == 2

    def test_get_access_days_no_log_fallback(self, store_with_schema):
        """Entity without access_log rows → not in result (caller defaults to 1)."""
        eid = store_with_schema.upsert_entity("TestNoLog", "Test")
        # Don't call record_access — no log entries

        result = store_with_schema.get_access_days([eid])
        assert eid not in result  # Caller should default to 1

    def test_get_access_days_empty_input(self, store_with_schema):
        """Empty entity_ids → empty dict."""
        result = store_with_schema.get_access_days([])
        assert result == {}

    def test_get_access_days_multiple_entities(self, store_with_schema):
        """get_access_days returns correct counts for multiple entities."""
        e1 = store_with_schema.upsert_entity("MultiE1", "Test")
        e2 = store_with_schema.upsert_entity("MultiE2", "Test")

        store_with_schema.record_access(e1)
        store_with_schema.record_access(e1)  # Same day, increments count

        # e2: simulate 3 different days
        for date_str in ["2025-01-01", "2025-01-02", "2025-01-03"]:
            store_with_schema.db.execute(
                """
                INSERT INTO entity_access_log (entity_id, access_date, access_count)
                VALUES (?, ?, 1)
                ON CONFLICT(entity_id, access_date) DO UPDATE SET
                    access_count = access_count + 1
                """,
                (e2, date_str),
            )
        store_with_schema.db.commit()

        result = store_with_schema.get_access_days([e1, e2])
        assert result[e1] == 1  # Only 1 unique day
        assert result[e2] == 3  # 3 unique days


# ------------------------------------------------------------------
# rank_candidates with access_days_data
# ------------------------------------------------------------------


class TestRankCandidatesConsolidation:
    """Tests for rank_candidates() with access_days_data parameter."""

    def test_rank_candidates_with_access_days(self):
        """Entity with more access_days gets higher limbic_score."""
        now = now_str()

        knn_results = [
            {"entity_id": 1, "distance": 0.1},
            {"entity_id": 2, "distance": 0.1},
        ]

        # Same access_count but different access_days
        access_data = {
            1: {"access_count": 10, "last_access": now},
            2: {"access_count": 10, "last_access": now},
        }
        degree_data = {1: 0, 2: 0}
        cooc_data = {}
        entity_created = {1: now, 2: now}

        # Entity 2 has more access days (consolidated)
        access_days_data = {1: 1, 2: 5}

        result = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            access_days_data=access_days_data,
        )

        # Both should be returned
        assert len(result) == 2
        # Entity 2 (consolidated) should rank higher
        scores = {r["entity_id"]: r["limbic_score"] for r in result}
        assert scores[2] > scores[1]

    def test_rank_candidates_without_access_days_empty_dict(self):
        """access_days_data={} → retrocompatible (scoring identical to old behavior)."""
        now = now_str()

        knn_results = [
            {"entity_id": 1, "distance": 0.1},
        ]
        access_data = {1: {"access_count": 10, "last_access": now}}
        degree_data = {1: 5}
        cooc_data = {}
        entity_created = {1: now}

        result_without = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
            access_days_data={},
        )

        result_none = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
        )

        # Both should produce identical scores
        assert result_without[0]["limbic_score"] == pytest.approx(
            result_none[0]["limbic_score"]
        )


class TestRankHybridCandidatesConsolidation:
    """Tests for rank_hybrid_candidates() with access_days_data parameter."""

    def test_rank_hybrid_with_access_days(self):
        """rank_hybrid_candidates respects access_days_data."""
        now = now_str()

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
            {"entity_id": 2, "rrf_score": 0.5, "distance": 0.1},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": now},
            2: {"access_count": 10, "last_access": now},
        }
        degree_data = {1: 0, 2: 0}
        cooc_data = {}
        entity_created = {1: now, 2: now}

        access_days_data = {1: 1, 2: 5}

        result = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            access_days_data=access_days_data,
        )

        scores = {r["entity_id"]: r["limbic_score"] for r in result}
        assert scores[2] > scores[1]

    def test_rank_hybrid_retrocompatible_no_access_days(self):
        """rank_hybrid_candidates without access_days_data → old behavior."""
        now = now_str()

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
        ]
        access_data = {1: {"access_count": 10, "last_access": now}}
        degree_data = {1: 5}
        cooc_data = {}
        entity_created = {1: now}

        result = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
        )

        # Should not raise and should produce valid score
        assert len(result) == 1
        assert result[0]["limbic_score"] > 0


# ------------------------------------------------------------------
# Migration idempotency
# ------------------------------------------------------------------


class TestMigrationIdempotency:
    """Tests that init_db() can be called multiple times without errors."""

    def test_init_db_idempotent(self, memory_store):
        """Calling init_db() twice does not raise."""
        memory_store.init_db()
        memory_store.init_db()  # Second call should be fine

        # Verify table exists
        tables = [
            r[0]
            for r in memory_store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "entity_access_log" in tables

    def test_init_db_idempotent_with_data(self, store_with_schema):
        """Calling init_db() after data exists does not lose data."""
        eid = store_with_schema.upsert_entity("Idempotent", "Test")
        store_with_schema.record_access(eid)

        # Re-init
        store_with_schema.init_db()

        # Data should still be there
        result = store_with_schema.get_access_days([eid])
        assert result[eid] == 1
