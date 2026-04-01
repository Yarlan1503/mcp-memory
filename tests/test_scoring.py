"""Tests for MCP Memory scoring module.

Tests pure functions that don't require database access.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.mcp_memory.scoring import (
    BETA_DEG,
    BETA_SAL,
    COOC_TEMPORAL_FLOOR,
    D_MAX,
    GAMMA,
    LAMBDA_HOURLY,
    TEMPORAL_FLOOR,
    compute_cooc_decay,
    compute_importance,
    compute_temporal_factor,
    rank_candidates,
    rank_hybrid_candidates,
    rank_with_routing_strategy,
    reciprocal_rank_fusion,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_utc_now_offset(hours_offset: float) -> datetime:
    """Create a UTC datetime offset by hours_offset from NOW."""
    return datetime.now(timezone.utc) - timedelta(hours=hours_offset)


def format_dt(dt: datetime) -> str:
    """Format datetime as SQLite-compatible string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------
# Temporal Decay Tests
# ------------------------------------------------------------------


class TestComputeTemporalDecay:
    """Tests for compute_temporal_factor()."""

    def test_compute_temporal_decay_recent(self):
        """Entity accessed 1 hour ago → decay near 1.0 (little decay)."""
        recent_dt = make_utc_now_offset(1)
        result = compute_temporal_factor(
            last_access_str=format_dt(recent_dt),
            created_at_str="1970-01-01 00:00:00",
        )
        assert result == pytest.approx(1.0, abs=0.001)

    def test_compute_temporal_decay_old(self):
        """Entity accessed 10000 hours ago (~416 days) → decay near TEMPORAL_FLOOR."""
        old_dt = make_utc_now_offset(10000)
        expected = max(TEMPORAL_FLOOR, __import__("math").exp(-LAMBDA_HOURLY * 10000))
        result = compute_temporal_factor(
            last_access_str=format_dt(old_dt),
            created_at_str="1970-01-01 00:00:00",
        )
        # Should be close to the calculated value, but also respect floor
        assert result == pytest.approx(expected, abs=0.01)
        assert result >= TEMPORAL_FLOOR

    def test_compute_temporal_decay_floor(self):
        """Very old entity → decay does not go below 0.1."""
        very_old_dt = make_utc_now_offset(100000)  # ~11.4 years
        result = compute_temporal_factor(
            last_access_str=format_dt(very_old_dt),
            created_at_str="1970-01-01 00:00:00",
        )
        assert result == pytest.approx(TEMPORAL_FLOOR)
        assert result >= TEMPORAL_FLOOR

    def test_compute_temporal_decay_invalid_date(self):
        """Invalid date string → returns 1.0 (neutral)."""
        result = compute_temporal_factor(
            last_access_str="not-a-date",
            created_at_str="also-invalid",
        )
        assert result == 1.0


# ------------------------------------------------------------------
# Co-occurrence Decay Tests
# ------------------------------------------------------------------


class TestComputeCoocDecay:
    """Tests for compute_cooc_decay()."""

    def test_compute_cooc_decay(self):
        """Similar to temporal decay but for co-occurrences."""
        recent_dt = make_utc_now_offset(1)
        result = compute_cooc_decay(format_dt(recent_dt))
        assert result == pytest.approx(1.0, abs=0.001)

    def test_compute_cooc_decay_floor(self):
        """Verify COOC_TEMPORAL_FLOOR = 0.1."""
        very_old_dt = make_utc_now_offset(100000)
        result = compute_cooc_decay(format_dt(very_old_dt))
        assert result == pytest.approx(COOC_TEMPORAL_FLOOR)
        assert result >= COOC_TEMPORAL_FLOOR


# ------------------------------------------------------------------
# Importance Tests
# ------------------------------------------------------------------


class TestComputeImportance:
    """Tests for compute_importance()."""

    def test_compute_importance(self):
        """Verify formula: importance = log2(1+access)/log2(1+max) × (1 + β_deg × min(degree, D_MAX)/D_MAX)."""
        max_access = 100
        access_count = 50
        degree = 10

        access_norm = __import__("math").log2(1 + access_count) / __import__(
            "math"
        ).log2(1 + max_access)
        degree_norm = min(degree, D_MAX) / D_MAX
        expected = access_norm * (1 + BETA_DEG * degree_norm)

        result = compute_importance(
            access_count=access_count,
            max_access=max_access,
            degree=degree,
        )
        assert result == pytest.approx(expected, abs=0.0001)

    def test_compute_importance_zero_max_access(self):
        """max_access=0 → returns 0.0."""
        result = compute_importance(
            access_count=10,
            max_access=0,
            degree=5,
        )
        assert result == 0.0

    def test_compute_importance_degree_capped_at_d_max(self):
        """Degree above D_MAX is capped."""
        result_capped = compute_importance(
            access_count=50,
            max_access=100,
            degree=100,  # Way above D_MAX (15)
        )
        result_normal = compute_importance(
            access_count=50,
            max_access=100,
            degree=15,  # At D_MAX
        )
        assert result_capped == pytest.approx(result_normal)


# ------------------------------------------------------------------
# Hybrid Ranking Tests
# ------------------------------------------------------------------


class TestRankHybridCandidates:
    """Tests for rank_hybrid_candidates()."""

    def test_rank_hybrid_candidates_orders_by_score(self):
        """3 candidates with different scores, verify descending order by limbic_score."""
        now = datetime.now(timezone.utc)

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
            {"entity_id": 2, "rrf_score": 0.8, "distance": 0.2},
            {"entity_id": 3, "rrf_score": 0.3, "distance": 0.05},
        ]

        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
            3: {"access_count": 5, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10, 3: 2}
        cooc_data = {}
        entity_created = {1: format_dt(now), 2: format_dt(now), 3: format_dt(now)}

        result = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=3,
        )

        assert len(result) == 3
        # Verify descending order by limbic_score
        assert result[0]["limbic_score"] >= result[1]["limbic_score"]
        assert result[1]["limbic_score"] >= result[2]["limbic_score"]
        # Verify all entity_ids are present
        assert set(r["entity_id"] for r in result) == {1, 2, 3}

    def test_rank_hybrid_candidates_respects_limit(self):
        """Request limit=2 from 5 candidates, returns only 2."""
        now = datetime.now(timezone.utc)

        merged_results = [
            {"entity_id": i, "rrf_score": 0.5 + i * 0.1, "distance": 0.1}
            for i in range(1, 6)
        ]

        access_data = {
            i: {"access_count": 10, "last_access": format_dt(now)} for i in range(1, 6)
        }
        degree_data = {i: 5 for i in range(1, 6)}
        cooc_data = {}
        entity_created = {i: format_dt(now) for i in range(1, 6)}

        result = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
        )

        assert len(result) == 2

    def test_rank_hybrid_candidates_empty(self):
        """Empty list → returns [].造的"""
        result = rank_hybrid_candidates(
            merged_results=[],
            access_data={},
            degree_data={},
            cooc_data={},
            entity_created={},
            limit=10,
        )
        assert result == []


# ------------------------------------------------------------------
# Reciprocal Rank Fusion Tests
# ------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion()."""

    def test_reciprocal_rank_fusion(self):
        """Entities in both rankings get boost."""
        semantic_results = [
            {"entity_id": 1, "distance": 0.1},
            {"entity_id": 2, "distance": 0.2},
            {"entity_id": 3, "distance": 0.3},
        ]
        fts_results = [
            {"entity_id": 2, "rank": 1.0},
            {"entity_id": 3, "rank": 0.8},
            {"entity_id": 4, "rank": 0.6},
        ]

        result = reciprocal_rank_fusion(semantic_results, fts_results)

        # Entity 2 is in both rankings - should get combined boost
        entity_2_score = next(r["rrf_score"] for r in result if r["entity_id"] == 2)
        # Entity 1 only in semantic
        entity_1_score = next(r["rrf_score"] for r in result if r["entity_id"] == 1)
        # Entity 4 only in FTS
        entity_4_score = next(r["rrf_score"] for r in result if r["entity_id"] == 4)

        # Entity in both should have higher score than either single source
        assert entity_2_score > entity_1_score
        assert entity_2_score > entity_4_score

    def test_reciprocal_rank_fusion_order(self):
        """Results sorted by rrf_score descending."""
        semantic_results = [
            {"entity_id": 1, "distance": 0.1},
            {"entity_id": 2, "distance": 0.2},
            {"entity_id": 3, "distance": 0.3},
        ]
        fts_results = [
            {"entity_id": 3, "rank": 1.0},
            {"entity_id": 2, "rank": 0.8},
            {"entity_id": 1, "rank": 0.6},
        ]

        result = reciprocal_rank_fusion(semantic_results, fts_results)

        # Verify descending order
        for i in range(len(result) - 1):
            assert result[i]["rrf_score"] >= result[i + 1]["rrf_score"]

    def test_reciprocal_rank_fusion_preserves_distance(self):
        """Semantic entities preserve distance, FTS-only have None."""
        semantic_results = [{"entity_id": 1, "distance": 0.1}]
        fts_results = [{"entity_id": 2, "rank": 1.0}]

        result = reciprocal_rank_fusion(semantic_results, fts_results)

        entity_1 = next(r for r in result if r["entity_id"] == 1)
        entity_2 = next(r for r in result if r["entity_id"] == 2)

        assert entity_1["distance"] == 0.1
        assert entity_2["distance"] is None


# ------------------------------------------------------------------
# Gamma/Beta Parameter Override Tests
# ------------------------------------------------------------------


class TestRankCandidatesParameterOverride:
    """Tests for gamma and beta_sal parameter override in rank_candidates()."""

    def test_rank_candidates_accepts_gamma_and_beta_sal(self):
        """rank_candidates should accept gamma and beta_sal as optional parameters."""
        now = datetime.now(timezone.utc)

        knn_results = [
            {"entity_id": 1, "distance": 0.1},
            {"entity_id": 2, "distance": 0.2},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10}
        cooc_data = {}
        entity_created = {1: format_dt(now), 2: format_dt(now)}

        # Should not raise any error
        result = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            gamma=0.05,
            beta_sal=0.75,
        )

        assert len(result) == 2

    def test_rank_candidates_uses_defaults_when_not_passed(self):
        """rank_candidates should use module constants when gamma/beta_sal not passed."""
        now = datetime.now(timezone.utc)

        knn_results = [
            {"entity_id": 1, "distance": 0.1},
            {"entity_id": 2, "distance": 0.2},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10}
        cooc_data = {}
        entity_created = {1: format_dt(now), 2: format_dt(now)}

        # Call without gamma/beta_sal
        result_default = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
        )

        # Call with explicit None
        result_explicit_none = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            gamma=None,
            beta_sal=None,
        )

        # Both should produce identical results
        assert len(result_default) == len(result_explicit_none)
        for i in range(len(result_default)):
            assert result_default[i]["limbic_score"] == pytest.approx(
                result_explicit_none[i]["limbic_score"]
            )

    def test_rank_candidates_uses_passed_values(self):
        """rank_candidates should use passed gamma/beta_sal values over defaults."""
        now = datetime.now(timezone.utc)

        knn_results = [
            {"entity_id": 1, "distance": 0.1},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
        }
        degree_data = {1: 5}
        cooc_data = {(1, 1): {"co_count": 10, "last_co": format_dt(now)}}
        entity_created = {1: format_dt(now)}

        # Call with default values
        result_default = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
        )

        # Call with different gamma (much higher to see effect on cooc boost)
        result_custom_gamma = rank_candidates(
            knn_results=knn_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
            gamma=1.0,  # Much higher than default 0.01
            beta_sal=0.5,
        )

        # Cooc boost effect should be much more visible with gamma=1.0
        # The score should be higher when gamma is higher (since cooc_boost > 0)
        assert (
            result_custom_gamma[0]["limbic_score"] != result_default[0]["limbic_score"]
        )


class TestRankHybridCandidatesParameterOverride:
    """Tests for gamma and beta_sal parameter override in rank_hybrid_candidates()."""

    def test_rank_hybrid_candidates_accepts_gamma_and_beta_sal(self):
        """rank_hybrid_candidates should accept gamma and beta_sal as optional parameters."""
        now = datetime.now(timezone.utc)

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
            {"entity_id": 2, "rrf_score": 0.8, "distance": 0.2},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10}
        cooc_data = {}
        entity_created = {1: format_dt(now), 2: format_dt(now)}

        # Should not raise any error
        result = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            gamma=0.05,
            beta_sal=0.75,
        )

        assert len(result) == 2

    def test_rank_hybrid_candidates_uses_passed_values(self):
        """rank_hybrid_candidates should use passed gamma/beta_sal values."""
        now = datetime.now(timezone.utc)

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
        }
        degree_data = {1: 5}
        cooc_data = {(1, 1): {"co_count": 10, "last_co": format_dt(now)}}
        entity_created = {1: format_dt(now)}

        result_custom = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
            gamma=1.0,
            beta_sal=0.5,
        )

        result_default = rank_hybrid_candidates(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=1,
        )

        # Scores should differ when using custom gamma
        assert result_custom[0]["limbic_score"] != result_default[0]["limbic_score"]


class TestRankWithRoutingStrategyParameterOverride:
    """Tests for gamma and beta_sal parameter override in rank_with_routing_strategy()."""

    def test_rank_with_routing_strategy_accepts_gamma_and_beta_sal(self):
        """rank_with_routing_strategy should accept gamma and beta_sal as optional parameters."""
        from src.mcp_memory.scoring import RoutingStrategy

        now = datetime.now(timezone.utc)

        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
            {"entity_id": 2, "rrf_score": 0.8, "distance": 0.2},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10}
        cooc_data = {}
        entity_created = {1: format_dt(now), 2: format_dt(now)}

        # Should not raise any error
        result = rank_with_routing_strategy(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            strategy=RoutingStrategy.HYBRID_BALANCED,
            gamma=0.05,
            beta_sal=0.75,
        )

        assert len(result) == 2

    def test_rank_with_routing_strategy_passes_params_to_rank_hybrid(self):
        """rank_with_routing_strategy should pass gamma/beta_sal to rank_hybrid_candidates."""
        from src.mcp_memory.scoring import RoutingStrategy

        now = datetime.now(timezone.utc)

        # Use multiple results so normalization distinguishes gamma values
        merged_results = [
            {"entity_id": 1, "rrf_score": 0.5, "distance": 0.1},
            {"entity_id": 2, "rrf_score": 0.8, "distance": 0.2},
        ]
        access_data = {
            1: {"access_count": 10, "last_access": format_dt(now)},
            2: {"access_count": 50, "last_access": format_dt(now)},
        }
        degree_data = {1: 5, 2: 10}
        # Co-occurrence data that will produce different boosts for different gamma
        cooc_data = {
            (1, 2): {"co_count": 10, "last_co": format_dt(now)},
            (2, 1): {"co_count": 5, "last_co": format_dt(now)},
        }
        entity_created = {1: format_dt(now), 2: format_dt(now)}

        result_custom = rank_with_routing_strategy(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            strategy=RoutingStrategy.HYBRID_BALANCED,
            gamma=1.0,  # Much higher gamma
            beta_sal=0.5,
        )

        result_default = rank_with_routing_strategy(
            merged_results=merged_results,
            access_data=access_data,
            degree_data=degree_data,
            cooc_data=cooc_data,
            entity_created=entity_created,
            limit=2,
            strategy=RoutingStrategy.HYBRID_BALANCED,
        )

        # The limbic_scores should differ when using custom gamma
        # (final_score may end up same due to min-max normalization, but the underlying
        # limbic_scores computed with rank_hybrid_candidates should be different)
        assert result_custom[0]["limbic_score"] != result_default[0]["limbic_score"]
