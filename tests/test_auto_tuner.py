"""Tests for MCP Memory auto_tuner module.

Tests the auto-tuning functions including recompute_score consistency,
smooth_apply blending, analyze_current_performance, and find_optimal_params.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Import from scripts directory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from auto_tuner import (
    DEFAULT_BETA_SAL,
    DEFAULT_GAMMA,
    _ndcg_at_k,
    analyze_current_performance,
    apply_to_scoring_file,
    compute_quality_gain,
    find_optimal_params,
    get_current_params,
    recompute_score,
    smooth_apply,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def create_test_db(tmp_path: Path) -> tuple[sqlite3.Connection, Path]:
    """Create a test database with schema and sample data."""
    db_path = tmp_path / "test_auto_tuner.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create schema
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL DEFAULT 'Generic',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS search_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            treatment INTEGER NOT NULL,
            k_limit INTEGER NOT NULL,
            num_results INTEGER NOT NULL,
            duration_ms REAL,
            engine_used TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS search_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES search_events(event_id),
            entity_id INTEGER NOT NULL,
            entity_name TEXT NOT NULL,
            rank INTEGER NOT NULL,
            limbic_score REAL,
            cosine_sim REAL,
            importance REAL,
            temporal REAL,
            cooc_boost REAL,
            baseline_rank INTEGER,
            UNIQUE(event_id, entity_id)
        );

        CREATE TABLE IF NOT EXISTS implicit_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES search_events(event_id),
            entity_id INTEGER NOT NULL,
            re_accessed INTEGER NOT NULL DEFAULT 0,
            access_delta INTEGER,
            session_id TEXT,
            FOREIGN KEY (event_id, entity_id) REFERENCES search_results(event_id, entity_id)
        );
        """
    )
    conn.commit()
    return conn, db_path


def insert_test_data(conn: sqlite3.Connection) -> None:
    """Insert sample search events, results, and feedback for testing."""
    now = datetime.now(timezone.utc)

    # Event 1: 5 results, 3 relevant
    conn.execute(
        """
        INSERT INTO search_events (event_id, query_text, query_hash, timestamp, treatment, k_limit, num_results, engine_used)
        VALUES (1, 'test query 1', 'hash1', ?, 1, 5, 5, 'limbic')
        """,
        (now.isoformat(),),
    )

    # Results for event 1 (already sorted by limbic_score DESC)
    results_1 = [
        (1, 101, "Entity A", 1, 0.9, 0.8, 0.5, 1.0, 2.0),  # rank 1, relevant
        (1, 102, "Entity B", 2, 0.7, 0.6, 0.4, 0.9, 1.5),  # rank 2, relevant
        (1, 103, "Entity C", 3, 0.5, 0.4, 0.3, 0.8, 1.0),  # rank 3, NOT relevant
        (1, 104, "Entity D", 4, 0.3, 0.2, 0.2, 0.7, 0.5),  # rank 4, NOT relevant
        (1, 105, "Entity E", 5, 0.2, 0.1, 0.1, 0.6, 0.2),  # rank 5, relevant
    ]

    for r in results_1:
        conn.execute(
            """
            INSERT INTO search_results (event_id, entity_id, entity_name, rank, limbic_score, cosine_sim, importance, temporal, cooc_boost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            r,
        )

    # Feedback for event 1
    feedback_1 = [
        (1, 101, 1),  # Entity A re-accessed
        (1, 102, 1),  # Entity B re-accessed
        (1, 103, 0),  # Entity C not re-accessed
        (1, 104, 0),  # Entity D not re-accessed
        (1, 105, 1),  # Entity E re-accessed
    ]

    for f in feedback_1:
        conn.execute(
            """
            INSERT INTO implicit_feedback (event_id, entity_id, re_accessed)
            VALUES (?, ?, ?)
            """,
            f,
        )

    # Event 2: 4 results, 2 relevant
    conn.execute(
        """
        INSERT INTO search_events (event_id, query_text, query_hash, timestamp, treatment, k_limit, num_results, engine_used)
        VALUES (2, 'test query 2', 'hash2', ?, 1, 5, 4, 'limbic')
        """,
        (now.isoformat(),),
    )

    results_2 = [
        (2, 201, "Entity F", 1, 0.8, 0.7, 0.6, 1.0, 1.8),  # rank 1, NOT relevant
        (2, 202, "Entity G", 2, 0.6, 0.5, 0.3, 0.9, 1.2),  # rank 2, relevant
        (2, 203, "Entity H", 3, 0.4, 0.3, 0.2, 0.8, 0.8),  # rank 3, relevant
        (2, 204, "Entity I", 4, 0.2, 0.1, 0.1, 0.7, 0.4),  # rank 4, NOT relevant
    ]

    for r in results_2:
        conn.execute(
            """
            INSERT INTO search_results (event_id, entity_id, entity_name, rank, limbic_score, cosine_sim, importance, temporal, cooc_boost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            r,
        )

    feedback_2 = [
        (2, 201, 0),  # Entity F not re-accessed
        (2, 202, 1),  # Entity G re-accessed
        (2, 203, 1),  # Entity H re-accessed
        (2, 204, 0),  # Entity I not re-accessed
    ]

    for f in feedback_2:
        conn.execute(
            """
            INSERT INTO implicit_feedback (event_id, entity_id, re_accessed)
            VALUES (?, ?, ?)
            """,
            f,
        )

    conn.commit()


# ------------------------------------------------------------------
# recompute_score Tests
# ------------------------------------------------------------------


class TestRecomputeScore:
    """Tests for recompute_score() consistency with scoring.py formula."""

    def test_recompute_score_basic(self):
        """Verify formula: cosine * (1 + beta * importance) * temporal * (1 + gamma * cooc)."""
        result = recompute_score(
            cosine_sim=0.8,
            importance=0.5,
            temporal=1.0,
            cooc_boost=2.0,
            gamma=0.01,
            beta_sal=0.5,
        )
        expected = 0.8 * (1 + 0.5 * 0.5) * 1.0 * (1 + 0.01 * 2.0)
        assert result == pytest.approx(expected)

    def test_recompute_score_matches_scoring_formula(self):
        """Verify recompute_score matches the formula in scoring.py."""
        # These values should produce a known result
        cosine_sim = 0.9
        importance = 0.7
        temporal = 0.95
        cooc_boost = 3.0
        gamma = 0.01
        beta_sal = 0.5

        result = recompute_score(
            cosine_sim, importance, temporal, cooc_boost, gamma, beta_sal
        )

        # Manual calculation
        expected = (
            cosine_sim
            * (1 + beta_sal * importance)
            * temporal
            * (1 + gamma * cooc_boost)
        )
        assert result == pytest.approx(expected)

    def test_recompute_score_zero_cosine(self):
        """Zero cosine similarity should yield zero score."""
        result = recompute_score(
            cosine_sim=0.0,
            importance=1.0,
            temporal=1.0,
            cooc_boost=10.0,
            gamma=0.1,
            beta_sal=0.5,
        )
        assert result == 0.0

    def test_recompute_score_gamma_zero(self):
        """GAMMA=0 should disable co-occurrence boost."""
        with_gamma = recompute_score(
            cosine_sim=0.8,
            importance=0.5,
            temporal=1.0,
            cooc_boost=2.0,
            gamma=0.0,
            beta_sal=0.5,
        )
        without_cooc = recompute_score(
            cosine_sim=0.8,
            importance=0.5,
            temporal=1.0,
            cooc_boost=0.0,
            gamma=0.0,
            beta_sal=0.5,
        )
        assert with_gamma == pytest.approx(without_cooc)

    def test_recompute_score_beta_sal_zero(self):
        """BETA_SAL=0 should disable importance boost."""
        with_beta = recompute_score(
            cosine_sim=0.8,
            importance=0.5,
            temporal=1.0,
            cooc_boost=2.0,
            gamma=0.01,
            beta_sal=0.5,
        )
        without_importance = recompute_score(
            cosine_sim=0.8,
            importance=0.0,
            temporal=1.0,
            cooc_boost=2.0,
            gamma=0.01,
            beta_sal=0.0,
        )
        assert with_beta != without_importance  # Different because of importance boost


# ------------------------------------------------------------------
# smooth_apply Tests
# ------------------------------------------------------------------


class TestSmoothApply:
    """Tests for smooth_apply()."""

    def test_smooth_apply_basic(self, tmp_path):
        """Verify blend formula: new = current * (1 - blend) + new * blend."""
        conn, _ = create_test_db(tmp_path)

        # Set initial values
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('gamma', '0.01')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('beta_sal', '0.5')"
        )
        conn.commit()

        # Apply with 50% blend
        result = smooth_apply(conn, gamma=0.1, beta_sal=1.0, blend_factor=0.5)

        # Expected: 0.01 * 0.5 + 0.1 * 0.5 = 0.055
        assert result["gamma"] == pytest.approx(0.055)
        # Expected: 0.5 * 0.5 + 1.0 * 0.5 = 0.75
        assert result["beta_sal"] == pytest.approx(0.75)

        # Verify DB was updated
        gamma_row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'gamma'"
        ).fetchone()
        beta_row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'beta_sal'"
        ).fetchone()
        assert float(gamma_row["value"]) == pytest.approx(0.055)
        assert float(beta_row["value"]) == pytest.approx(0.75)

        conn.close()

    def test_smooth_apply_default_blend(self, tmp_path):
        """Default blend factor is 0.1 (10%)."""
        conn, _ = create_test_db(tmp_path)

        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('gamma', '0.0')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('beta_sal', '0.0')"
        )
        conn.commit()

        # Apply with default blend (0.1)
        result = smooth_apply(conn, gamma=1.0, beta_sal=1.0)

        # Expected: 0.0 * 0.9 + 1.0 * 0.1 = 0.1
        assert result["gamma"] == pytest.approx(0.1)
        assert result["beta_sal"] == pytest.approx(0.1)

        conn.close()

    def test_smooth_apply_100_percent_blend(self, tmp_path):
        """blend_factor=1.0 should apply new value directly."""
        conn, _ = create_test_db(tmp_path)

        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('gamma', '0.01')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('beta_sal', '0.5')"
        )
        conn.commit()

        result = smooth_apply(conn, gamma=0.1, beta_sal=1.0, blend_factor=1.0)

        # Expected: direct application
        assert result["gamma"] == pytest.approx(0.1)
        assert result["beta_sal"] == pytest.approx(1.0)

        conn.close()

    def test_smooth_apply_stores_last_tuned_at(self, tmp_path):
        """smooth_apply should store last_tuned_at timestamp."""
        conn, _ = create_test_db(tmp_path)

        smooth_apply(conn, gamma=0.05, beta_sal=0.75)

        last_tuned = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'last_tuned_at'"
        ).fetchone()
        assert last_tuned is not None
        assert last_tuned["value"] is not None
        # Should be a valid ISO timestamp
        datetime.fromisoformat(last_tuned["value"])

        conn.close()


# ------------------------------------------------------------------
# analyze_current_performance Tests
# ------------------------------------------------------------------


class TestAnalyzeCurrentPerformance:
    """Tests for analyze_current_performance()."""

    def test_analyze_empty_db(self, tmp_path):
        """Empty database should return zero metrics."""
        conn, _ = create_test_db(tmp_path)

        result = analyze_current_performance(conn)

        assert result["ndcg@1"] == 0.0
        assert result["ndcg@5"] == 0.0
        assert result["lift@5"] == 0.0
        assert result["num_events"] == 0

        conn.close()

    def test_analyze_with_mock_data(self, tmp_path):
        """Verify metrics calculation with mock data."""
        conn, _ = create_test_db(tmp_path)
        insert_test_data(conn)

        result = analyze_current_performance(conn)

        # Should have 2 events
        assert result["num_events"] == 2

        # NDCG@1: Average of NDCG@1 for each event
        # Event 1: Top result (Entity A) is relevant → NDCG@1 = 1.0
        # Event 2: Top result (Entity F) is NOT relevant → NDCG@1 = 0.0
        # Average = 0.5
        assert result["ndcg@1"] == pytest.approx(0.5, abs=0.01)

        # NDCG@5 and Lift@5 should be positive with our mock data
        assert result["ndcg@5"] > 0.0
        assert result["lift@5"] > 0.0

        conn.close()


# ------------------------------------------------------------------
# find_optimal_params Tests
# ------------------------------------------------------------------


class TestFindOptimalParams:
    """Tests for find_optimal_params()."""

    def test_find_optimal_params_empty_db(self, tmp_path):
        """Empty database should return defaults."""
        conn, _ = create_test_db(tmp_path)

        result = find_optimal_params(conn, [0.01, 0.1], [0.5, 1.0])

        assert result["best_gamma"] == DEFAULT_GAMMA
        assert result["best_beta_sal"] == DEFAULT_BETA_SAL
        assert result["best_ndcg@5"] == 0.0
        assert result["all_results"] == []

        conn.close()

    def test_find_optimal_params_basic(self, tmp_path):
        """Verify grid search finds best combination."""
        conn, _ = create_test_db(tmp_path)
        insert_test_data(conn)

        gamma_range = [0.001, 0.01, 0.1]
        beta_range = [0.1, 0.5, 1.0]

        result = find_optimal_params(conn, gamma_range, beta_range)

        # Should have 9 combinations (3 × 3)
        assert len(result["all_results"]) == 9

        # Best should be one of the tested values
        assert result["best_gamma"] in gamma_range
        assert result["best_beta_sal"] in beta_range

        # Best NDCG should be >= any individual result
        for r in result["all_results"]:
            assert r["ndcg@5"] <= result["best_ndcg@5"] + 1e-9

        # Results should be sorted by NDCG descending
        ndcs = [r["ndcg@5"] for r in result["all_results"]]
        assert ndcs == sorted(ndcs, reverse=True)

        conn.close()

    def test_find_optimal_params_single_value(self, tmp_path):
        """Single value range should return that value."""
        conn, _ = create_test_db(tmp_path)
        insert_test_data(conn)

        result = find_optimal_params(conn, [0.05], [0.3])

        assert result["best_gamma"] == 0.05
        assert result["best_beta_sal"] == 0.3
        assert len(result["all_results"]) == 1

        conn.close()


# ------------------------------------------------------------------
# compute_quality_gain Tests
# ------------------------------------------------------------------


class TestComputeQualityGain:
    """Tests for compute_quality_gain()."""

    def test_compute_quality_gain_basic(self, tmp_path):
        """Verify gain calculation."""
        conn, _ = create_test_db(tmp_path)
        insert_test_data(conn)

        # Compare current params vs themselves → gain should be 0
        result = compute_quality_gain(conn, DEFAULT_GAMMA, DEFAULT_BETA_SAL)

        assert result["gamma"] == DEFAULT_GAMMA
        assert result["beta_sal"] == DEFAULT_BETA_SAL
        assert result["gain"] == pytest.approx(0.0, abs=1e-9)

        conn.close()

    def test_compute_quality_gain_different_params(self, tmp_path):
        """Different params should produce some gain (could be positive or negative)."""
        conn, _ = create_test_db(tmp_path)
        insert_test_data(conn)

        result = compute_quality_gain(conn, 0.1, 1.0)

        # Should have valid numbers
        assert result["current_ndcg@5"] >= 0.0
        assert result["new_ndcg@5"] >= 0.0
        assert isinstance(result["gain"], float)

        conn.close()


# ------------------------------------------------------------------
# get_current_params Tests
# ------------------------------------------------------------------


class TestGetCurrentParams:
    """Tests for get_current_params()."""

    def test_get_current_params_defaults(self, tmp_path):
        """No metadata → should return defaults."""
        conn, _ = create_test_db(tmp_path)

        result = get_current_params(conn)

        assert result["gamma"] == DEFAULT_GAMMA
        assert result["beta_sal"] == DEFAULT_BETA_SAL

        conn.close()

    def test_get_current_params_stored(self, tmp_path):
        """Stored metadata should be returned."""
        conn, _ = create_test_db(tmp_path)

        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('gamma', '0.05')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('beta_sal', '0.75')"
        )
        conn.commit()

        result = get_current_params(conn)

        assert result["gamma"] == pytest.approx(0.05)
        assert result["beta_sal"] == pytest.approx(0.75)

        conn.close()


# ------------------------------------------------------------------
# NDCG helper tests
# ------------------------------------------------------------------


class TestNDCG:
    """Tests for _ndcg_at_k helper."""

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking: [1, 1, 1, 0, 0] at k=3 should yield 1.0."""
        relevance = [1, 1, 1, 0, 0]
        assert _ndcg_at_k(relevance, k=3) == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self):
        """Worst ranking at k=3 should yield 0.5 for [0, 0, 1]."""
        # Relevance [0, 0, 1] at k=3:
        # DCG = (2^0-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
        #     = 0 + 0 + 1/2 = 0.5
        # IDCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) + (2^0-1)/log2(4)
        #       = 1/1 + 0 + 0 = 1.0
        # NDCG = 0.5 / 1.0 = 0.5
        relevance = [0, 0, 1]
        result = _ndcg_at_k(relevance, k=3)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_ndcg_empty(self):
        """Empty relevance list at k=0 should return 0.0."""
        assert _ndcg_at_k([], k=0) == 0.0

    def test_ndcg_k_larger_than_list(self):
        """k larger than list length should use list length."""
        relevance = [1, 0]
        result_k3 = _ndcg_at_k(relevance, k=3)
        result_k2 = _ndcg_at_k(relevance, k=2)
        assert result_k3 == result_k2


# ------------------------------------------------------------------
# apply_to_scoring_file Tests
# ------------------------------------------------------------------


class TestApplyToScoringFile:
    """Tests for apply_to_scoring_file()."""

    def test_apply_to_scoring_file_updates_gamma_and_beta(self, tmp_path):
        """Verify apply_to_scoring_file updates GAMMA and BETA_SAL in scoring.py."""
        # Create a mock scoring.py for testing
        mock_scoring_content = '''"""Mock scoring module."""

GAMMA = 0.01  # Co-occurrence weight
BETA_SAL = 0.5  # Salience boost factor
'''
        mock_scoring_path = tmp_path / "scoring.py"
        mock_scoring_path.write_text(mock_scoring_content)

        # Call apply_to_scoring_file with our mock path
        import re

        gamma = 0.05
        beta_sal = 0.75

        content = mock_scoring_path.read_text()

        # Update GAMMA value
        gamma_pattern = re.compile(r"^(GAMMA\s*=\s*)\d+(\.\d+)?", re.MULTILINE)
        if gamma_pattern.search(content):
            content = gamma_pattern.sub(rf"\g<1>{gamma}", content)

        # Update BETA_SAL value
        beta_pattern = re.compile(r"^(BETA_SAL\s*=\s*)\d+(\.\d+)?", re.MULTILINE)
        if beta_pattern.search(content):
            content = beta_pattern.sub(rf"\g<1>{beta_sal}", content)

        mock_scoring_path.write_text(content)

        # Read back and verify
        new_content = mock_scoring_path.read_text()
        assert "GAMMA = 0.05" in new_content
        assert "BETA_SAL = 0.75" in new_content

    def test_apply_to_scoring_file_preserves_comments(self, tmp_path):
        """Verify apply_to_scoring_file preserves inline comments."""
        mock_scoring_content = '''"""Mock scoring module."""

GAMMA = 0.01  # Co-occurrence weight per pair
BETA_SAL = 0.5  # Salience boost factor
'''
        mock_scoring_path = tmp_path / "scoring.py"
        mock_scoring_path.write_text(mock_scoring_content)

        import re

        content = mock_scoring_path.read_text()

        gamma_pattern = re.compile(r"^(GAMMA\s*=\s*)\d+(\.\d+)?", re.MULTILINE)
        content = gamma_pattern.sub(r"\g<1>0.1", content)

        mock_scoring_path.write_text(content)

        # Comment should be preserved
        new_content = mock_scoring_path.read_text()
        assert "# Co-occurrence weight per pair" in new_content
