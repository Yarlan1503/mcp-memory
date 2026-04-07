"""Tests for consolidation_report tool and get_consolidation_data storage method."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_memory.storage import MemoryStore
from mcp_memory.server import consolidation_report
from mcp_memory.entity_splitter import _get_threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """MemoryStore with schema initialized."""
    db_path = str(tmp_path / "test_consolidation.db")
    s = MemoryStore(db_path)
    s.init_db()
    yield s
    s.close()


def _set_last_access(store: MemoryStore, entity_id: int, days_ago: int) -> None:
    """Set entity_access.last_access to N days ago via direct SQL."""
    store.db.execute(
        "UPDATE entity_access SET last_access = datetime('now', ? || ' days') "
        "WHERE entity_id = ?",
        (str(-days_ago), entity_id),
    )
    # If no row exists, insert one
    if store.db.total_changes == 0:
        store.db.execute(
            "INSERT INTO entity_access (entity_id, access_count, last_access) "
            "VALUES (?, 1, datetime('now', ? || ' days'))",
            (entity_id, str(-days_ago)),
        )
    store.db.commit()


def _insert_flagged_obs(store: MemoryStore, entity_id: int, content: str) -> int:
    """Insert an observation with similarity_flag=1. Returns obs id."""
    cursor = store.db.execute(
        "INSERT INTO observations (entity_id, content, similarity_flag) "
        "VALUES (?, ?, 1)",
        (entity_id, content),
    )
    store.db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Test: Empty DB
# ---------------------------------------------------------------------------


class TestConsolidationReportEmpty:
    """Report on empty database returns all counters at 0."""

    def test_consolidation_report_empty_db(self, store):
        """All counts should be 0 on empty DB."""
        # Patch the global store in server.py
        import mcp_memory.server as srv

        original_store = srv.store
        srv.store = store
        try:
            report = consolidation_report(stale_days=90.0)

            assert "summary" in report
            assert "error" not in report
            s = report["summary"]
            assert s["total_entities"] == 0
            assert s["total_observations"] == 0
            assert s["split_candidates_count"] == 0
            assert s["flagged_observations_count"] == 0
            assert s["stale_entities_count"] == 0
            assert s["large_entities_count"] == 0

            assert report["split_candidates"] == []
            assert report["flagged_observations"] == {}
            assert report["stale_entities"] == []
            assert report["large_entities"] == []
        finally:
            srv.store = original_store


# ---------------------------------------------------------------------------
# Test: Split candidate detection
# ---------------------------------------------------------------------------


class TestConsolidationReportSplitCandidate:
    """Entity exceeding threshold with topic diversity appears in split_candidates."""

    def test_consolidation_report_split_candidate(self, store):
        """Entity with diverse observations exceeding threshold is a split candidate."""
        import mcp_memory.server as srv

        # Create entity with enough diverse observations for Generic type (threshold=20)
        eid = store.upsert_entity("BigEntity", "Generic")
        diverse_topics = (
            [
                f"Topic alpha discussion about machine learning algorithms {i}"
                for i in range(10)
            ]
            + [
                f"Topic beta discussion about cooking recipes and food {i}"
                for i in range(10)
            ]
            + [
                f"Topic gamma discussion about astronomy and space {i}"
                for i in range(5)
            ]
        )
        store.add_observations(eid, diverse_topics)

        original_store = srv.store
        srv.store = store
        try:
            report = consolidation_report(stale_days=90.0)

            assert "error" not in report
            sc = report["split_candidates"]
            assert len(sc) > 0

            names = [c["entity_name"] for c in sc]
            assert "BigEntity" in names

            candidate = next(c for c in sc if c["entity_name"] == "BigEntity")
            assert candidate["observation_count"] == 25
            assert candidate["threshold"] == 20
            assert candidate["entity_type"] == "Generic"
            assert "recommendation" in candidate
            assert "Split recommended" in candidate["recommendation"]
        finally:
            srv.store = original_store


# ---------------------------------------------------------------------------
# Test: Flagged observations
# ---------------------------------------------------------------------------


class TestConsolidationReportFlaggedObs:
    """Observations with similarity_flag=1 appear in flagged_observations."""

    def test_consolidation_report_flagged_obs(self, store):
        """Flagged observations appear with content_preview and content_length."""
        import mcp_memory.server as srv

        eid = store.upsert_entity("FlaggedEntity", "Generic")
        # Add some normal observations
        store.add_observations(
            eid, ["normal observation one", "normal observation two"]
        )

        # Add flagged observations directly
        _insert_flagged_obs(
            store,
            eid,
            "This is a very long observation that should be truncated in the content preview field because it exceeds the maximum length",
        )
        _insert_flagged_obs(store, eid, "Short flagged obs")

        original_store = srv.store
        srv.store = store
        try:
            report = consolidation_report(stale_days=90.0)

            assert "error" not in report
            assert report["summary"]["flagged_observations_count"] == 2

            flagged = report["flagged_observations"]
            assert "FlaggedEntity" in flagged
            obs_list = flagged["FlaggedEntity"]
            assert len(obs_list) == 2

            # Check long content is truncated
            long_obs = next(o for o in obs_list if o["content_length"] > 80)
            assert long_obs["content_preview"].endswith("...")
            assert len(long_obs["content_preview"]) > 80
            assert long_obs["content_length"] == len(
                "This is a very long observation that should be truncated in the content preview field because it exceeds the maximum length"
            )

            # Check short content is NOT truncated
            short_obs = next(o for o in obs_list if o["content_length"] <= 80)
            assert not short_obs["content_preview"].endswith("...")
            assert short_obs["content_preview"] == "Short flagged obs"
        finally:
            srv.store = original_store


# ---------------------------------------------------------------------------
# Test: Stale entities
# ---------------------------------------------------------------------------


class TestConsolidationReportStale:
    """Entities not accessed in N days appear in stale_entities."""

    def test_consolidation_report_stale_entity(self, store):
        """Entity without recent access appears as stale."""
        import mcp_memory.server as srv

        eid = store.upsert_entity("StaleEntity", "Generic")
        store.add_observations(eid, ["observation one"])
        store.init_access(eid)

        # Set last_access to 100 days ago, access_count = 1
        _set_last_access(store, eid, days_ago=100)

        original_store = srv.store
        srv.store = store
        try:
            report = consolidation_report(stale_days=90.0)

            assert "error" not in report
            stale = report["stale_entities"]
            assert len(stale) > 0

            names = [s["entity_name"] for s in stale]
            assert "StaleEntity" in names

            entity = next(s for s in stale if s["entity_name"] == "StaleEntity")
            assert entity["access_count"] == 1
            assert entity["observation_count"] == 1
            assert entity["days_since_access"] is not None
            assert entity["days_since_access"] >= 99  # Allow 1-day margin
            assert "consider archiving" in entity["recommendation"]
        finally:
            srv.store = original_store

    def test_stale_days_parameter(self, store):
        """Different stale_days values detect different entity sets."""
        import mcp_memory.server as srv

        # Entity A: last accessed 50 days ago
        eid_a = store.upsert_entity("RecentEntity", "Generic")
        store.add_observations(eid_a, ["obs a"])
        store.init_access(eid_a)
        _set_last_access(store, eid_a, days_ago=50)

        # Entity B: last accessed 100 days ago
        eid_b = store.upsert_entity("OldEntity", "Generic")
        store.add_observations(eid_b, ["obs b"])
        store.init_access(eid_b)
        _set_last_access(store, eid_b, days_ago=100)

        # Entity C: last accessed 200 days ago
        eid_c = store.upsert_entity("AncientEntity", "Generic")
        store.add_observations(eid_c, ["obs c"])
        store.init_access(eid_c)
        _set_last_access(store, eid_c, days_ago=200)

        original_store = srv.store
        srv.store = store
        try:
            # With stale_days=30: all three are stale (> 30 days)
            report_30 = consolidation_report(stale_days=30.0)
            assert "error" not in report_30
            names_30 = {s["entity_name"] for s in report_30["stale_entities"]}
            assert "RecentEntity" in names_30
            assert "OldEntity" in names_30
            assert "AncientEntity" in names_30

            # With stale_days=180: only C is stale (> 180 days)
            report_180 = consolidation_report(stale_days=180.0)
            assert "error" not in report_180
            names_180 = {s["entity_name"] for s in report_180["stale_entities"]}
            assert "RecentEntity" not in names_180
            assert "OldEntity" not in names_180
            assert "AncientEntity" in names_180
        finally:
            srv.store = original_store


# ---------------------------------------------------------------------------
# Test: Large entities (exceed threshold but not split candidates)
# ---------------------------------------------------------------------------


class TestConsolidationReportLargeNotSplit:
    """Entity exceeding threshold without sufficient topic diversity appears
    in large_entities but not split_candidates.

    Note: With the current entity_splitter algorithm, any entity exceeding
    its observation threshold will be a split candidate (because count_score
    > 1.0 always when obs > threshold). This test verifies that split
    candidates are correctly excluded from large_entities, even when the
    large_entities list may be empty in practice.
    """

    def test_consolidation_report_large_not_split(self, store):
        """Entity exceeding threshold that IS a split candidate does NOT appear in large_entities."""
        import mcp_memory.server as srv

        # Create entity that exceeds Generic threshold (20) with diverse topics
        eid = store.upsert_entity("LargeButDiverse", "Generic")
        diverse_obs = [
            f"Machine learning topic about neural networks and deep learning {i}"
            for i in range(25)
        ]
        store.add_observations(eid, diverse_obs)

        original_store = srv.store
        srv.store = store
        try:
            report = consolidation_report(stale_days=90.0)

            assert "error" not in report

            # Entity should appear as split candidate (diverse enough)
            sc_names = [c["entity_name"] for c in report["split_candidates"]]
            assert "LargeButDiverse" in sc_names

            # Entity should NOT appear in large_entities (correctly excluded)
            le_names = [e["entity_name"] for e in report["large_entities"]]
            assert "LargeButDiverse" not in le_names
        finally:
            srv.store = original_store


# ---------------------------------------------------------------------------
# Test: Report is read-only
# ---------------------------------------------------------------------------


class TestConsolidationReportReadOnly:
    """Running consolidation_report does not modify the knowledge graph."""

    def test_consolidation_report_is_read_only(self, store):
        """After running the report, the KG state is unchanged."""
        import mcp_memory.server as srv

        # Set up some data
        eid = store.upsert_entity("ReadOnlyTest", "Generic")
        store.add_observations(eid, ["observation one", "observation two"])
        store.init_access(eid)

        # Capture state before report
        entities_before = store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[
            0
        ]
        obs_before = store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        relations_before = store.db.execute(
            "SELECT COUNT(*) FROM relations"
        ).fetchone()[0]

        original_store = srv.store
        srv.store = store
        try:
            # Run report
            consolidation_report(stale_days=90.0)
        finally:
            srv.store = original_store

        # Verify no changes
        entities_after = store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        obs_after = store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        relations_after = store.db.execute("SELECT COUNT(*) FROM relations").fetchone()[
            0
        ]

        assert entities_before == entities_after
        assert obs_before == obs_after
        assert relations_before == relations_after


# ---------------------------------------------------------------------------
# Test: get_consolidation_data method directly
# ---------------------------------------------------------------------------


class TestGetConsolidationData:
    """Direct tests for MemoryStore.get_consolidation_data()."""

    def test_returns_all_keys(self, store):
        """get_consolidation_data returns all expected keys."""
        data = store.get_consolidation_data(stale_days=90.0)
        assert "total_entities" in data
        assert "total_observations" in data
        assert "flagged_observations" in data
        assert "stale_entities" in data
        assert "entity_sizes" in data

    def test_entity_sizes_counts(self, store):
        """entity_sizes correctly counts observations per entity."""
        e1 = store.upsert_entity("SizeTest1", "Generic")
        e2 = store.upsert_entity("SizeTest2", "Generic")
        store.add_observations(e1, ["a", "b", "c"])
        store.add_observations(e2, ["x", "y"])

        data = store.get_consolidation_data(stale_days=90.0)
        assert data["entity_sizes"]["SizeTest1"] == 3
        assert data["entity_sizes"]["SizeTest2"] == 2

    def test_flagged_observations_query(self, store):
        """Only similarity_flag=1 observations are returned."""
        eid = store.upsert_entity("FlaggedTest", "Generic")
        store.add_observations(eid, ["normal obs"])  # flag=0
        _insert_flagged_obs(store, eid, "flagged obs")  # flag=1

        data = store.get_consolidation_data(stale_days=90.0)
        flagged = data["flagged_observations"]
        assert "FlaggedTest" in flagged
        assert len(flagged["FlaggedTest"]) == 1
        assert flagged["FlaggedTest"][0]["content"] == "flagged obs"
        assert flagged["FlaggedTest"][0]["similarity_flag"] == 1

    def test_stale_entity_no_access_record(self, store):
        """Entity with observations but no entity_access record is considered stale."""
        eid = store.upsert_entity("NeverAccessed", "Generic")
        store.add_observations(eid, ["some observation"])
        # Do NOT call init_access — no entity_access row exists

        data = store.get_consolidation_data(stale_days=90.0)
        stale_names = [s["entity_name"] for s in data["stale_entities"]]
        assert "NeverAccessed" in stale_names

        entity = next(
            s for s in data["stale_entities"] if s["entity_name"] == "NeverAccessed"
        )
        assert entity["access_count"] == 0
        assert entity["last_access"] is None
