"""Tests for search result shadow logging."""

import logging

from mcp_memory.tools import search as search_tools


class FakeSearchLoggingStore:
    def __init__(self):
        self.logged_results = None
        self.updated_events = []
        self.recorded_accesses = []
        self.recorded_co_occurrences = []

    def get_entities_batch(self, entity_ids):
        return {eid: {"name": f"Entity {eid}"} for eid in entity_ids}

    def log_search_results(self, event_id, results):
        self.logged_results = (event_id, results)

    def update_search_event_completion(
        self, event_id, num_results, duration_ms, engine_used
    ):
        self.updated_events.append((event_id, num_results, duration_ms, engine_used))

    def record_access(self, entity_id):
        self.recorded_accesses.append(entity_id)

    def record_co_occurrences(self, entity_ids):
        self.recorded_co_occurrences.append(entity_ids)


def test_log_shadow_and_track_preserves_none_cosine_for_fts_only_result(
    monkeypatch, caplog
):
    """FTS-only hybrid rows have distance=None and should not break logging."""
    import mcp_memory.server as server

    fake_store = FakeSearchLoggingStore()
    monkeypatch.setattr(server, "store", fake_store)

    ranked = [
        {
            "entity_id": 1,
            "distance": None,
            "limbic_score": 0.7,
            "importance": 0.2,
            "temporal_factor": 1.0,
            "cooc_boost": 0.0,
        }
    ]

    with caplog.at_level(logging.WARNING):
        search_tools._log_shadow_and_track(
            event_id=123,
            ranked=ranked,
            treatment=1,
            baseline_ranked=[],
            top_k_ids=[1],
            shadow_start_time=0.0,
        )

    assert "Shadow mode result logging failed" not in caplog.text
    assert fake_store.logged_results is not None
    _, logged_results = fake_store.logged_results
    assert logged_results[0]["cosine_sim"] is None


def test_cosine_sim_from_distance_converts_numeric_distance():
    assert search_tools._cosine_sim_from_distance(0.25) == 0.75
    assert search_tools._cosine_sim_from_distance(1.25) == 0.0
