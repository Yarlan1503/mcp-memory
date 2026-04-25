"""Reproducible concurrency stress tests for the SQLite MemoryStore.

The defaults are intentionally modest for local/CI runs. Increase with:

    MCP_MEMORY_STRESS_THREADS=16 MCP_MEMORY_STRESS_ITERATIONS=25 pytest tests/test_concurrency_stress.py
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from mcp_memory.embeddings import DIMENSION
from mcp_memory.storage import MemoryStore


STRESS_THREADS = int(os.getenv("MCP_MEMORY_STRESS_THREADS", "8"))
STRESS_ITERATIONS = int(os.getenv("MCP_MEMORY_STRESS_ITERATIONS", "10"))
JOIN_TIMEOUT_SECONDS = float(os.getenv("MCP_MEMORY_STRESS_JOIN_TIMEOUT", "30"))

# These texts are explicitly forbidden in all stress test failures.
FORBIDDEN_ERROR_TEXTS = (
    "Already borrowed",
    "database is locked",
    "cannot start a transaction within a transaction",
    "Recursive use of cursors not allowed",
    "cannot commit",
    "cannot rollback",
)


@dataclass(frozen=True)
class ThreadFailure:
    thread_id: int
    iteration: int
    exc_type: str
    message: str


class DeterministicFakeEmbeddingEngine:
    """Thread-safe, deterministic embedding mock for semantic paths."""

    available = True

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = [self._vector_for_text(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _vector_for_text(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm


def _new_store(tmp_path, db_name: str) -> MemoryStore:
    store = MemoryStore(str(tmp_path / db_name))
    store.init_db()
    return store


def _run_concurrently(
    worker: Callable[[int, int], None],
    *,
    n_threads: int = STRESS_THREADS,
    iterations: int = STRESS_ITERATIONS,
) -> list[ThreadFailure]:
    failures: list[ThreadFailure] = []
    failures_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def thread_main(thread_id: int) -> None:
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError as exc:
            with failures_lock:
                failures.append(ThreadFailure(thread_id, -1, type(exc).__name__, str(exc)))
            return

        for iteration in range(iterations):
            try:
                worker(thread_id, iteration)
            except Exception as exc:  # noqa: BLE001 - stress test records all failures
                with failures_lock:
                    failures.append(
                        ThreadFailure(
                            thread_id,
                            iteration,
                            type(exc).__name__,
                            str(exc),
                        )
                    )

    threads = [
        threading.Thread(target=thread_main, args=(thread_id,), name=f"stress-{thread_id}")
        for thread_id in range(n_threads)
    ]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + JOIN_TIMEOUT_SECONDS
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)

    for thread_id, thread in enumerate(threads):
        if thread.is_alive():
            failures.append(
                ThreadFailure(
                    thread_id,
                    -1,
                    "JoinTimeout",
                    f"thread did not finish within {JOIN_TIMEOUT_SECONDS:.1f}s",
                )
            )

    return failures


def _assert_no_forbidden_or_unexpected_failures(failures: list[ThreadFailure]) -> None:
    combined = "\n".join(
        f"thread={f.thread_id} iter={f.iteration} {f.exc_type}: {f.message}"
        for f in failures
    )
    combined_lower = combined.lower()
    forbidden_hits = [text for text in FORBIDDEN_ERROR_TEXTS if text.lower() in combined_lower]
    assert forbidden_hits == [], (
        "Forbidden SQLite concurrency text(s) observed: "
        f"{forbidden_hits}\nFailures:\n{combined}"
    )
    assert failures == [], f"Unexpected thread failure(s):\n{combined}"


def test_concurrent_pure_writes_stress(tmp_path):
    """Many threads perform only write operations against shared entity pools."""
    store = _new_store(tmp_path, "pure_writes.db")

    try:
        with patch.object(store, "_get_embedding_engine", return_value=None):

            def worker(thread_id: int, iteration: int) -> None:
                shared_name = f"WritePool-{iteration % 4}"
                unique_name = f"WriteUnique-{thread_id}-{iteration}"
                shared_id = store.upsert_entity(shared_name, "StressWrite")
                unique_id = store.upsert_entity(unique_name, "StressWrite")
                store.add_observations(
                    shared_id,
                    [f"shared observation from {thread_id}-{iteration}"],
                    kind="stress",
                )
                store.add_observations(
                    unique_id,
                    [f"unique observation from {thread_id}-{iteration}"],
                    kind="stress",
                )

            failures = _run_concurrently(worker)
            _assert_no_forbidden_or_unexpected_failures(failures)

            entities = store.search_entities("WriteUnique-")
            assert len(entities) == STRESS_THREADS * STRESS_ITERATIONS
    finally:
        store.close()


def test_concurrent_reads_with_side_effects_stress(tmp_path):
    """Read-heavy workload also records access and co-occurrence side effects."""
    store = _new_store(tmp_path, "reads_with_side_effects.db")

    entity_ids: list[int] = []
    entity_names: list[str] = []
    try:
        for index in range(8):
            entity_id = store.upsert_entity(f"ReadSideEffect-{index}", "StressRead")
            store.add_observations(entity_id, [f"read side effect seed {index}"])
            entity_ids.append(entity_id)
            entity_names.append(f"ReadSideEffect-{index}")

        def worker(thread_id: int, iteration: int) -> None:
            first = (thread_id + iteration) % len(entity_names)
            second = (first + 1) % len(entity_names)
            entity = store.get_entity_by_name(entity_names[first])
            assert entity is not None
            store.get_observations(entity["id"])
            store.record_access(entity["id"])
            store.record_co_occurrences([entity["id"], entity_ids[second]])

        failures = _run_concurrently(worker)
        _assert_no_forbidden_or_unexpected_failures(failures)

        access_data = store.get_access_data(entity_ids)
        assert sum(data["access_count"] for data in access_data.values()) >= (
            STRESS_THREADS * STRESS_ITERATIONS
        )
    finally:
        store.close()


def test_concurrent_duplicate_semantic_heavy_stress(tmp_path):
    """Concurrent exact duplicate filtering plus semantic-dedup path with a fake engine."""
    store = _new_store(tmp_path, "duplicate_semantic_heavy.db")
    fake_engine = DeterministicFakeEmbeddingEngine()

    entity_pairs: list[tuple[int, int]] = []
    try:
        with patch.object(store, "_get_embedding_engine", return_value=None):
            for entity_index in range(4):
                entity_id = store.upsert_entity(f"SemanticHeavy-{entity_index}", "StressSemantic")
                store.add_observations(
                    entity_id,
                    [f"semantic seed {entity_index}-{obs_index}" for obs_index in range(8)],
                    kind="seed",
                )
                entity_pairs.append((entity_id, entity_index))

        with patch.object(store, "_get_embedding_engine", return_value=fake_engine):

            def worker(thread_id: int, iteration: int) -> None:
                entity_id, entity_index = entity_pairs[
                    (thread_id + iteration) % len(entity_pairs)
                ]
                duplicate = f"semantic seed {entity_index}-{iteration % 8}"
                store.add_observations(
                    entity_id,
                    [
                        duplicate,
                        f"semantic candidate thread={thread_id} iteration={iteration}",
                    ],
                    kind="semantic-stress",
                )

            failures = _run_concurrently(worker)
            _assert_no_forbidden_or_unexpected_failures(failures)

            inserted = sum(
                len(store.get_observations(entity_id)) for entity_id, _ in entity_pairs
            )
            assert inserted >= len(entity_pairs) * 8 + STRESS_THREADS * STRESS_ITERATIONS
    finally:
        store.close()


def test_concurrent_mixed_read_write_semantic_stress(tmp_path):
    """Mixed workload: writes, reads with side effects, semantic adds, and searches."""
    store = _new_store(tmp_path, "mixed.db")
    fake_engine = DeterministicFakeEmbeddingEngine()

    entity_ids: list[int] = []
    try:
        with patch.object(store, "_get_embedding_engine", return_value=None):
            for index in range(6):
                entity_id = store.upsert_entity(f"Mixed-{index}", "StressMixed")
                store.add_observations(entity_id, [f"mixed seed observation {index}"])
                entity_ids.append(entity_id)

        with patch.object(store, "_get_embedding_engine", return_value=fake_engine):

            def worker(thread_id: int, iteration: int) -> None:
                selector = (thread_id + iteration) % 4
                entity_id = entity_ids[(thread_id * 3 + iteration) % len(entity_ids)]

                if selector == 0:
                    new_id = store.upsert_entity(
                        f"Mixed-New-{thread_id}-{iteration}", "StressMixed"
                    )
                    store.add_observations(new_id, [f"mixed new obs {thread_id}-{iteration}"])
                elif selector == 1:
                    entity = store.get_entity_by_id(entity_id)
                    assert entity is not None
                    store.get_observations(entity_id)
                    store.record_access(entity_id)
                elif selector == 2:
                    store.add_observations(
                        entity_id,
                        [f"mixed semantic obs {thread_id}-{iteration}"],
                        kind="semantic-stress",
                    )
                else:
                    store.search_entities("Mixed")
                    store.record_co_occurrences(entity_ids[:3])

            failures = _run_concurrently(worker)
            _assert_no_forbidden_or_unexpected_failures(failures)

            assert store.search_entities("Mixed")
    finally:
        store.close()
