"""Test to reproduce concurrency bug in MemoryStore.

Bug: MemoryStore opens a single sqlite3 connection with check_same_thread=False.
When multiple threads call add_observations() or upsert_entity() on the same
instance simultaneously, SQLite raises:
    "cannot start a transaction within a transaction"

This happens because the C-level sqlite3 connection does not support nested
transactions, and multiple Python threads share the same connection.
"""

import sqlite3
import threading
import pytest
from mcp_memory.storage import MemoryStore


# Number of threads to spawn. Must be high enough to trigger races.
N_THREADS = 20
# Iterations each thread performs.
ITERATIONS = 5


def test_concurrency_bug_same_entity(tmp_path):
    """Reproduce concurrency bug by hammering the SAME entity from N threads."""
    db_path = str(tmp_path / "concurrency_test.db")
    store = MemoryStore(db_path)
    store.init_db()

    exceptions = []
    exc_lock = threading.Lock()
    barrier = threading.Barrier(N_THREADS)

    def worker(tid: int):
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            return

        for i in range(ITERATIONS):
            try:
                # All threads fight over the same entity row → maximum contention
                eid = store.upsert_entity("SharedEntity", "TestType")
                store.add_observations(eid, [f"obs-{tid}-{i}"])
            except Exception as exc:
                with exc_lock:
                    exceptions.append((tid, i, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    store.close()

    # Filter for the specific concurrency errors we expect
    concurrency_errors = [
        e for e in exceptions
        if "transaction" in e[3].lower() or "locked" in e[3].lower()
    ]

    # The test should FAIL when the bug is present.
    # If we see any "cannot start a transaction within a transaction" or similar,
    # that proves the bug exists.
    assert len(concurrency_errors) == 0, (
        f"Detected {len(concurrency_errors)} concurrency error(s):\n"
        + "\n".join(f"  thread={tid} iter={i} {cls}: {msg}" for tid, i, cls, msg in concurrency_errors[:10])
    )


def test_concurrency_bug_shared_entities(tmp_path):
    """Reproduce concurrency bug using a small pool of shared entities."""
    db_path = str(tmp_path / "concurrency_test2.db")
    store = MemoryStore(db_path)
    store.init_db()

    exceptions = []
    exc_lock = threading.Lock()
    barrier = threading.Barrier(N_THREADS)

    def worker(tid: int):
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            return

        for i in range(ITERATIONS):
            try:
                # Small entity pool → high collision probability
                entity_name = f"SharedEntity-{i % 4}"
                eid = store.upsert_entity(entity_name, "TestType")
                store.add_observations(eid, [f"obs-{tid}-{i}"])
            except Exception as exc:
                with exc_lock:
                    exceptions.append((tid, i, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    store.close()

    concurrency_errors = [
        e for e in exceptions
        if "transaction" in e[3].lower() or "locked" in e[3].lower()
    ]

    assert len(concurrency_errors) == 0, (
        f"Detected {len(concurrency_errors)} concurrency error(s):\n"
        + "\n".join(f"  thread={tid} iter={i} {cls}: {msg}" for tid, i, cls, msg in concurrency_errors[:10])
    )
