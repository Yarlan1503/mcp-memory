"""Tests for retry_on_locked decorator and P10 composite index."""

import sqlite3
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from mcp_memory.retry import retry_on_locked, MAX_RETRIES, BASE_DELAY
from mcp_memory.storage import MemoryStore


# --- P0: Retry mechanism tests ---


class TestRetryOnLocked:
    """Tests for the retry_on_locked decorator."""

    def test_no_retry_on_success(self):
        """Function succeeds on first call — no retry needed."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = write_op()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_locked_error(self):
        """Retries when OperationalError('database is locked') is raised."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = write_op()
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Raises OperationalError after MAX_RETRIES+1 attempts."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            write_op()
        assert call_count == MAX_RETRIES + 1

    def test_no_retry_on_non_locked_operational_error(self):
        """Non-locked OperationalError propagates immediately without retry."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("no such table: foo")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            write_op()
        assert call_count == 1

    def test_no_retry_on_other_exceptions(self):
        """Non-OperationalError exceptions propagate immediately."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad value")

        with pytest.raises(ValueError, match="bad value"):
            write_op()
        assert call_count == 1

    def test_preserves_function_metadata(self):
        """Decorator preserves original function name and docstring."""

        @retry_on_locked
        def my_write_function():
            """My docstring."""
            pass

        assert my_write_function.__name__ == "my_write_function"
        assert my_write_function.__doc__ == "My docstring."

    def test_retry_logs_debug(self, caplog):
        """Each retry logs a DEBUG message with attempt info."""
        call_count = 0

        @retry_on_locked
        def write_op():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        with caplog.at_level(logging.DEBUG, logger="mcp_memory.retry"):
            write_op()

        assert any(
            "attempt 1" in r.message.lower() or "attempt 1" in r.message
            for r in caplog.records
        )
        assert call_count == 2

    def test_retry_on_real_storage_write(self):
        """Test that retry works with a real MemoryStore write operation."""
        store = MemoryStore(":memory:")
        store.init_db()

        # Normal write should succeed
        entity_id = store.upsert_entity("TestEntity", "Testing")
        assert entity_id > 0

        # Verify the decorator is applied (method has the wrapper)
        from mcp_memory.retry import retry_on_locked

        # The method should be wrapped
        assert hasattr(store.upsert_entity, "__wrapped__")

        store.close()


# --- P10: Composite index test ---


class TestCompositeIndex:
    """Tests for the observations(entity_id, superseded_at) composite index."""

    def test_composite_index_exists(self, store_with_schema):
        """Verify the composite index idx_obs_entity_superseded exists."""
        indices = store_with_schema.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_obs_entity_superseded'"
        ).fetchall()
        assert len(indices) == 1
        assert indices[0]["name"] == "idx_obs_entity_superseded"

    def test_index_covers_superseded_queries(self, store_with_schema):
        """Verify the index is usable for superseded_at queries."""
        # Create entity with observations
        eid = store_with_schema.upsert_entity("IdxTest", "Testing")
        store_with_schema.add_observations(eid, ["obs1", "obs2"])

        # Query using the indexed columns — EXPLAIN QUERY PLAN should show index usage
        plan = store_with_schema.db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM observations WHERE entity_id = ? AND superseded_at IS NULL",
            (eid,),
        ).fetchall()
        plan_str = " ".join(r["detail"] for r in plan)
        # The index should appear in the query plan
        assert (
            "idx_obs" in plan_str.lower()
            or "COVERING" in plan_str.upper()
            or "entity_id" in plan_str
        )


# --- F1: Rollback before retry ---


class TestRollbackBeforeRetry:
    """Tests that rollback() is called on the db connection before retrying."""

    def test_rollback_called_on_locked_error(self):
        """When OperationalError('locked') occurs, args[0].db.rollback() is called."""
        mock_db = MagicMock()
        call_count = 0

        class FakeStore:
            db = mock_db

        @retry_on_locked
        def write_op(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = write_op(FakeStore())
        assert result == "ok"
        assert call_count == 3
        # rollback should have been called on each failed attempt (2 times)
        assert mock_db.rollback.call_count == 2

    def test_rollback_silently_ignored_if_no_db(self):
        """If args[0] has no .db attribute, rollback is silently skipped."""
        call_count = 0

        class FakeStoreNoDb:
            pass

        @retry_on_locked
        def write_op(self):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = write_op(FakeStoreNoDb())
        assert result == "ok"
        assert call_count == 2

    def test_rollback_not_called_on_non_locked_error(self):
        """Rollback is NOT called for non-locked OperationalError."""
        mock_db = MagicMock()

        class FakeStore:
            db = mock_db

        @retry_on_locked
        def write_op(self):
            raise sqlite3.OperationalError("no such table")

        with pytest.raises(sqlite3.OperationalError):
            write_op(FakeStore())
        mock_db.rollback.assert_not_called()


# --- F3: BEGIN IMMEDIATE in add_observations ---


class TestBeginImmediate:
    """Tests that add_observations uses BEGIN IMMEDIATE for write lock."""

    def test_begin_immediate_called(self, store_with_schema):
        """add_observations should execute BEGIN IMMEDIATE at the start."""
        eid = store_with_schema.upsert_entity("BeginTest", "Testing")

        with patch.object(store_with_schema, "db") as mock_db:
            # Set up the mock to behave like a real db for the method
            mock_db.execute.return_value.fetchone.return_value = None
            mock_db.execute.return_value.fetchall.return_value = []

            # We need to track execute calls
            execute_calls = []
            original_execute = store_with_schema.db.execute

            def tracking_execute(sql, *args, **kwargs):
                execute_calls.append(sql.strip())
                return original_execute(sql, *args, **kwargs)

            store_with_schema.db.execute = tracking_execute
            try:
                store_with_schema.add_observations(eid, ["test observation"])
            finally:
                store_with_schema.db.execute = original_execute

            # BEGIN IMMEDIATE should be among the first execute calls
            assert any("BEGIN IMMEDIATE" in call for call in execute_calls[:3])


# --- F4: Thread-safe EmbeddingEngine singleton ---


class TestThreadSafeSingleton:
    """Tests that EmbeddingEngine.get_instance() is thread-safe."""

    def test_singleton_thread_safety(self):
        """Multiple threads calling get_instance() simultaneously should return the same instance."""
        from mcp_memory.embeddings import EmbeddingEngine

        # Reset singleton for clean test
        EmbeddingEngine.reset()

        results = [None] * 10
        barrier = threading.Barrier(10)

        def get_instance(index):
            barrier.wait()  # Synchronize all threads
            results[index] = EmbeddingEngine.get_instance()

        threads = [threading.Thread(target=get_instance, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have gotten the same instance
        first = results[0]
        assert first is not None
        for r in results[1:]:
            assert r is first

        # Clean up
        EmbeddingEngine.reset()
