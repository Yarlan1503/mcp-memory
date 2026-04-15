"""Retry utility for SQLite write operations under concurrent access."""

import functools
import random
import sqlite3
import time
import logging

logger = logging.getLogger(__name__)

# Configurable constants
MAX_RETRIES = 5
BASE_DELAY = 0.1  # seconds
MAX_DELAY = 2.0  # seconds


def retry_on_locked(func):
    """Decorator that retries SQLite write operations on 'database is locked' errors.

    Uses exponential backoff with jitter. Only retries on OperationalError
    with 'locked' in the message. All other exceptions propagate immediately.

    Logs each retry at DEBUG level with attempt number and delay.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if attempt == MAX_RETRIES:
                    raise
                last_exc = exc
                # Rollback any pending transaction before retrying
                try:
                    args[0].db.rollback()
                except Exception:
                    pass  # No transaction active or no .db attribute
                delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                logger.debug(
                    "Database locked on %s (attempt %d/%d), retrying in %.3fs: %s",
                    func.__qualname__,
                    attempt + 1,
                    MAX_RETRIES,
                    total_delay,
                    exc,
                )
                time.sleep(total_delay)
        # Should not reach here, but just in case
        raise last_exc  # pragma: no cover

    return wrapper
