"""Bounded-concurrency helpers for expensive MCP tools.

This module keeps overload behavior local-first and explicit: heavyweight
maintenance tools may fail fast with a structured error instead of allowing
unbounded concurrent work to pile up in a single server process.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import logging
import os
from threading import BoundedSemaphore
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, os.getenv(name), default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.1f", name, os.getenv(name), default)
        return default


HEAVY_TOOL_MAX_CONCURRENCY = _env_int("MCP_MEMORY_HEAVY_TOOL_MAX_CONCURRENCY", 1)
HEAVY_TOOL_ACQUIRE_TIMEOUT_SECONDS = _env_float(
    "MCP_MEMORY_HEAVY_TOOL_ACQUIRE_TIMEOUT_SECONDS", 2.0
)

_heavy_tool_slots = BoundedSemaphore(HEAVY_TOOL_MAX_CONCURRENCY)


def bounded_heavy_tool(func: F) -> F:
    """Limit concurrent executions of heavyweight, local maintenance tools.

    The decorator returns a structured MCP payload on saturation instead of
    blocking indefinitely. Defaults are intentionally conservative because these
    tools can scan most of the graph and invoke embedding/topic logic.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        acquired = _heavy_tool_slots.acquire(
            timeout=HEAVY_TOOL_ACQUIRE_TIMEOUT_SECONDS
        )
        if not acquired:
            return {
                "error": "server_busy",
                "code": "heavy_tool_backpressure",
                "message": (
                    f"Too many heavy maintenance operations are running "
                    f"(limit={HEAVY_TOOL_MAX_CONCURRENCY}). Retry later."
                ),
                "retryable": True,
            }

        try:
            return func(*args, **kwargs)
        finally:
            _heavy_tool_slots.release()

    return cast(F, wrapper)
