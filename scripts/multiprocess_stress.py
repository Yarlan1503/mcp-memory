"""Multi-process stress harness for MCP Memory tool-layer concurrency.

This harness intentionally uses separate OS processes, each with its own
``MemoryStore`` connection, to approximate multiple independent OpenCode/MCP
clients contending on the same SQLite database. Defaults are small enough for
local/CI smoke runs; increase via flags or ``MCP_MEMORY_STRESS_*`` env vars.

Examples:
    uv run python scripts/multiprocess_stress.py --db /tmp/mcp-stress.db
    MCP_MEMORY_STRESS_PROCESSES=8 MCP_MEMORY_STRESS_ITERATIONS=100 \
        uv run python scripts/multiprocess_stress.py --heavy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mcp_memory.embeddings import DIMENSION, serialize_f32
from mcp_memory.storage import MemoryStore


DEFAULT_PROCESSES = int(os.getenv("MCP_MEMORY_STRESS_PROCESSES", "3"))
DEFAULT_ITERATIONS = int(os.getenv("MCP_MEMORY_STRESS_ITERATIONS", "12"))
DEFAULT_TIMEOUT_MS = float(os.getenv("MCP_MEMORY_STRESS_TIMEOUT_MS", "5000"))
DEFAULT_QUEUE_TIMEOUT = float(os.getenv("MCP_MEMORY_STRESS_QUEUE_TIMEOUT", "60"))

ERROR_PATTERNS = {
    "Connection closed": ("connection closed", "closed database"),
    "Not connected": ("not connected",),
    "database is locked": ("database is locked",),
    "cannot start transaction": (
        "cannot start a transaction",
        "cannot start transaction",
    ),
    "Already borrowed": ("already borrowed",),
    "timeouts": ("timeout", "timed out"),
}


class DeterministicFakeEmbeddingEngine:
    """Fast deterministic embedding engine usable from every process."""

    available = True

    def prepare_entity_text(
        self,
        name: str,
        entity_type: str,
        observations: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        *,
        status: str = "activo",
    ) -> str:
        obs_text = " ".join(str(obs.get("content", "")) for obs in observations)
        rel_text = " ".join(str(rel) for rel in relations)
        return f"{name} {entity_type} {status} {obs_text} {rel_text}"

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.array([self._vector_for_text(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _vector_for_text(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm


@dataclass(frozen=True)
class HarnessConfig:
    db_path: str
    processes: int
    iterations: int
    timeout_ms: float
    seed_entities: int
    seed_observations: int
    queue_timeout: float


def _classify_error(message: str) -> str:
    lower = message.lower()
    for label, patterns in ERROR_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return label
    return "other"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _install_process_store(db_path: str) -> MemoryStore:
    """Point the already-imported tool layer at this process' DB connection."""
    import mcp_memory._helpers as helpers
    import mcp_memory.server as server
    import mcp_memory.tools.reflections as reflection_tools
    import mcp_memory.tools.search as search_tools

    try:
        server.store.close()
    except Exception:
        pass

    store = MemoryStore(db_path)
    store.init_db()
    fake_engine = DeterministicFakeEmbeddingEngine()

    server.store = store
    helpers._get_engine = lambda: fake_engine  # type: ignore[assignment]
    search_tools._get_engine = lambda: fake_engine  # type: ignore[assignment]
    reflection_tools._get_engine = lambda: fake_engine  # type: ignore[assignment]
    return store


def _seed_database(db_path: str, seed_entities: int, seed_observations: int) -> None:
    store = MemoryStore(db_path)
    store.init_db()
    engine = DeterministicFakeEmbeddingEngine()
    try:
        for entity_index in range(seed_entities):
            name = f"StressSeed-{entity_index}"
            entity_id = store.upsert_entity(name, "StressSeed")
            store.add_observations(
                entity_id,
                [
                    f"seed observation {entity_index}-{obs_index} topic concurrency sqlite"
                    for obs_index in range(seed_observations)
                ],
                kind="seed",
            )
            obs_data = store.get_observations_with_ids(entity_id)
            text = engine.prepare_entity_text(
                name, "StressSeed", obs_data, store.get_relations_for_entity(entity_id)
            )
            store.store_embedding(entity_id, serialize_f32(engine.encode([text])[0]))
            store.init_access(entity_id)
    finally:
        store.close()


def _record_operation(
    result_queue: mp.Queue,
    process_id: int,
    op_name: str,
    timeout_ms: float,
    func,
) -> None:
    start = time.perf_counter()
    status = "ok"
    error_type = None
    error_message = None
    try:
        result = func()
        if isinstance(result, dict) and result.get("error"):
            status = "error"
            error_message = str(result["error"])
            error_type = _classify_error(error_message)
    except Exception as exc:  # noqa: BLE001 - stress harness records all failures
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        error_type = _classify_error(error_message)
        result_queue.put(
            {
                "kind": "traceback",
                "process_id": process_id,
                "op": op_name,
                "traceback": traceback.format_exc(),
            }
        )

    latency_ms = (time.perf_counter() - start) * 1000
    if latency_ms > timeout_ms and status == "ok":
        status = "error"
        error_type = "timeouts"
        error_message = f"operation exceeded timeout budget: {latency_ms:.1f}ms > {timeout_ms:.1f}ms"

    result_queue.put(
        {
            "kind": "operation",
            "process_id": process_id,
            "op": op_name,
            "status": status,
            "latency_ms": latency_ms,
            "error_type": error_type,
            "error_message": error_message,
        }
    )


def _worker_main(config: HarnessConfig, process_id: int, result_queue: mp.Queue) -> None:
    store = _install_process_store(config.db_path)
    rng = random.Random(process_id)
    try:
        from mcp_memory.tools.core import add_observations, create_entities
        from mcp_memory.tools.entity_mgmt import consolidation_report, find_split_candidates
        from mcp_memory.tools.reflections import add_reflection
        from mcp_memory.tools.search import open_nodes, search_semantic

        local_entity = f"StressClient-{process_id}"
        create_entities(
            [
                {
                    "name": local_entity,
                    "entityType": "StressClient",
                    "observations": [f"bootstrap from process {process_id}"],
                }
            ]
        )

        operations = (
            "add_observations",
            "add_reflection",
            "search_semantic",
            "consolidation_report",
            "find_split_candidates",
            "long_read_then_write",
        )

        for iteration in range(config.iterations):
            op_name = operations[(process_id + iteration) % len(operations)]
            shared_entity = f"StressSeed-{iteration % config.seed_entities}"

            if op_name == "add_observations":
                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    lambda: add_observations(
                        name=shared_entity,
                        observations=[
                            f"mp obs process={process_id} iteration={iteration} rand={rng.random()}"
                        ],
                        kind="stress-multiprocess",
                    ),
                )
            elif op_name == "add_reflection":
                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    lambda: add_reflection(
                        target_type="global",
                        target_id=None,
                        author="sofia" if iteration % 2 == 0 else "nolan",
                        content=f"stress reflection process={process_id} iteration={iteration}",
                        mood="insight",
                    ),
                )
            elif op_name == "search_semantic":
                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    lambda: search_semantic(
                        query=f"concurrency sqlite process {process_id} iteration {iteration}",
                        limit=5,
                    ),
                )
            elif op_name == "consolidation_report":
                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    lambda: consolidation_report(stale_days=1.0),
                )
            elif op_name == "find_split_candidates":
                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    find_split_candidates,
                )
            else:
                def long_read_then_write() -> dict[str, Any]:
                    opened = open_nodes([f"StressSeed-{i}" for i in range(config.seed_entities)])
                    written = add_observations(
                        name=local_entity,
                        observations=[f"post-long-read write {iteration}"],
                        kind="stress-long-read-write",
                    )
                    if isinstance(opened, dict) and opened.get("error"):
                        return opened
                    return written

                _record_operation(
                    result_queue,
                    process_id,
                    op_name,
                    config.timeout_ms,
                    long_read_then_write,
                )
    finally:
        store.close()
        result_queue.put({"kind": "worker_done", "process_id": process_id})


def run_harness(config: HarnessConfig) -> dict[str, Any]:
    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(config.db_path).exists():
        Path(config.db_path).unlink()
    _seed_database(config.db_path, config.seed_entities, config.seed_observations)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker_main, args=(config, process_id, result_queue))
        for process_id in range(config.processes)
    ]

    start = time.perf_counter()
    for process in processes:
        process.start()

    expected_done = config.processes
    done = 0
    operation_rows: list[dict[str, Any]] = []
    tracebacks: list[dict[str, Any]] = []
    deadline = time.monotonic() + config.queue_timeout
    while done < expected_done and time.monotonic() < deadline:
        try:
            item = result_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item.get("kind") == "worker_done":
            done += 1
        elif item.get("kind") == "operation":
            operation_rows.append(item)
        elif item.get("kind") == "traceback":
            tracebacks.append(item)

    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)

    while True:
        try:
            item = result_queue.get_nowait()
        except queue.Empty:
            break
        if item.get("kind") == "operation":
            operation_rows.append(item)
        elif item.get("kind") == "traceback":
            tracebacks.append(item)

    by_op: dict[str, dict[str, Any]] = {}
    latencies_by_op: dict[str, list[float]] = defaultdict(list)
    errors_by_type: Counter[str] = Counter()
    for row in operation_rows:
        op = row["op"]
        latencies_by_op[op].append(float(row["latency_ms"]))
        if row["status"] == "error":
            errors_by_type[str(row.get("error_type") or "other")] += 1

    for op, latencies in sorted(latencies_by_op.items()):
        by_op[op] = {
            "count": len(latencies),
            "errors": sum(
                1 for row in operation_rows if row["op"] == op and row["status"] == "error"
            ),
            "avg_ms": round(statistics.fmean(latencies), 2),
            "p95_ms": round(_percentile(latencies, 95) or 0.0, 2),
            "p99_ms": round(_percentile(latencies, 99) or 0.0, 2),
            "max_ms": round(max(latencies), 2),
        }

    error_summary = {label: errors_by_type.get(label, 0) for label in ERROR_PATTERNS}
    error_summary["other"] = errors_by_type.get("other", 0)

    return {
        "config": config.__dict__,
        "duration_s": round(time.perf_counter() - start, 2),
        "operations": len(operation_rows),
        "worker_done": done,
        "process_exitcodes": [process.exitcode for process in processes],
        "errors_by_type": error_summary,
        "latencies_by_op": by_op,
        "tracebacks": tracebacks[:5],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("MCP_MEMORY_STRESS_DB", "/tmp/mcp-memory-stress.db"))
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--timeout-ms", type=float, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--seed-entities", type=int, default=int(os.getenv("MCP_MEMORY_STRESS_SEED_ENTITIES", "8")))
    parser.add_argument("--seed-observations", type=int, default=int(os.getenv("MCP_MEMORY_STRESS_SEED_OBSERVATIONS", "6")))
    parser.add_argument("--queue-timeout", type=float, default=DEFAULT_QUEUE_TIMEOUT)
    parser.add_argument("--heavy", action="store_true", help="Use heavier local defaults unless explicit flags/env override them.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.heavy:
        args.processes = max(args.processes, 8)
        args.iterations = max(args.iterations, 100)
        args.seed_entities = max(args.seed_entities, 20)
        args.seed_observations = max(args.seed_observations, 15)

    config = HarnessConfig(
        db_path=args.db,
        processes=args.processes,
        iterations=args.iterations,
        timeout_ms=args.timeout_ms,
        seed_entities=args.seed_entities,
        seed_observations=args.seed_observations,
        queue_timeout=args.queue_timeout,
    )
    summary = run_harness(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    has_errors = any(count for count in summary["errors_by_type"].values())
    return 1 if has_errors or any(code for code in summary["process_exitcodes"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
