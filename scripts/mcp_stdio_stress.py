"""Stress harness for MCP Memory through the real MCP stdio transport.

The older ``scripts/multiprocess_stress.py`` exercises the Python tool layer
directly. This harness is intentionally closer to production: every worker is
an OS process that starts its own MCP server subprocess over stdio using the
repository's real ``mcp-memory`` entrypoint, then invokes tools through a
FastMCP client.

Defaults are deliberately small and write to an isolated temporary HOME so a
smoke run does not touch the user's real memory database.

Examples:
    uv run python scripts/mcp_stdio_stress.py --processes 2 --iterations 6
    uv run python scripts/mcp_stdio_stress.py --processes 8 --iterations 100 --heavy
    uv run python scripts/mcp_stdio_stress.py --home /tmp/mcp-stdio-home
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


DEFAULT_PROCESSES = int(os.getenv("MCP_MEMORY_STDIO_STRESS_PROCESSES", "2"))
DEFAULT_ITERATIONS = int(os.getenv("MCP_MEMORY_STDIO_STRESS_ITERATIONS", "6"))
DEFAULT_TIMEOUT_MS = float(os.getenv("MCP_MEMORY_STDIO_STRESS_TIMEOUT_MS", "15000"))
DEFAULT_QUEUE_TIMEOUT = float(os.getenv("MCP_MEMORY_STDIO_STRESS_QUEUE_TIMEOUT", "90"))

ERROR_PATTERNS = {
    "Connection closed": ("connection closed", "closed database", "closed resource"),
    "Not connected": ("not connected",),
    "-32001": ("-32001",),
    "-32000": ("-32000",),
    "database locked": ("database is locked", "database locked", "sqlite_busy"),
    "transaction errors": (
        "cannot start a transaction",
        "cannot start transaction",
        "cannot commit",
        "cannot rollback",
        "no transaction is active",
        "within a transaction",
    ),
    "Already borrowed": ("already borrowed",),
    "timeouts": ("timeout", "timed out", "deadline"),
}


@dataclass(frozen=True)
class HarnessConfig:
    home: str
    processes: int
    iterations: int
    timeout_ms: float
    seed_entities: int
    seed_observations: int
    queue_timeout: float
    command: str = "uv"
    args: list[str] = field(default_factory=lambda: ["run", "mcp-memory"])
    cwd: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[1]))
    model_cache_source: str | None = None


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


def _server_env(config: HarnessConfig, process_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = config.home
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["MCP_MEMORY_STDIO_STRESS_CLIENT"] = "seed" if process_id is None else str(process_id)
    return env


def _transport(config: HarnessConfig, process_id: int | None = None) -> StdioTransport:
    return StdioTransport(
        command=config.command,
        args=config.args,
        env=_server_env(config, process_id),
        cwd=config.cwd,
        keep_alive=False,
    )


def _stringify_result(result: Any) -> str:
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        return repr(result)


def _result_error_message(result: Any) -> str | None:
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])
    if isinstance(result, list):
        text = "\n".join(str(getattr(item, "text", item)) for item in result)
        if '"error"' in text or " error" in text.lower():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("error"):
                    return str(parsed["error"])
            except Exception:
                return text
    return None


async def _call_tool(
    client: Client,
    name: str,
    arguments: dict[str, Any] | None,
    timeout_ms: float,
) -> Any:
    return await asyncio.wait_for(
        client.call_tool(
            name,
            arguments or {},
            raise_on_error=False,
            timeout=timeout_ms / 1000,
        ),
        timeout=timeout_ms / 1000,
    )


async def _seed_database(config: HarnessConfig) -> None:
    _prepare_home(config)
    async with Client(_transport(config), timeout=config.timeout_ms / 1000) as client:
        entities = []
        for entity_index in range(config.seed_entities):
            entities.append(
                {
                    "name": f"StdioStressSeed-{entity_index}",
                    "entityType": "StdioStressSeed",
                    "observations": [
                        f"seed observation {entity_index}-{obs_index} topic concurrency sqlite stdio"
                        for obs_index in range(config.seed_observations)
                    ],
                }
            )
        result = await _call_tool(client, "create_entities", {"entities": entities}, config.timeout_ms)
        error_message = _result_error_message(result)
        if error_message:
            raise RuntimeError(f"seed create_entities failed: {error_message}")


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
        result_error = _result_error_message(result)
        if result_error:
            status = "error"
            error_message = result_error
            error_type = _classify_error(result_error)
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


async def _record_operation_async(
    result_queue: mp.Queue,
    process_id: int,
    op_name: str,
    timeout_ms: float,
    coro_factory,
) -> None:
    start = time.perf_counter()
    status = "ok"
    error_type = None
    error_message = None
    try:
        result = await coro_factory()
        result_error = _result_error_message(result)
        if result_error:
            status = "error"
            error_message = result_error
            error_type = _classify_error(result_error)
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


async def _worker_async(config: HarnessConfig, process_id: int, result_queue: mp.Queue) -> None:
    rng = random.Random(process_id)
    operations = (
        "add_observations",
        "add_reflection",
        "search_semantic",
        "consolidation_report",
        "find_split_candidates",
        "long_read_then_write",
    )
    async with Client(_transport(config, process_id), timeout=config.timeout_ms / 1000) as client:
        local_entity = f"StdioStressClient-{process_id}"
        await _call_tool(
            client,
            "create_entities",
            {
                "entities": [
                    {
                        "name": local_entity,
                        "entityType": "StdioStressClient",
                        "observations": [f"bootstrap from stdio process {process_id}"],
                    }
                ]
            },
            config.timeout_ms,
        )

        for iteration in range(config.iterations):
            op_name = operations[(process_id + iteration) % len(operations)]
            shared_entity = f"StdioStressSeed-{iteration % config.seed_entities}"

            if op_name == "add_observations":
                async def add_observations_call(
                    shared_entity: str = shared_entity,
                    iteration: int = iteration,
                ) -> Any:
                    return await _call_tool(
                        client,
                        "add_observations",
                        {
                            "name": shared_entity,
                            "observations": [
                                f"stdio obs process={process_id} iteration={iteration} rand={rng.random()}"
                            ],
                            "kind": "stress-stdio",
                        },
                        config.timeout_ms,
                    )

                coro_factory = add_observations_call
            elif op_name == "add_reflection":
                async def add_reflection_call(iteration: int = iteration) -> Any:
                    return await _call_tool(
                        client,
                        "add_reflection",
                        {
                            "target_type": "global",
                            "target_id": None,
                            "author": "sofia" if iteration % 2 == 0 else "nolan",
                            "content": f"stdio stress reflection process={process_id} iteration={iteration}",
                            "mood": "insight",
                        },
                        config.timeout_ms,
                    )

                coro_factory = add_reflection_call
            elif op_name == "search_semantic":
                async def search_semantic_call(iteration: int = iteration) -> Any:
                    return await _call_tool(
                        client,
                        "search_semantic",
                        {"query": f"concurrency sqlite stdio {process_id} {iteration}", "limit": 5},
                        config.timeout_ms,
                    )

                coro_factory = search_semantic_call
            elif op_name == "consolidation_report":
                async def consolidation_report_call() -> Any:
                    return await _call_tool(
                        client,
                        "consolidation_report",
                        {"stale_days": 1.0},
                        config.timeout_ms,
                    )

                coro_factory = consolidation_report_call
            elif op_name == "find_split_candidates":
                async def find_split_candidates_call() -> Any:
                    return await _call_tool(client, "find_split_candidates", {}, config.timeout_ms)

                coro_factory = find_split_candidates_call
            else:
                async def long_read_then_write() -> Any:
                    opened = await _call_tool(
                        client,
                        "open_nodes",
                        {
                            "names": [
                                f"StdioStressSeed-{i}" for i in range(config.seed_entities)
                            ],
                            "kinds": None,
                            "include_superseded": False,
                        },
                        config.timeout_ms,
                    )
                    opened_error = _result_error_message(opened)
                    if opened_error:
                        return {"error": opened_error}
                    return await _call_tool(
                        client,
                        "add_observations",
                        {
                            "name": local_entity,
                            "observations": [f"post-long-read stdio write {iteration}"],
                            "kind": "stress-long-read-write",
                        },
                        config.timeout_ms,
                    )

                coro_factory = long_read_then_write

            await _record_operation_async(
                result_queue,
                process_id,
                op_name,
                config.timeout_ms,
                coro_factory,
            )


def _worker_main(config: HarnessConfig, process_id: int, result_queue: mp.Queue) -> None:
    try:
        asyncio.run(_worker_async(config, process_id, result_queue))
    finally:
        result_queue.put({"kind": "worker_done", "process_id": process_id})


def _prepare_home(config: HarnessConfig) -> None:
    """Create isolated HOME and optionally reuse the user's model cache.

    The server stores the database under ``~/.config`` and the embedding model
    under ``~/.cache``. For safety we isolate HOME, but symlink the read-only-ish
    model cache when available so smoke runs do not spend minutes downloading
    ONNX assets into a throwaway directory.
    """
    home = Path(config.home)
    home.mkdir(parents=True, exist_ok=True)
    if not config.model_cache_source:
        return
    source = Path(config.model_cache_source).expanduser()
    if not source.exists():
        return
    cache_parent = home / ".cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    target = cache_parent / "mcp-memory-v2"
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)


def run_harness(config: HarnessConfig) -> dict[str, Any]:
    asyncio.run(_seed_database(config))

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker_main, args=(config, process_id, result_queue))
        for process_id in range(config.processes)
    ]

    start = time.perf_counter()
    for process in processes:
        process.start()

    done = 0
    operation_rows: list[dict[str, Any]] = []
    tracebacks: list[dict[str, Any]] = []
    deadline = time.monotonic() + config.queue_timeout
    while done < config.processes and time.monotonic() < deadline:
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

    latencies_by_op: dict[str, list[float]] = defaultdict(list)
    errors_by_type: Counter[str] = Counter()
    for row in operation_rows:
        op = row["op"]
        latencies_by_op[op].append(float(row["latency_ms"]))
        if row["status"] == "error":
            errors_by_type[str(row.get("error_type") or "other")] += 1

    by_op: dict[str, dict[str, Any]] = {}
    for op, latencies in sorted(latencies_by_op.items()):
        by_op[op] = {
            "count": len(latencies),
            "errors": sum(1 for row in operation_rows if row["op"] == op and row["status"] == "error"),
            "avg_ms": round(statistics.fmean(latencies), 2),
            "p95_ms": round(_percentile(latencies, 95) or 0.0, 2),
            "p99_ms": round(_percentile(latencies, 99) or 0.0, 2),
            "max_ms": round(max(latencies), 2),
        }

    error_summary = {label: errors_by_type.get(label, 0) for label in ERROR_PATTERNS}
    error_summary["other"] = errors_by_type.get("other", 0)
    return {
        "config": {**config.__dict__, "args": list(config.args)},
        "db_path": str(Path(config.home) / ".config" / "opencode" / "mcp-memory" / "memory.db"),
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
    parser.add_argument("--home", default=os.getenv("MCP_MEMORY_STDIO_STRESS_HOME"))
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--timeout-ms", type=float, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--seed-entities", type=int, default=int(os.getenv("MCP_MEMORY_STDIO_STRESS_SEED_ENTITIES", "6")))
    parser.add_argument("--seed-observations", type=int, default=int(os.getenv("MCP_MEMORY_STDIO_STRESS_SEED_OBSERVATIONS", "4")))
    parser.add_argument("--queue-timeout", type=float, default=DEFAULT_QUEUE_TIMEOUT)
    parser.add_argument("--command", default=os.getenv("MCP_MEMORY_STDIO_STRESS_COMMAND", "uv"))
    parser.add_argument("--arg", action="append", dest="args", help="Server command argument; repeat to override default 'run mcp-memory'.")
    parser.add_argument("--cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--model-cache-source",
        default=os.getenv(
            "MCP_MEMORY_STDIO_STRESS_MODEL_CACHE_SOURCE",
            str(Path.home() / ".cache" / "mcp-memory-v2"),
        ),
        help="Existing mcp-memory-v2 cache to symlink into isolated HOME; set empty to disable.",
    )
    parser.add_argument("--heavy", action="store_true", help="Use heavier local defaults unless explicit flags/env override them.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.heavy:
        args.processes = max(args.processes, 8)
        args.iterations = max(args.iterations, 100)
        args.seed_entities = max(args.seed_entities, 20)
        args.seed_observations = max(args.seed_observations, 15)

    home = args.home or tempfile.mkdtemp(prefix="mcp-memory-stdio-stress-")
    config = HarnessConfig(
        home=home,
        processes=args.processes,
        iterations=args.iterations,
        timeout_ms=args.timeout_ms,
        seed_entities=args.seed_entities,
        seed_observations=args.seed_observations,
        queue_timeout=args.queue_timeout,
        command=args.command,
        args=args.args or ["run", "mcp-memory"],
        cwd=args.cwd,
        model_cache_source=args.model_cache_source or None,
    )
    summary = run_harness(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    has_errors = any(count for count in summary["errors_by_type"].values())
    bad_exit = any(code for code in summary["process_exitcodes"])
    incomplete = summary["worker_done"] != config.processes
    return 1 if has_errors or bad_exit or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
