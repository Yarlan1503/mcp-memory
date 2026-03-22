# mcp-memory

A **drop-in replacement** for [Anthropic's MCP Memory server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — with SQLite persistence, vector embeddings, and semantic search.

**Why?** The original server writes the entire knowledge graph to a JSONL file on every operation, with no locking or atomic writes. Under concurrent access (multiple MCP clients), this causes data corruption. This server replaces that with a proper SQLite database.

## Features

- **Drop-in compatible** with Anthropic's 9 MCP tools (same API, same behavior)
- **SQLite + WAL** — safe concurrent access, no more corrupted JSONL
- **Semantic search** via sqlite-vec + ONNX embeddings (50+ languages)
- **Lightweight** — ~150 MB total vs ~1.4 GB for similar solutions
- **Migration** — one-click import from Anthropic's JSONL format
- **Zero config** — works out of the box, optional model download for semantic search

## Quick Start

### 1. Add to your MCP config

```json
{
  "mcpServers": {
    "memory": {
      "command": ["uvx", "--from", "git+https://github.com/Yarlan1503/mcp-memory", "mcp-memory"]
    }
  }
}
```

Or clone and run locally:

```json
{
  "mcpServers": {
    "memory": {
      "command": ["uv", "run", "--directory", "/path/to/mcp-memory", "mcp-memory"]
    }
  }
}
```

### 2. Enable semantic search (optional)

```bash
cd /path/to/mcp-memory
uv run python scripts/download_model.py
```

This downloads a multilingual sentence model (~80 MB) to `~/.cache/mcp-memory-v2/models/`. Without it, all tools work fine — only `search_semantic` will be unavailable.

### 3. Migrate existing data (optional)

If you have an Anthropic MCP Memory JSONL file, use the `migrate` tool or call it directly:

```bash
uv run python -c "
from mcp_memory.storage import MemoryStore
from mcp_memory.migrate import migrate_jsonl
store = MemoryStore()
store.init_db()
result = migrate_jsonl(store, '~/.config/opencode/mcp-memory.jsonl')
print(result)
"
```

## MCP Tools

### Compatible with Anthropic (9 tools)

| Tool | Description |
|------|-------------|
| `create_entities` | Create or update entities (merges observations on conflict) |
| `create_relations` | Create typed relations between entities |
| `add_observations` | Add observations to an existing entity |
| `delete_entities` | Delete entities (cascades to observations + relations) |
| `delete_observations` | Delete specific observations |
| `delete_relations` | Delete specific relations |
| `search_nodes` | Search by substring (name, type, observation content) |
| `open_nodes` | Retrieve entities by name |
| `read_graph` | Read the entire knowledge graph |

### New tools (2)

| Tool | Description |
|------|-------------|
| `search_semantic` | Semantic search via vector embeddings (cosine similarity) |
| `migrate` | Import from Anthropic's JSONL format (idempotent) |

## Architecture

```
server.py (FastMCP)  ←→  storage.py (SQLite + sqlite-vec)
                              ↑
                        embeddings.py (ONNX Runtime)
                              ↑
                        paraphrase-multilingual-MiniLM-L12-v2
                        (384d, 50+ languages, CPU-only)
```

- **Storage**: SQLite with WAL journaling, 5-second busy timeout, CASCADE deletes
- **Embeddings**: Singleton ONNX model loaded once at startup, L2-normalized cosine search
- **Concurrency**: SQLite handles locking internally — no fcntl, no fs wars

## How It Works

Each entity gets an embedding vector generated from its concatenated content:

```
"{name} ({entity_type}): {observation_1}. {observation_2}. ..."
```

When you call `search_semantic`, the query is encoded with the same model and compared against all entity vectors using k-nearest neighbors (cosine distance) via `sqlite-vec`.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `pydantic` | Request/response validation |
| `sqlite-vec` | Vector similarity search in SQLite |
| `onnxruntime` | ONNX model inference (CPU) |
| `tokenizers` | HuggingFace fast tokenizer |
| `numpy` | Vector operations |
| `huggingface-hub` | Model download |

## License

MIT
