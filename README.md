> **📖 [Full Documentation](https://cachorro.space/docs/mcp-memory)** — guides, API reference, architecture, and more at cachorro.space

# mcp-memory

A **drop-in replacement** for [Anthropic's MCP Memory server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — with SQLite persistence, vector embeddings, semantic search, and **🧠 Limbic Scoring** for dynamic ranking.

**Why?** The original server writes the entire knowledge graph to a JSONL file on every operation, with no locking or atomic writes. Under concurrent access (multiple MCP clients), this causes data corruption. This server replaces that with a proper SQLite database.

## Features

- **Drop-in compatible** with Anthropic's 9 MCP tools (same API, same behavior)
- **SQLite + WAL** — safe concurrent access, no more corrupted JSONL
- **Semantic search** via sqlite-vec + ONNX embeddings (94+ languages)
- **Hybrid search** (FTS5 + KNN) — combines full-text BM25 and semantic vector search via Reciprocal Rank Fusion. Finds entities by exact terms or semantic similarity — or both at once.
- **🧠 Limbic Scoring** — dynamic re-ranking with salience, temporal decay, co-occurrence signals, and hybrid search scores. Transparent to the API.
- **Lightweight** — ~500 MB total vs ~1.4 GB for similar solutions
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

This downloads a multilingual sentence model (~465 MB) to `~/.cache/mcp-memory-v2/models/`. Without it, all tools work fine — only `search_semantic` will be unavailable.

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
| `search_semantic` | Semantic search via vector embeddings (cosine similarity) with **Limbic Scoring** re-ranking |
| `migrate` | Import from Anthropic's JSONL format (idempotent) |

## Architecture

```
server.py (FastMCP)  ←→  storage.py (SQLite + sqlite-vec + FTS5)
                              ↑
                        embeddings.py (ONNX Runtime)
                              ↑
                        intfloat/multilingual-e5-small
                        (384d, 94+ languages, asymmetric retrieval)
                              ↑
                        scoring.py (Limbic Scoring + RRF)
                        salience · temporal decay · co-occurrence
```

- **Storage**: SQLite with WAL journaling, 5-second busy timeout, CASCADE deletes
- **Embeddings**: Singleton ONNX model loaded once at startup, L2-normalized cosine search
- **Limbic Scoring**: Re-ranks hybrid (KNN + FTS5) candidates using importance signals, temporal decay, co-occurrence patterns, and RRF scores — transparent to the API
- **Concurrency**: SQLite handles locking internally — no fcntl, no fs wars

## How It Works

Each entity gets an embedding vector generated from its text using a **Head+Tail+Diversity** selection strategy (budget: 480 tokens):

```
"{name} ({entity_type}) | {obs1} | {obs2} | ... | Rel: type → target; ..."
```

When you call `search_semantic`, the pipeline runs in parallel:

1. **Semantic (KNN)** — the query is encoded with the `"query: "` prefix and compared against entity vectors via `sqlite-vec`
2. **Full-text (FTS5)** — the query is searched against a BM25 index covering names, types, and observation content
3. **Merge (RRF)** — results from both branches are combined using Reciprocal Rank Fusion (`score(d) = Σ 1/(k + rank)`)

The merged candidates are then re-ranked by the **Limbic Scoring** engine, which considers:

- **Salience** — frequently accessed and well-connected entities rank higher
- **Temporal decay** — recently used entities stay fresh; untouched entities fade
- **Co-occurrence** — entities that appear together often reinforce each other

The output includes `limbic_score`, `scoring` (importance/temporal/cooc breakdown), and optionally `rrf_score` when FTS5 contributes results.

> **For full technical details**, see [DOCUMENTATION.md](docs/DOCUMENTATION.md) — includes the scoring formula, RRF constants, schema DDL, and architecture diagrams.

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
