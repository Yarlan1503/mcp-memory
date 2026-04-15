> **Full Documentation** -- guides, tools reference, architecture, and maintenance at [cachorro.space](https://cachorro.space/mcp-memory/getting-started/)

# mcp-memory

A **drop-in replacement** for [Anthropic's MCP Memory server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) -- with SQLite persistence, vector embeddings, semantic search, and **Limbic Scoring** for dynamic ranking.

**Why?** The original server writes the entire knowledge graph to a JSONL file on every operation, with no locking or atomic writes. Under concurrent access (multiple MCP clients), this causes data corruption. This server replaces that with a proper SQLite database.

## Features

- **Drop-in compatible** with Anthropic's 8 MCP tools (same API, same behavior)
- **SQLite + WAL** -- safe concurrent access, no more corrupted JSONL
- **Semantic search** via sqlite-vec + ONNX embeddings (94+ languages)
- **Hybrid search** (FTS5 + KNN) -- combines full-text BM25 and semantic vector search via Reciprocal Rank Fusion. Finds entities by exact terms or semantic similarity -- or both at once.
- **Limbic Scoring** -- dynamic re-ranking with salience, temporal decay, co-occurrence signals, and hybrid search scores. Transparent to the API.
- **Semantic deduplication** -- automatic `similarity_flag` on new observations when cosine similarity >= 0.85 (with containment scoring for asymmetric text lengths)
- **Consolidation reports** -- read-only health checks for split candidates, flagged observations, stale entities, and large entities
- **Improved recency decay** -- `entity_access_log` tracking with `ALPHA_CONS=0.2` multi-day consolidation signal
- **Containment fix** -- proper handling of asymmetric text lengths (ratio >= 2.0) in deduplication scoring
- **Observation kinds** -- semantic classification of observations (hallazgo, decision, estado, spec, metrica, metadata, generic)
- **Observation supersedes** -- explicit replacement chain: new observations can supersede old ones, which get timestamped as superseded
- **Entity status** -- lifecycle tracking: activo, pausado, completado, archivado (with status-aware search de-boosting)
- **Relation context + vigencia** -- relations carry optional context, active/ended_at fields for temporal validity
- **Automatic inverse relations** -- contains/parte_de pairs created automatically
- **Reflections** -- independent narrative layer: free-form prose attached to entities/sessions/relations/global, with author and mood metadata, searchable via semantic + FTS5 hybrid search
- **Lightweight** -- ~500 MB total vs ~1.4 GB for similar solutions
- **Migration** -- one-click import from Anthropic's JSONL format
- **Zero config** -- works out of the box, optional model download for semantic search

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

This downloads a multilingual sentence model (~465 MB) to `~/.cache/mcp-memory-v2/models/`. Without it, all tools work fine -- only `search_semantic` will be unavailable.

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

### Core (Anthropic-compatible)

| Tool | Description |
|------|-------------|
| `create_entities` | Create or update entities (merges observations on conflict). Accepts `status` field. |
| `create_relations` | Create typed relations between entities. Accepts `context`; auto-creates inverse relations where defined. |
| `add_observations` | Add observations to an existing entity. Accepts `kind` and `supersedes` params for semantic classification and explicit replacement. |
| `delete_entities` | Delete entities and all their relations/observations |
| `delete_observations` | Delete specific observations from an entity |
| `delete_relations` | Delete specific relations between entities |
| `search_nodes` | Search by substring (name, type, observation content) |
| `open_nodes` | Retrieve entities by name. Accepts `kinds` filter, `include_superseded` flag. Returns `reflections` and relation metadata (context, active, ended_at). |

### Search & Analysis

| Tool | Description |
|------|-------------|
| `search_semantic` | Semantic search via vector embeddings with **Limbic Scoring** re-ranking |
| `find_duplicate_observations` | Find semantically duplicated observations within an entity (cosine + containment) |
| `consolidation_report` | Generate a read-only consolidation report (split candidates, flagged obs, stale entities) |

### Entity Management

| Tool | Description |
|------|-------------|
| `migrate` | Import from Anthropic's JSONL format (idempotent) |
| `analyze_entity_split` | Analyze if an entity needs splitting (TF-IDF topic grouping) |
| `propose_entity_split_tool` | Propose a split with suggested entity names and relations |
| `execute_entity_split_tool` | Execute an approved split (atomic transaction) |
| `find_split_candidates` | Find all entities that need splitting |

### Reflections

| Tool | Description |
|------|-------------|
| `add_reflection` | Add a narrative reflection to any entity, session, relation, or global. Accepts author, content, and mood. |
| `search_reflections` | Search reflections via semantic + FTS5 hybrid (RRF). Optional filters: author, mood, target_type. |

## Entity Types

8 canonical types:

| Type | Purpose |
|------|---------|
| `Proyecto` | Long-running projects |
| `Sesion` | Working sessions |
| `Sistema` | Systems and tools |
| `Decision` | Architectural/technical decisions |
| `Evento` | Time-bound events |
| `Persona` | People |
| `Recurso` | External resources |
| `Generic` | Default fallback |

## Observation Kinds

Semantic classification for observations:

| Kind | Purpose |
|------|---------|
| `hallazgo` | Findings and discoveries |
| `decision` | Decisions made |
| `estado` | State/status snapshots |
| `spec` | Specifications and requirements |
| `metrica` | Quantitative measurements |
| `metadata` | System-generated metadata |
| `generic` | Default (no classification) |

## Relation Types

4 categories:

| Category | Types | Example |
|----------|-------|---------|
| Structural | `contiene` / `parte_de` | Project contains Session |
| Production | `producido_por`, `contribuye_a` | Session produces Decision |
| Dependency | `depende_de`, `usa` | Project depends on System |
| Succession | `continua`, `sucedido_por` | Session continues Session |

Inverse relations are auto-created for `contiene`/`parte_de` pairs.

## Architecture

```
server.py (97 lines)          — FastMCP init + tool registration
├── tools/
│   ├── core.py              — 6 CRUD tools (Anthropic-compatible)
│   ├── search.py            — 3 search tools + ranking helpers
│   ├── entity_mgmt.py       — 6 entity management tools
│   ├── reflections.py       — 2 reflection tools
│   └── relations.py         — 2 tools (migrate, end_relation)
├── storage/                  — 7 mixins via multiple inheritance
│   ├── __init__.py           — MemoryStore facade (134 lines)
│   ├── schema.py            — SchemaMixin (migrations)
│   ├── core.py              — CoreMixin (entity/obs CRUD)
│   ├── relations.py         — RelationsMixin
│   ├── search.py            — SearchMixin (FTS + embeddings)
│   ├── access.py            — AccessMixin
│   ├── reflections.py       — ReflectionsMixin
│   └── consolidation.py     — ConsolidationMixin
├── embeddings.py             — EmbeddingEngine (ONNX, lazy load)
├── scoring.py                — Limbic Scoring + RRF
├── entity_splitter.py        — TF-IDF entity splitting
├── retry.py                  — retry_on_locked (concurrency)
└── config.py                 — Input limits + A/B config
```

- **Storage**: SQLite with WAL journaling, 5-second busy timeout, CASCADE deletes
- **Embeddings**: Singleton ONNX model loaded once at startup, L2-normalized cosine search
- **Limbic Scoring**: Re-ranks hybrid (KNN + FTS5) candidates using importance signals, temporal decay, co-occurrence patterns, and RRF scores -- transparent to the API
- **Concurrency**: `retry_on_locked` decorator with exponential backoff + jitter on 21 write methods. Safe multi-client access (tested with concurrent opencode sessions)
- **Reflections**: Parallel FTS5 (`reflection_fts`) and vector (`reflection_embeddings`) indexes for narrative layer, searched via the same RRF hybrid pipeline

## How It Works

Each entity gets an embedding vector generated from its text using a **Head+Tail+Diversity** selection strategy (budget: 480 tokens):

```
"{name} ({entity_type}) | {obs1} | {obs2} | ... | Rel: type -> target; ..."
```

When you call `search_semantic`, the pipeline runs in parallel:

1. **Semantic (KNN)** -- the query is encoded and compared against entity vectors via `sqlite-vec`
2. **Full-text (FTS5)** -- the query is searched against a BM25 index covering names, types, and observation content
3. **Merge (RRF)** -- results from both branches are combined using Reciprocal Rank Fusion (`score(d) = Sum 1/(k + rank)`)

The merged candidates are then re-ranked by the **Limbic Scoring** engine, which considers:

- **Salience** -- frequently accessed and well-connected entities rank higher
- **Temporal decay** -- recently used entities stay fresh; untouched entities fade
- **Co-occurrence** -- entities that appear together often reinforce each other

The output includes `limbic_score`, `scoring` (importance/temporal/cooc breakdown), and optionally `rrf_score` when FTS5 contributes results.

> **For full technical details**, see [DOCUMENTATION.md](docs/DOCUMENTATION.md) -- includes the scoring formula, RRF constants, schema DDL, and architecture diagrams.

## Testing

```bash
uv run pytest tests/ -v
```

313 tests across 15 test files covering all tools, embeddings, scoring, and edge cases. Zero regressions.

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
