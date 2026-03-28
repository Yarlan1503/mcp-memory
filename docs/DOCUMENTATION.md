# MCP Memory v2 — Documentación Técnica

> **Versión**: 0.1.0 | **Licencia**: MIT | **Repositorio**: [github.com/Yarlan1503/mcp-memory](https://github.com/Yarlan1503/mcp-memory)

---

## Tabla de Contenidos

- [Visión General](#visión-general)
- [Arquitectura](#arquitectura)
- [Instalación y Configuración](#instalación-y-configuración)
- [Dependencias](#dependencias)
- [Modelo de Datos](#modelo-de-datos)
- [Modelos Pydantic](#modelos-pydantic)
- [MCP Tools](#mcp-tools)
- [Sistema de Embeddings](#sistema-de-embeddings)
- [Sistema Límbico — Scoring Dinámico](#sistema-límbico--scoring-dinámico)
- [Migración desde Anthropic](#migración-desde-anthropic)
- [Consideraciones y Gotchas](#consideraciones-y-gotchas)

---

## Visión General

### Qué es MCP Memory v2

MCP Memory v2 es un servidor de memoria persistente para el protocolo MCP (Model Context Protocol). Proporciona un *knowledge graph* donde los modelos de IA pueden almacenar entidades, observaciones y relaciones de forma estructurada, y recuperarlas en sesiones posteriores.

Se diseña como **reemplazo drop-in** del servidor oficial de Anthropic (`@modelcontextprotocol/server-memory`), conservando compatibilidad total con su API mientras añade búsqueda semántica y un backend de almacenamiento que escala correctamente.

### Por qué existe

El servidor oficial de Anthropic almacena todo el knowledge graph en un único archivo JSONL. Este enfoque funciona para demos, pero tiene problemas serios en producción:

| Problema | JSONL (Anthropic) | MCP Memory v2 |
|---|---|---|
| **Indexación** | Ninguna — recorrido lineal del archivo completo | Índices SQLite por nombre, tipo y contenido |
| **Búsqueda semántica** | No disponible | KNN con embeddings ONNX |
| **Concurrencia** | Race conditions confirmadas (fs.writeFile sin locking) | SQLite WAL con busy timeout |
| **Escala** | Degradación proporcional al tamaño del archivo | Consultas O(log n) indexadas |
| **Corrupción de datos** | Documentada en issues #1819, #2579 (mayo 2025, sin merge) | Transacciones ACID + rollback automático |

El servidor oficial reescribe el archivo completo en cada operación. Sin locking ni escritura atómica, las operaciones concurrentes generan fusión de JSONs y líneas duplicadas. MCP Memory v2 resuelve estos problemas de raíz con un motor de almacenamiento diseñado para datos persistentes.

### Compatibilidad con Anthropic MCP Memory

De las 11 tools que expone MCP Memory v2, **9 son 100% compatibles** con la API de Anthropic:

| Tool | Compatibilidad |
|---|---|
| `create_entities` | ✅ Anthropic |
| `create_relations` | ✅ Anthropic |
| `add_observations` | ✅ Anthropic |
| `delete_entities` | ✅ Anthropic |
| `delete_observations` | ✅ Anthropic |
| `delete_relations` | ✅ Anthropic |
| `search_nodes` | ✅ Anthropic |
| `open_nodes` | ✅ Anthropic |
| `read_graph` | ✅ Anthropic |
| `search_semantic` | 🆕 Nueva |
| `migrate` | 🆕 Nueva |

Las dos tools nuevas extienden la funcionalidad sin romper la API existente:

- **`search_semantic`** — Búsqueda por similitud semántica usando embeddings vectoriales. Encuentra entidades relevantes aunque no compartan palabras clave con la consulta.
- **`migrate`** — Migración idempotente desde el formato JSONL de Anthropic a SQLite. Procesa el archivo línea por línea, ignora líneas corruptas, y genera embeddings para todas las entidades importadas.

### Stack tecnológico

```
Python >= 3.12
├── FastMCP >= 2.0          # Framework MCP (servidor + registro de tools)
├── SQLite (stdlib)         # Base de datos persistente
│   └── WAL mode            # Escritura concurrente sin bloqueos
├── sqlite-vec >= 0.1.6     # Extensión vectorial para KNN
├── ONNX Runtime >= 1.17    # Inferencia de embeddings en CPU
├── HuggingFace tokenizers  # Tokenización rápida (Rust)
├── numpy >= 1.26           # Vectores y operaciones numéricas
└── pydantic >= 2.0         # Validación de inputs/outputs
```

**Build system**: hatchling  
**Comando de entrada**: `mcp-memory` (resuelve a `mcp_memory.server:main`)  
**Transporte**: stdio — compatible con Claude Desktop, OpenCode, Cursor y cualquier cliente MCP que use transporte estándar.

### Modelo de embeddings

El motor semántico usa **intfloat/multilingual-e5-small** (modelo de retrieval asimétrico entrenado por Intel):

- **Dimensiones**: 384 (float32, ONNX FP32)
- **Idiomas**: 94+ (incluyendo español, inglés, francés, alemán, chino, japonés)
- **Runtime**: CPU puro (no requiere GPU)
- **Tamaño**: ~465 MB (modelo ONNX + tokenizer)
- **Cache**: `~/.cache/mcp-memory-v2/models/`
- **Prefixes**: `"query: "` para consultas, `"passage: "` para entidades y documentos (requisito del modelo e5)

El pipeline de encoding: prepend prefix → tokenización → forward ONNX → mean pooling (masking de PAD) → normalización L2. Los vectores resultantes se serializan a raw bytes para almacenarlos en sqlite-vec.

---

## Arquitectura

### Vista de alto nivel

MCP Memory v2 sigue una arquitectura en capas con tres componentes principales: el servidor MCP (transporte y tools), la capa de almacenamiento (SQLite) y el motor de embeddings (ONNX).

```
┌──────────────────────────────────────┐
│         MCP Client (Claude Desktop,  │
│         OpenCode, Cursor, ...)       │
└──────────────┬───────────────────────┘
               │ stdio (JSON-RPC)
               ▼
┌──────────────────────────────────────┐
│         FastMCP Server               │
│  ┌────────────────────────────┐      │
│  │   11 MCP Tools             │      │
│  │   (9 Anthropic + 2 new)    │      │
│  └────┬──────────────┬────────┘      │
│       │              │               │
│  ┌────▼──────┐ ┌────▼──────────┐     │
│  │ Pydantic  │ │ EmbeddingEng. │     │
│  │ Validation│ │ (ONNX, lazy)  │     │
│  └────┬──────┘ └────┬──────────┘     │
│       │              │               │
│  ┌────▼──────────────▼────────┐      │
│  │       MemoryStore           │      │
│  │   (SQLite + sqlite-vec)     │      │
│  └──────────┬──────────────────┘      │
│             │                         │
│  ┌──────────▼──────────────────┐      │
│  │   scoring.py (Limbic)      │      │
│  │   Salience · Decay · Co-oc │      │
│  └─────────────────────────────┘      │
└──────────────────────────────────────┘
         │
         ▼
   ~/.config/opencode/
      mcp-memory/
         memory.db          (datos + vectores)
```

El servidor arranca como un proceso stdio que escucha JSON-RPC en *stdin* y responde por *stdout*. Los logs van a *stderr* para no interferir con el protocolo MCP.

### Flujo de datos: escritura (create_entities)

Cuando un cliente invoca `create_entities`, los datos pasan por cuatro etapas hasta persistir con su embedding semántico:

```
┌──────────┐
│  Client   │  Llama a create_entities([{name,
│           │    entityType, observations}])
└────┬─────┘
     │ JSON-RPC (stdio)
     ▼
┌──────────┐
│  FastMCP │  Deserializa, invoca handler
│  Server  │
└────┬─────┘
     │
     ▼
┌──────────┐
│  Pydantic│  EntityInput.model_validate(dict)
│  Model   │  → valida name (non-empty),
│          │    entityType, observations
└────┬─────┘
     │ entity_id
     ▼
┌──────────┐
│  Memory  │  1. upsert_entity(name, type)
│  Store   │  2. add_observations(entity_id,
│ (SQLite) │     obs) [dedup por contenido]
└────┬─────┘
     │ entity_id + datos completos
     ▼
┌──────────┐  Si engine.available:
│ Embedding│  1. prepare_entity_text(name,
│  Engine  │     type, obs)
│  (ONNX)  │  2. encode([text]) → float[384]
│          │  3. serialize_f32() → bytes
└────┬─────┘
     │ bytes (1536 bytes)
     ▼
┌──────────┐
│ sqlite-  │  INSERT OR REPLACE
│  vec     │  entity_embeddings(rowid,
│          │  embedding)
└──────────┘
```

El embedding se recalcula cada vez que cambia el contenido de una entidad (creación, observaciones nuevas, eliminación de observaciones). Esto garantiza que la representación vectorial refleje siempre el estado actual de la entidad.

### Flujo de datos: búsqueda semántica

La búsqueda semántica usa KNN sobre los vectores almacenados en sqlite-vec, seguido de un re-ranking con el Sistema Límbico:

```
┌──────────┐
│  Client   │  search_semantic(
│           │    "memoria del proyecto",
│           │    limit=10)
└────┬─────┘
     │ JSON-RPC (stdio)
     ▼
┌──────────┐
│  FastMCP │  Verifica engine.available
│  Server  │  → si no: error con instrucción
│          │    para descargar modelo
└────┬─────┘
     │ query string
     ▼
┌──────────┐
│ Embedding│  encode(["memoria del proyecto"])
│  Engine  │  → float[384] L2-normalised
│  (ONNX)  │
└────┬─────┘
     │ serialize_f32() → bytes
     ▼
┌──────────┐
│ sqlite-  │  SELECT rowid, distance
│  vec     │  FROM entity_embeddings
│ (KNN)    │  WHERE embedding MATCH ?
│          │  ORDER BY distance LIMIT 10×3
│          │  (over-retrieve by EXPANSION_FACTOR)
└────┬─────┘
     │ [{entity_id, distance}, ...] × 3×
     ▼
┌──────────┐
│ scoring  │  Fetch limbic signals:
│   .py    │    access_data, degree_data,
│ (Limbic) │    cooc_data, created_at
│          │  Compute limbic_score for each
│          │  candidate → re-rank → top-K
└────┬─────┘
     │ [{entity_id, distance, limbic_score}]
     ▼
┌──────────┐
│  Memory  │  Por cada resultado:
│  Store   │    get_entity_by_id(entity_id)
│          │    + get_observations(entity_id)
│          │  + record_access() (post-response)
│          │  + record_co_occurrences()
└────┬─────┘
     │ [{name, entityType, obs, distance}]
     ▼
┌──────────┐
│  Client   │  Respuesta con entidades
│           │  ordenadas por limbic score
└──────────┘
```

La métrica de distancia es **coseno** (`distance_metric=cosine` en la tabla virtual). Los resultados se ordenan de menor a mayor distancia, donde 0 = identidad y 2 = opuesto total.

### Capa de almacenamiento: MemoryStore

`MemoryStore` es la capa de persistencia. Envuelve una conexión SQLite y expone operaciones CRUD para entidades, observaciones y relaciones.

**Configuración SQLite**:

```python
PRAGMA journal_mode = WAL       # Escritura sin bloquear lecturas
PRAGMA busy_timeout  = 5000     # Espera 5s si hay lock
PRAGMA synchronous   = NORMAL   # Balance entre seguridad y velocidad
PRAGMA cache_size    = -64000   # 64 MB de caché
PRAGMA temp_store    = MEMORY   # Temporales en RAM
PRAGMA foreign_keys  = ON       # Integridad referencial
```

**Schema**:

| Tabla | Propósito |
|---|---|
| `entities` | Nodos del grafo (id, name, entity_type, timestamps) |
| `observations` | Datos adjuntos a entidades (entity_id FK, content) |
| `relations` | Aristas del grafo (from_entity, to_entity, relation_type) |
| `db_metadata` | Metadatos clave-valor del sistema |
| `entity_embeddings` (vec0) | Tabla virtual sqlite-vec para vectores float[384] |

**Índices**: Se crean índices sobre `observations(entity_id)`, `relations(from_entity)`, `relations(to_entity)`, `relations(relation_type)`, `entities(name)` y `entities(entity_type)`.

**Borrado en cascada**: Las entidades eliminadas cascadean a observaciones y relaciones automáticamente vía `ON DELETE CASCADE`. Los embeddings se eliminan manualmente antes porque `vec0` no soporta CASCADE nativo.

**Ruta por defecto**: `~/.config/opencode/mcp-memory/memory.db` (se crea el directorio si no existe).

### Motor de embeddings: EmbeddingEngine

El `EmbeddingEngine` encapsula toda la lógica de inferencia de embeddings usando un modelo ONNX.

**Patrón singleton + lazy load**:

```
┌─────────────────────────────────────────────┐
│           Arranque del servidor              │
│                                             │
│  server.py:                                 │
│    store = MemoryStore()  → init_db()       │
│    store.init_db()        → schema listo    │
│                                             │
│  EmbeddingEngine._instance = None           │
│  (NO se carga el modelo todavía)            │
└──────────────────┬──────────────────────────┘
                   │
                   │  Primer call a _get_engine()
                   │  (ej: search_semantic o
                   │   _recompute_embedding)
                   ▼
┌─────────────────────────────────────────────┐
│         Inicialización bajo demanda          │
│                                             │
│  EmbeddingEngine.get_instance()             │
│    → cls() si _instance es None             │
│    → busca model.onnx + tokenizer.json      │
│       en ~/.cache/mcp-memory-v2/models/     │
│    → ONNX InferenceSession (CPU)            │
│    → HuggingFace Tokenizer                  │
│    → self._available = True                 │
│                                             │
│  Si archivos no existen:                    │
│    → self._available = False                │
│    → server funciona SIN embeddings         │
└─────────────────────────────────────────────┘
```

El servidor **arranca sin modelo**. Las 9 tools compatibles con Anthropic funcionan inmediatamente (usan solo SQLite). El motor de embeddings se inicializa bajo demanda la primera vez que se necesita: ya sea una llamada a `search_semantic`, o una operación de escritura que recalcula embeddings (`create_entities`, `add_observations`, `delete_observations`).

Si los archivos del modelo no están en disco, `engine.available` es `False` y el servidor continúa funcionando — las operaciones CRUD siguen normales, pero la búsqueda semántica y el recálculo de embeddings se saltan silenciosamente (con un log de advertencia).

**Pipeline de encoding**:

```
Texto de entrada
       │
       ▼
┌──────────────┐
│  Tokenizer   │  HuggingFace fast tokenizer
│  (trunc=512, │  input_ids + attention_mask
│   pad=512)   │  → int64 arrays
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ONNX Runtime│  Forward pass en CPU
│  Inference   │  → (batch, seq_len, 384)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Mean Pool   │  Promedio de token embeddings
│  (masked)    │  ignorando [PAD] via attention_mask
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  L2 Normalise│  ||v|| = 1 para cosine similarity
│  → float32   │
└──────────────┘
```

**Formateo de entidades**: Antes de codificar, las entidades se transforman a texto plano usando Head+Tail+Diversity (budget 480 tokens):

```
"{name} ({entity_type}) | {obs1} | {obs2} | ... | Rel: tipo → destino; ..."
```

Este formato permite que el embedding capture el nombre, tipo, observaciones clave y contexto relacional, respetando el límite de 480 tokens del modelo e5-small.

---

## Instalación y Configuración

### Requisitos Previos

- **Python** >= 3.12
- **pip** o **uv** (recomendado) para gestión de dependencias
- **Git** para clonar el repositorio

### Instalación desde Source

Clona el repositorio y sincroniza las dependencias con `uv`:

```bash
git clone https://github.com/Yarlan1503/mcp-memory.git
cd mcp-memory
uv sync
```

`uv sync` crea un entorno virtual, resuelve todas las dependencias declaradas en `pyproject.toml` y genera el entry point `mcp-memory`.

### Descarga del Modelo de Embeddings

El servidor usa un modelo ONNX de sentence-transformers para búsqueda semántica. Descárgalo antes de usar `search_semantic`:

```bash
uv run python scripts/download_model.py
```

Esto descarga los siguientes archivos a `~/.cache/mcp-memory-v2/models/`:

| Archivo | Origen en el repositorio HF |
|---------|-----------------------------|
| `model.onnx` | `onnx/model.onnx` |
| `tokenizer.json` | Raíz |
| `tokenizer_config.json` | Raíz |
| `special_tokens_map.json` | Raíz |

**Fuente del modelo**: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

> **Nota:** Esta descarga es opcional. El servidor arranca sin el modelo (ver [Lazy loading del modelo](#nota-sobre-el-modelo-de-embeddings)).

### Ejecución del Servidor

Inicia el servidor MCP en modo stdio:

```bash
uv run mcp-memory
```

- **Transporte:** stdio (protocolo MCP)
- **Logs:** se escriben en stderr (no interferir con la comunicación MCP por stdout)
- **Servidor:** se registra como `"memory"` en el protocolo MCP

### Configuración para Claude Desktop

Añade el servidor al archivo de configuración JSON de Claude Desktop:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "mcp-memory"],
      "cwd": "/path/to/mcp-memory"
    }
  }
}
```

Reemplaza `/path/to/mcp-memory` con la ruta absoluta al directorio donde clonaste el repositorio.

### Configuración para OpenCode

Añade el servidor a la sección `mcp` de `opencode.json`:

```json
{
  "mcp": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory", "run", "mcp-memory"]
    }
  }
}
```

Reemplaza `/path/to/mcp-memory` con la ruta absoluta al repositorio.

### Nota sobre el Modelo de Embeddings

El servidor arranca **sin cargar** el modelo de embeddings. La carga es lazy — solo se instancia la primera vez que una tool la necesita (`search_semantic` o cualquier operación CRUD que genere embeddings).

Comportamiento según disponibilidad del modelo:

| Escenario | Comportamiento |
|-----------|---------------|
| Modelo descargado | Búsqueda semántica y generación de embeddings funcionan normalmente |
| Modelo **no** descargado | `search_semantic` retorna un error claro: `Embedding model not available. Run 'python scripts/download_model.py' to download the model first.` |
| Modelo **no** descargado | Las demás 10 tools (CRUD de entidades, relaciones, observaciones, lectura del grafo, migración) funcionan sin afectación |

---

## Dependencias

### Runtime

| Paquete | Versión mínima | Propósito |
|---------|---------------|-----------|
| `fastmcp` | >= 2.0 | Framework MCP para registro de tools y transporte stdio |
| `pydantic` | >= 2.0 | Validación de modelos de entrada/salida (entities, relations) |
| `numpy` | >= 1.26 | Operaciones numéricas y manipulación de vectores de embeddings |
| `sqlite-vec` | >= 0.1.6 | Extensión SQLite para búsqueda vectorial (KNN) |
| `tokenizers` | >= 0.19 | Tokenización rápida de HuggingFace para el modelo de embeddings |
| `onnxruntime` | >= 1.17 | Inferencia ONNX en CPU para el modelo de embeddings |
| `huggingface-hub` | >= 0.20 | Descarga de modelos desde HuggingFace Hub |

### Desarrollo

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `pytest` | >= 8.0 | Ejecución de tests |

### Build

| Componente | Valor |
|-----------|-------|
| Build backend | `hatchling` |
| Python requerido | >= 3.12 |

### Nota sobre `sqlite-vec`

La extensión `sqlite-vec` es **opcional** en tiempo de ejecución. Si no se puede cargar (por ejemplo, en plataformas donde no hay binarios precompilados), el servidor continúa funcionando normalmente pero sin capacidad de búsqueda semántica. Las operaciones CRUD del grafo de conocimiento y las demás tools no se ven afectadas.

---

## Modelo de Datos

### Visión General

MCP Memory v2 almacena un *knowledge graph* en SQLite compuesto por tres elementos fundamentales: **entidades**, **observaciones** y **relaciones**. Sobre esta estructura se extiende una cuarta capa de **embeddings vectoriales** (via `sqlite-vec`) que habilita la búsqueda semántica.

Las entidades representan nodos del grafo. Las observaciones son hechos atados a una entidad. Las relaciones conectan dos entidades con un tipo de vínculo. Los embeddings proyectan cada entidad en un espacio vectorial de 384 dimensiones para permitir consultas por similitud semántica (cosine distance).

### Diagrama Entidad-Relación

```
  ┌──────────────┐       ┌─────────────────┐
  │   entities   │──1:N──│  observations   │
  │              │       │                 │
  │  id (PK)     │       │  id (PK)        │
  │  name        │       │  entity_id (FK) │
  │  entity_type │       │  content        │
  │  created_at  │       │  created_at     │
  │  updated_at  │       └─────────────────┘
  │              │
  │              │──N:1──┌─────────────────┐
  │              │       │   relations     │
  │              │◄──────│                 │
  │              │ from  │  id (PK)        │
  └──────┬───────┘       │  from_entity(FK)│
         │               │  to_entity  (FK)│
         │               │  relation_type  │
         │               │  created_at     │
         │               │  UNIQUE(from,   │
         │               │    to, type)    │
         │               └────────┬────────┘
         │                        │
         │  ┌─────────────────┐   │
         └──│entity_embeddings│   │
     1:1  │  (VIRTUAL)      │   │
              │  rowid = id    │   │
              │  embedding[384]│   │
              │  cosine dist.  │   │
              └─────────────────┘

  ┌─────────────────┐        ┌──────────────────┐
  │  entity_access  │        │ co_occurrences   │
  │                 │        │                  │
  │  entity_id (PK  │        │ entity_a_id (PK  │
  │    → entities)  │        │ entity_b_id (PK  │
  │  access_count   │        │   → entities)    │
  │  last_access    │        │ co_count         │
  └─────────────────┘        │ last_co          │
                             └──────────────────┘

  ┌─────────────────┐
  │  db_metadata    │  (independiente)
  │  key (PK)       │
  │  value          │
  └─────────────────┘
```

### Tablas

#### `entities`

Tabla principal del knowledge graph. Cada fila representa un nodo con nombre único.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Identificador interno único |
| `name` | `TEXT` | `NOT NULL UNIQUE` | Nombre legible de la entidad. Clave de negocio — no pueden existir dos entidades con el mismo nombre |
| `entity_type` | `TEXT` | `NOT NULL DEFAULT 'Generic'` | Clasificación de la entidad (ej. `Sesion`, `Componente`, `Sistema`) |
| `created_at` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Marca temporal de creación en formato ISO-8601 |
| `updated_at` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Marca temporal de última actualización |

#### `observations`

Hechos o datos atados a una entidad. Una entidad puede tener cero o muchas observaciones.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Identificador interno único |
| `entity_id` | `INTEGER` | `NOT NULL REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad padre. El cascade elimina observaciones cuando se borra la entidad |
| `content` | `TEXT` | `NOT NULL` | Texto libre del hecho o dato observado |
| `created_at` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Marca temporal de creación |

> **Nota**: `ON DELETE CASCADE` garantiza integridad referencial: eliminar una entidad elimina todas sus observaciones en cascada, sin huérfanos.

#### `relations`

Aristas que conectan dos entidades con un tipo de relación semántica.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Identificador interno único |
| `from_entity` | `INTEGER` | `NOT NULL REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad de origen |
| `to_entity` | `INTEGER` | `NOT NULL REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad de destino |
| `relation_type` | `TEXT` | `NOT NULL` | Tipo de relación (ej. `uses`, `depends_on`, `part_of`) |
| `created_at` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Marca temporal de creación |

> **Nota**: La restricción `UNIQUE(from_entity, to_entity, relation_type)` impide duplicar relaciones idénticas entre las mismas dos entidades. No puede existir más de una relación del mismo tipo entre un par dado.

#### `entity_embeddings` (Virtual)

Tabla virtual implementada con la extensión `sqlite-vec` (`vec0`). Almacena el vector de embedding de cada entidad para búsqueda semántica.

| Columna | Tipo | Descripción |
|---|---|---|
| `embedding` | `float[384]` | Vector de 384 dimensiones generado por el modelo ONNX. Métrica de distancia: **cosine** |
| `rowid` | `INTEGER` (implícito) | Corresponde a `entities.id`. Es la clave que vincula el embedding con su entidad |

> **Nota técnica**: Las tablas virtuales `vec0` usan el `rowid` implícito de SQLite como clave primaria. Al insertar un embedding para la entidad con `id = N`, se usa `rowid = N`. Esto permite JOINs directos entre `entity_embeddings` y `entities` sin una columna FK explícita:

```sql
SELECT e.name, e.entity_type
FROM entities e
JOIN entity_embeddings ee ON e.id = ee.rowid
WHERE ee.embedding MATCH ?
ORDER BY distance;
```

#### `db_metadata`

Tabla auxiliar key-value para almacenar metadatos del sistema.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `key` | `TEXT` | `PRIMARY KEY` | Clave única del metadato |
| `value` | `TEXT` | `NOT NULL` | Valor asociado |

Se utiliza internamente para persistir información como la versión del schema, timestamp de última migración y configuraciones del sistema.

#### `entity_access`

Tabla de soporte para el Sistema Límbico. Registra la frecuencia y recencia de acceso a cada entidad durante `search_semantic`.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `entity_id` | `INTEGER` | `PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad. Una fila por entidad. |
| `access_count` | `INTEGER` | `NOT NULL DEFAULT 1` | Número de veces que la entidad apareció en resultados de `search_semantic` |
| `last_access` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Timestamp del último acceso (usado para temporal decay) |

#### `co_occurrences`

Tabla de soporte para el Sistema Límbico. Registra cuántas veces dos entidades aparecen juntas en los resultados de `search_semantic`.

| Columna | Tipo | Restricciones | Descripción |
|---|---|---|---|
| `entity_a_id` | `INTEGER` | `NOT NULL REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad con ID menor (orden canónico) |
| `entity_b_id` | `INTEGER` | `NOT NULL REFERENCES entities(id) ON DELETE CASCADE` | FK a la entidad con ID mayor |
| `co_count` | `INTEGER` | `NOT NULL DEFAULT 1` | Número de co-ocurrencias registradas |
| `last_co` | `TEXT` | `NOT NULL DEFAULT (datetime('now'))` | Timestamp de la última co-ocurrencia |

> **Nota**: La PK compuesta `(entity_a_id, entity_b_id)` garantiza orden canónico: `entity_a_id < entity_b_id` siempre. Esto evita duplicados como `(A, B)` y `(B, A)`.

### Índices

Los siguientes índices optimizan las consultas más frecuentes:

| Índice | Tabla | Columna(s) | Propósito |
|---|---|---|---|
| `idx_entities_name` | `entities` | `name` | Búsqueda rápida por nombre |
| `idx_entities_type` | `entities` | `entity_type` | Filtrado por tipo de entidad |
| `idx_obs_entity` | `observations` | `entity_id` | Recuperar observaciones de una entidad |
| `idx_rel_from` | `relations` | `from_entity` | Relaciones que parten de una entidad |
| `idx_rel_to` | `relations` | `to_entity` | Relaciones que llegan a una entidad |
| `idx_rel_type` | `relations` | `relation_type` | Filtrado por tipo de relación |
| `idx_access_last` | `entity_access` | `last_access` | Ordenar por recencia de acceso |
| `idx_cooc_b` | `co_occurrences` | `entity_b_id` | Lookup de co-ocurrencias por entidad B |

### Schema SQL Completo

```sql
CREATE TABLE entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    entity_type TEXT    NOT NULL DEFAULT 'Generic',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    from_entity   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_entity     INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(from_entity, to_entity, relation_type)
);

CREATE VIRTUAL TABLE entity_embeddings
USING vec0(embedding float[384] distance_metric=cosine);

CREATE TABLE db_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Limbic scoring tables
CREATE TABLE entity_access (
    entity_id    INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    access_count INTEGER NOT NULL DEFAULT 1,
    last_access  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE co_occurrences (
    entity_a_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    co_count     INTEGER NOT NULL DEFAULT 1,
    last_co      TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (entity_a_id, entity_b_id)
);

-- Índices
CREATE INDEX idx_entities_name  ON entities(name);
CREATE INDEX idx_entities_type  ON entities(entity_type);
CREATE INDEX idx_obs_entity     ON observations(entity_id);
CREATE INDEX idx_rel_from       ON relations(from_entity);
CREATE INDEX idx_rel_to         ON relations(to_entity);
CREATE INDEX idx_rel_type       ON relations(relation_type);
CREATE INDEX idx_access_last    ON entity_access(last_access);
CREATE INDEX idx_cooc_b         ON co_occurrences(entity_b_id);
```

---

## Modelos Pydantic

### Propósito

Los modelos Pydantic cumplen una función dual en MCP Memory v2:

1. **Validación de inputs**: Cada MCP tool recibe datos del cliente (típicamente JSON). Los modelos validan que la estructura y los tipos sean correctos antes de tocar la base de datos.
2. **Serialización de outputs**: Las respuestas de las tools se serializan a JSON de forma consistente y tipada, garantizando que el cliente reciba siempre la misma estructura.

En total, las **11 MCP tools** usan estos modelos para validar y devolver datos sobre entidades y relaciones.

### `EntityInput`

Modelo de entrada para crear o actualizar entidades. Valida que el nombre no esté vacío y asigna valores por defecto razonables.

```python
class EntityInput(BaseModel):
    name: str = Field(..., min_length=1)           # Requerido, mínimo 1 carácter
    entityType: str = Field(default="Generic")     # Opcional, "Generic" por defecto
    observations: list[str] = Field(default_factory=list)  # Opcional, lista vacía
```

| Campo | Tipo | Requerido | Default | Validación |
|---|---|---|---|---|
| `name` | `str` | Sí | — | `min_length=1` |
| `entityType` | `str` | No | `"Generic"` | — |
| `observations` | `list[str]` | No | `[]` | Nueva lista por instancia (`default_factory`) |

> **Nota**: `observations` usa `default_factory=list` en vez de `default=[]` para evitar el bug clásico de mutable default arguments en Python.

### `EntityOutput`

Modelo de salida para entidades. Refleja la estructura que el cliente recibe como respuesta.

```python
class EntityOutput(BaseModel):
    name: str
    entityType: str
    observations: list[str]
```

Todos los campos son obligatorios. No hay valores por defecto porque el servidor siempre los rellena desde la base de datos.

### `RelationInput` / `RelationOutput`

Ambos modelos comparten la misma estructura de campos. La diferencia está en el uso: `RelationInput` valida los datos que llega del cliente, `RelationOutput` serializa la respuesta.

```python
class RelationInput(BaseModel):
    from_entity: str = Field(..., alias="from")
    to_entity: str   = Field(..., alias="to")
    relationType: str
    model_config = {"populate_by_name": True}

class RelationOutput(BaseModel):
    from_entity: str = Field(..., alias="from")
    to_entity: str   = Field(..., alias="to")
    relationType: str
    model_config = {"populate_by_name": True}
```

| Campo | Alias JSON | Tipo | Requerido |
|---|---|---|---|
| `from_entity` | `"from"` | `str` | Sí |
| `to_entity` | `"to"` | `str` | Sí |
| `relationType` | — | `str` | Sí |

### Notas sobre Aliases

Los campos `from_entity` y `to_entity` usan **aliases** en Pydantic por una razón técnica: `from` y `to` son **palabras reservadas de Python** (keywords), por lo que no pueden usarse como nombres de atributo en una clase.

Pydantic resuelve esto con `Field(..., alias="from")`:

- **En Python**: se accede como `relation.from_entity` (nombre legal)
- **En JSON**: se serializa como `"from"` (formato esperado por Anthropic)

La configuración `populate_by_name=True` permite que el modelo acepte tanto el nombre del atributo como el alias durante la deserialización:

```python
# Ambas formas funcionan con populate_by_name=True:
RelationInput(**{"from": "EntityA", "to": "EntityB", "relationType": "uses"})
RelationInput(**{"from_entity": "EntityA", "to_entity": "EntityB", "relationType": "uses"})
```

Esto garantiza **compatibilidad bidireccional** con el formato de Anthropic MCP (que envía `"from"` / `"to"`) y con llamadas internas que usan los nombres Python.

### Código Fuente

Archivo `models.py` — código íntegro:

```python
from pydantic import BaseModel, Field


class EntityInput(BaseModel):
    name: str = Field(..., min_length=1)
    entityType: str = Field(default="Generic")
    observations: list[str] = Field(default_factory=list)


class EntityOutput(BaseModel):
    name: str
    entityType: str
    observations: list[str]


class RelationInput(BaseModel):
    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str
    model_config = {"populate_by_name": True}


class RelationOutput(BaseModel):
    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    relationType: str
    model_config = {"populate_by_name": True}
```

---

## MCP Tools

MCP Memory v2 expone 11 tools vía el protocolo MCP. Las primeras 9 son **100% compatibles** con el formato del servidor MCP Memory de Anthropic, lo que permite usarlo como drop-in replacement. Las 2 restantes son extensiones propias que añaden búsqueda semántica y migración desde el formato JSONL original.

### Tools compatibles con Anthropic (9)

#### 1. `create_entities`

**Descripción**: Create or update entities in the knowledge graph. If an entity already exists, merge observations (don't overwrite). Returns the created/updated entities.

**Firma**: `create_entities(entities: list[dict[str, Any]]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `entities` | `list[dict[str, Any]]` | Sí | Lista de entidades a crear o actualizar |

Cada dict dentro de `entities` se valida con el modelo `EntityInput` de Pydantic:

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|-----------|---------|-------------|
| `name` | `str` | Sí | — | Nombre de la entidad (mínimo 1 carácter). Actúa como identificador único. |
| `entityType` | `str` | No | `"Generic"` | Tipo/categoría de la entidad (ej. `"Sesion"`, `"Componente"`, `"Tarea"`). |
| `observations` | `list[str]` | No | `[]` | Lista de observaciones asociadas a la entidad. |

**Respuesta**:

```json
{
  "entities": [
    {
      "name": "MCP Memory v2",
      "entityType": "Sistema",
      "observations": [
        "Stack: FastMCP + SQLite-vec + ONNX embeddings",
        "11 tools MCP: 9 Anthropic-compat + 2 nuevos"
      ]
    }
  ]
}
```

Error de validación:

```json
{
  "error": "1 validation error for EntityInput\nname\n  Field required [type=missing, ...]"
}
```

**Notas**:
- Implementa **upsert** vía `INSERT … ON CONFLICT(name) DO UPDATE`. Si la entidad ya existe, actualiza `entity_type` y `updated_at` sin borrar observaciones previas.
- Las observaciones se **mergan**: las nuevas se añaden a las existentes y los duplicados exactos se descartan.
- Genera el **embedding** al finalizar con el snapshot completo de la entidad. Si el motor de embeddings no está disponible, la operación se completa sin error.

---

#### 2. `create_relations`

**Descripción**: Create relations between entities. Both entities must exist. Returns created relations or errors for missing entities.

**Firma**: `create_relations(relations: list[dict[str, Any]]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `relations` | `list[dict[str, Any]]` | Sí | Lista de relaciones a crear |

Cada dict dentro de `relations` se valida con el modelo `RelationInput` de Pydantic:

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `from` | `str` | Sí | Nombre de la entidad origen. Debe existir en el grafo. |
| `to` | `str` | Sí | Nombre de la entidad destino. Debe existir en el grafo. |
| `relationType` | `str` | Sí | Tipo de relación (ej. `"contiene"`, `"depende_de"`, `"usa"`). |

**Respuesta** (todas creadas exitosamente):

```json
{
  "relations": [
    {
      "from": "Proyecto Infraestructura",
      "to": "Tarea D1 BigQuery",
      "relationType": "contiene"
    }
  ]
}
```

Respuesta con errores (entidad no encontrada):

```json
{
  "relations": [
    { "error": "Entity not found: Entidad Inexistente" }
  ],
  "errors": [
    "Entity not found: Entidad Inexistente"
  ]
}
```

Respuesta con relación duplicada:

```json
{
  "relations": [
    {
      "from": "A",
      "to": "B",
      "relationType": "usa",
      "error": "Relation already exists"
    }
  ]
}
```

**Notas**:
- Ambas entidades (`from` y `to`) **deben existir** previamente. Si falta alguna, la relación no se crea y se registra el error.
- La tabla `relations` tiene constraint `UNIQUE(from_entity, to_entity, relation_type)`. Si la relación ya existe, se retorna el dict con `"error": "Relation already exists"` dentro de la lista de resultados.
- **No toca embeddings**. Las relaciones son metadatos estructurales que no participan en la búsqueda semántica.

---

#### 3. `add_observations`

**Descripción**: Add observations to an existing entity.

**Firma**: `add_observations(name: str, observations: list[str]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `name` | `str` | Sí | Nombre exacto de la entidad. Debe existir. |
| `observations` | `list[str]` | Sí | Lista de observaciones a añadir. Duplicados exactos se descartan. |

**Respuesta**:

```json
{
  "entity": {
    "name": "MCP Memory v2",
    "entityType": "Sistema",
    "observations": [
      "Stack: FastMCP + SQLite-vec + ONNX embeddings",
      "Bug de transacciones corregido",
      "Nueva observación agregada"
    ]
  }
}
```

Error (entidad no encontrada):

```json
{
  "error": "Entity not found: Nombre Inexistente"
}
```

**Notas**:
- La entidad **debe existir**. No crea entidades nuevas — use `create_entities` para eso.
- Las observaciones duplicadas exactas se descartan silenciosamente.
- **Regenera el embedding** con el snapshot actualizado de la entidad (todas las observaciones existentes + las nuevas).

---

#### 4. `delete_entities`

**Descripción**: Delete entities and all their relations/observations.

**Firma**: `delete_entities(entityNames: list[str]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `entityNames` | `list[str]` | Sí | Lista de nombres de entidades a eliminar. |

**Respuesta**:

```json
{
  "deleted": ["Entidad A", "Entidad B"]
}
```

Con errores (entidades no encontradas):

```json
{
  "deleted": ["Entidad A"],
  "errors": ["Entity not found: Entidad Inexistente"]
}
```

**Notas**:
- Eliminación en **cascada**: al borrar una entidad se eliminan automáticamente todas sus observaciones y relaciones (`ON DELETE CASCADE`).
- **CRÍTICO — embeddings**: la tabla virtual `vec0` de sqlite-vec **no soporta CASCADE**. El código elimina los embeddings manualmente **antes** de eliminar las entidades: (1) buscar IDs, (2) eliminar de `entity_embeddings` por `rowid`, (3) eliminar de `entities`. Todo dentro de una transacción implícita de SQLite.
- Puede fallar esporádicamente con `"cannot start a transaction"` bajo alta concurrencia en modo WAL. El reintento suele resolverlo.

---

#### 5. `delete_observations`

**Descripción**: Delete specific observations from an entity.

**Firma**: `delete_observations(name: str, observations: list[str]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `name` | `str` | Sí | Nombre exacto de la entidad. Debe existir. |
| `observations` | `list[str]` | Sí | Lista de observaciones a eliminar. Match por contenido exacto. |

**Respuesta**:

```json
{
  "entity": {
    "name": "MCP Memory v2",
    "entityType": "Sistema",
    "observations": [
      "Stack: FastMCP + SQLite-vec + ONNX embeddings"
    ]
  }
}
```

Error (entidad no encontrada):

```json
{
  "error": "Entity not found: Nombre Inexistente"
}
```

**Notas**:
- La eliminación es por **match exacto** del contenido de la observación. No usa patrones ni substrings.
- Si una observación no existe para esa entidad, simplemente no se elimina nada (no hay error por observación no encontrada).
- **Regenera el embedding** con las observaciones restantes.

---

#### 6. `delete_relations`

**Descripción**: Delete relations between entities.

**Firma**: `delete_relations(relations: list[dict[str, Any]]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `relations` | `list[dict[str, Any]]` | Sí | Lista de relaciones a eliminar |

Cada dict se valida con el modelo `RelationInput` (mismos campos que `create_relations`).

**Respuesta**:

```json
{
  "deleted": [
    {
      "from": "Proyecto Infraestructura",
      "to": "Tarea D1 BigQuery",
      "relationType": "contiene"
    }
  ]
}
```

Con errores:

```json
{
  "deleted": [],
  "errors": [
    "Relation not found: A -> B (contiene)"
  ]
}
```

**Notas**:
- Se requiere el **triple completo** (`from` + `to` + `relationType`) para identificar la relación.
- Si la entidad no existe: `"Entity not found: X or Y"`.
- Si la relación no existe: `"Relation not found: X -> Y (relationType)"`.
- **No toca embeddings**.

---

#### 7. `search_nodes`

**Descripción**: Search for nodes in the knowledge graph by name, type, or observation content.

**Firma**: `search_nodes(query: str) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `query` | `str` | Sí | Término de búsqueda. Se aplica como patrón LIKE a múltiples campos. |

**Respuesta**:

```json
{
  "entities": [
    {
      "name": "MCP Memory v2 - Implementación",
      "entityType": "Tarea",
      "observations": [
        "8 tareas secuenciadas: T1 Scaffold → T2 Models → ...",
        "Stack: FastMCP 3.1.1 + sqlite-vec + ONNX"
      ]
    }
  ]
}
```

**Notas**:
- Usa **búsqueda LIKE** con patrón `%query%` en tres campos simultáneos: `name`, `entity_type` y `content` de observaciones.
- La consulta usa `SELECT DISTINCT` para evitar duplicados.
- **No requiere** el modelo ONNX de embeddings.
- Búsqueda **case-sensitive** por defecto (comportamiento de SQLite con `LIKE`).

---

#### 8. `open_nodes`

**Descripción**: Open specific nodes by name. Returns full entity data with observations.

**Firma**: `open_nodes(names: list[str]) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `names` | `list[str]` | Sí | Lista de nombres de entidades a recuperar. |

**Respuesta**:

```json
{
  "entities": [
    {
      "name": "MCP Memory v2",
      "entityType": "Sistema",
      "observations": [
        "11 tools MCP: 9 Anthropic-compat + 2 nuevos",
        "Ubicación: ~/.config/opencode/mcp-memory/"
      ]
    }
  ]
}
```

**Notas**:
- Búsqueda por **nombre exacto** (`WHERE name = ?`). No usa patrones ni LIKE.
- Si un nombre no coincide con ninguna entidad, simplemente no se incluye en los resultados (no hay error).
- **No incluye relaciones** en la respuesta. Para obtener relaciones, use `read_graph`.

---

#### 9. `read_graph`

**Descripción**: Read the entire knowledge graph. Returns all entities with observations and all relations.

**Firma**: `read_graph() → dict[str, Any]`

**Parámetros**: Ninguno.

**Respuesta**:

```json
{
  "entities": [
    {
      "name": "MCP Memory v2",
      "entityType": "Sistema",
      "observations": [
        "11 tools MCP: 9 Anthropic-compat + 2 nuevos",
        "Repositorio: https://github.com/Yarlan1503/mcp-memory (MIT)"
      ]
    }
  ],
  "relations": [
    {
      "from": "Sesión 2026-03-21",
      "to": "MCP Memory v2",
      "relationType": "creo"
    }
  ]
}
```

**Notas**:
- Retorna el **dump completo** del grafo en formato Anthropic-compatible.
- Para cada entidad se incluye su lista completa de observaciones (ordenadas por `id`).
- Las relaciones se resuelven con `JOIN` para incluir nombres de entidades en lugar de IDs numéricos.
- **Costoso en tokens**: con un grafo grande, esta tool puede retornar miles de líneas. Prefiera `search_nodes` o `search_semantic` para consultas específicas.

---

### Tools nuevas de MCP Memory v2 (2)

#### 10. `search_semantic`

**Descripción**: Semantic search using vector embeddings. Finds entities most similar to the query. Results are re-ranked with the Limbic Scoring system (salience, temporal decay, co-occurrence) for improved relevance. Requires the embedding model to be downloaded (run `download_model.py` first).

**Firma**: `search_semantic(query: str, limit: int = 10) → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `query` | `str` | Sí | — | Texto de consulta. Se codifica como embedding y se compara contra los vectores almacenados. |
| `limit` | `int` | No | `10` | Número máximo de resultados a retornar. |

**Respuesta**:

```json
{
  "results": [
    {
      "name": "MCP Memory v2",
      "entityType": "Sistema",
      "observations": [
        "11 tools MCP: 9 Anthropic-compat + 2 nuevos",
        "Stack: FastMCP 3.1.1 + sqlite-vec + ONNX"
      ],
      "distance": 0.1234
    },
    {
      "name": "Skill: memoria-sofia",
      "entityType": "Componente",
      "observations": [
        "Fase 1 Diagnóstico: Equipo de Análisis"
      ],
      "distance": 0.3591
    }
  ]
}
```

Error (modelo no disponible):

```json
{
  "error": "Embedding model not available. Run 'python scripts/download_model.py' to download the model first."
}
```

**Notas**:
- **Requiere** que el modelo ONNX esté descargado. Si no lo está, retorna un error descriptivo.
- Usa **KNN search** con cosine distance sobre la tabla virtual `vec0` de sqlite-vec.
- El campo `distance` es la **distancia coseno**, redondeada a 4 decimales.
- **Menor distance = mayor similitud**. La fórmula es `d = 1 - cos(A, B)`, con rango `[0, 2]`: `0.0` = idénticos, `1.0` = ortogonales, `2.0` = opuestos.
- El texto que se codifica para cada entidad es el **snapshot completo**: nombre + tipo + todas las observaciones.
- **Limbic re-ranking**: after KNN retrieval, candidates are re-ranked using salience, temporal decay, and co-occurrence signals. The API output format remains identical — the `distance` field reflects the original cosine distance, while the internal ordering is driven by the composite limbic score. See [Sistema Límbico — Scoring Dinámico](#sistema-límbico--scoring-dinámico) for details.
- **Post-response tracking**: after returning results, the tool records access events and co-occurrences for the top-K entities to improve future rankings. This is best-effort and does not affect the response.

---

#### 11. `migrate`

**Descripción**: Migrate data from Anthropic MCP Memory JSONL format to SQLite. This is idempotent — running it multiple times won't duplicate data.

**Firma**: `migrate(source_path: str = "") → dict[str, Any]`

**Parámetros**:

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `source_path` | `str` | Sí | `""` | Ruta al archivo JSONL de Anthropic de origen. Debe existir. |

**Respuesta**:

```json
{
  "entities_imported": 32,
  "relations_imported": 37,
  "errors": 0,
  "skipped": 2
}
```

**Notas**:
- **Idempotente**: ejecutar múltiples veces no duplica datos. Las entidades se upsertan, las relaciones se crean solo si no existen previamente, y las observaciones duplicadas se descartan.
- Las relaciones se importan **solo si ambas entidades ya existen** en el grafo al momento de procesar la línea.
- **Generación batch de embeddings**: si el motor está disponible, al finalizar la migración se generan embeddings para todas las entidades importadas.

---

### Resumen de comportamiento de embeddings por operación

| Operación | Genera/Actualiza embedding | Detalle |
|-----------|:--------------------------:|---------|
| `create_entities` | ✅ Sí | Snapshot completo (nombre + tipo + todas las observaciones). |
| `add_observations` | ✅ Sí | Regenera con las observaciones actualizadas. |
| `delete_observations` | ✅ Sí | Regenera sin las observaciones eliminadas. |
| `delete_entities` | 🗑️ Elimina | Eliminación manual antes del CASCADE (vec0 no lo soporta). |
| `create_relations` | ❌ No | Las relaciones no participan en la búsqueda semántica. |
| `delete_relations` | ❌ No | Ídem. |
| `search_nodes` | ❌ No | Búsqueda LIKE, no requiere embeddings. |
| `open_nodes` | ❌ No | Lectura directa por nombre exacto. |
| `read_graph` | ❌ No | Dump completo, operación de solo lectura. |
| `search_semantic` | 📖 Usa (lectura) | Codifica el query, busca por cosine distance, re-rankea con Limbic Scoring. Registra access + co-occurrences post-response. |
| `migrate` | ✅ Sí (batch) | Genera embeddings para todas las entidades importadas al final. |

---

## Sistema de Embeddings

### Visión General

MCP Memory v2 incorpora búsqueda semántica por similitud vectorial. El sistema convierte cada entidad del knowledge graph en un vector numérico de 384 dimensiones y lo almacena en una tabla virtual sqlite-vec. Cuando un agente ejecuta `search_semantic`, el motor codifica la consulta en un vector del mismo espacio, busca los *k* vecinos más cercanos (KNN) y retorna las entidades más relevantes.

La pila técnica es:

| Componente | Tecnología | Rol |
|---|---|---|
| Modelo de inferencia | ONNX Runtime (CPU) | Codifica texto → vector |
| Modelo de lenguaje | intfloat/multilingual-e5-small | Sentence embeddings multilingüe (retrieval asimétrico) |
| Tokenización | HuggingFace Tokenizers (Rust) | Tokenización rápida con padding/truncation |
| Almacenamiento | sqlite-vec (vec0) | Búsqueda KNN con cosine distance |
| Álgebra | NumPy float32 | Operaciones vectoriales |

### Modelo

El modelo base es **`intfloat/multilingual-e5-small`** de [IntFloat](https://huggingface.co/intfloat) (familia E5):

- **Dimensionalidad**: 384 floats (float32, ONNX FP32)
- **Idiomas**: 94+ (multilingüe) — funciona nativamente con consultas en español, inglés y otros idiomas
- **Optimización**: diseñado para inferencia en CPU; no requiere GPU
- **Tamaño ONNX**: ~465 MB (modelo exportado)
- **Distancia**: cosine similarity (los vectores se normalizan L2)
- **Tipo**: retrieval asimétrico — requiere prefijos `"query: "` y `"passage: "` según la tarea

El modelo se exporta a ONNX y se almacena localmente junto con los archivos del tokenizer. La exportación se realiza una vez mediante el script `scripts/download_model.py`.

### Arquitectura del Motor

```
Texto de entrada
       │
       ▼
┌──────────────────┐
│  Tokenizer (HF)  │  enable_truncation(max_length=512)
│  fast tokenizer   │  enable_padding(length=512)
└────────┬─────────┘
         │  input_ids, attention_mask
         ▼
┌──────────────────┐
│  ONNX Inference  │  InferenceSession (CPUExecutionProvider)
│  Sentence-BERT   │  graph_optimization_level = ORT_ENABLE_ALL
└────────┬─────────┘
         │  token_embeddings: (batch, seq_len, 384)
         ▼
┌──────────────────┐
│   Mean Pooling   │  mask PAD tokens con attention_mask
└────────┬─────────┘
         │  mean_embeddings: (batch, 384)
         ▼
┌──────────────────┐
│  L2 Normalize   │  → unit vector (cosine-ready)
└────────┬─────────┘
         │
         ▼
   float32[384]
```

### Pipeline de Encoding

La clase `EmbeddingEngine.encode()` implementa el pipeline en cinco pasos:

```python
def encode(self, texts: list[str], task: str = "passage") -> np.ndarray:
    """Encode texts to embeddings.
    task: "query" prepends "query: " prefix, "passage" prepends "passage: ".
    """
    prefix = self.QUERY_PREFIX if task == "query" else self.PASSAGE_PREFIX
    prefixed = [f"{prefix}{t}" for t in texts]
    # ... tokenization → ONNX → mean pooling → L2 normalize
```

Los prefijos `"query: "` y `"passage: "` son un **requisito del modelo e5** para retrieval asimétrico. Las consultas de `search_semantic` usan `task="query"`, mientras que las entidades se codifican con `task="passage"` (default).

#### Paso 1 — Tokenización

```python
encoded = self._tokenizer.encode_batch(texts)

input_ids = np.array(
    [e.ids for e in encoded],
    dtype=np.int64,
)
attention_mask = np.array(
    [e.attention_mask for e in encoded],
    dtype=np.int64,
)
```

Se usa el **HuggingFace fast tokenizer** (implementación en Rust) con dos configuraciones fijas:

- `enable_truncation(max_length=512)`: trunca secuencias más largas a 512 tokens
- `enable_padding(length=512)`: rellena secuencias más cortas con `[PAD]` hasta 512 tokens

#### Paso 2 — Forward ONNX

```python
feed: dict[str, np.ndarray] = {}
for name in self._input_names:
    if name == "input_ids":
        feed[name] = input_ids
    elif name == "attention_mask":
        feed[name] = attention_mask
    elif name == "token_type_ids":
        feed[name] = np.zeros_like(input_ids)
    else:
        logger.warning("Unknown ONNX input '%s' — filling with zeros", name)
        feed[name] = np.zeros_like(input_ids)

outputs = self._session.run(None, feed)
token_embeddings = outputs[0]  # (batch, seq_len, 384)
```

Los nombres de los inputs se descubren dinámicamente del grafo ONNX (`self._session.get_inputs()`), lo que hace el código robusto ante variaciones menores del modelo exportado.

#### Paso 3 — Mean Pooling

```python
mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
mean_embeddings = sum_embeddings / sum_mask
```

Se promedian los embeddings de todos los tokens **reales** (no `[PAD]`). El *attention mask* se expande a 3D para multiplicar elemento a elemento contra `token_embeddings`, anulando la contribución de los tokens de padding.

#### Paso 4 — Normalización L2

```python
norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
norms = np.clip(norms, a_min=1e-9, a_max=None)
normalized = mean_embeddings / norms

return normalized.astype(np.float32)
```

La normalización L2 convierte cada vector en un *unit vector* (norma = 1). Esto permite usar **dot product** como proxy de **cosine similarity**, que es el `distance_metric=cosine` que usa sqlite-vec internamente.

### Formato de Texto

Antes de codificar una entidad, se construye un texto representativo usando la estrategia **Head+Tail+Diversity** con un budget de 480 tokens:

```python
MAX_TOKENS = 480

@staticmethod
def prepare_entity_text(
    name: str,
    entity_type: str,
    observations: list[str],
    relations: list[str] | None = None,
) -> str:
    # Head: primeras observaciones (si caben)
    # Tail: últimas observaciones (si no caben en head)
    # Diversity: observaciones intermedias seleccionadas para maximizar variedad
    # Separator: " | " entre observaciones
    # Relations: "Rel: type → target; ..." appended si existen
```

Formato: `"{name} ({entity_type}) | {obs1} | {obs2} | ... | Rel: tipo → destino; ..."`

Ejemplo:

```
MCP Memory v2 (Tarea) | 8 tareas secuenciadas: T1 → T2 → T3 | Pipeline: Arquitecto → Constructor → Auditor | Rel: usa → FastMCP; usa → SQLite; Rel: contiene → scoring.py
```

**Estrategia Head+Tail+Diversity**:
- **Head**: primeras observaciones (contenido más importante/estable)
- **Tail**: últimas observaciones (contenido más reciente)
- **Diversity**: observaciones intermedias seleccionadas para maximizar variedad semántica
- **Budget**: 480 tokens máximo (MAX_TOKENS=480), con `" | "` como separador
- **Relaciones**: contexto de relaciones se agrega al final con formato `"Rel: type → target; ..."` cuando existen

**Punto clave**: el texto se genera con un *snapshot seleccionado* de las observaciones (no necesariamente todas). Cada vez que las observaciones de una entidad cambian, el embedding se **regenera completamente** — no es incremental.

### Almacenamiento y Serialización

Los vectores se almacenan en la tabla virtual sqlite-vec `entity_embeddings`:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS entity_embeddings
USING vec0(embedding float[384] distance_metric=cosine);
```

- **entity_id = rowid**: sqlite-vec usa el `rowid` implícito como identificador
- **distance_metric=cosine**: distancia angular (0 = idéntico, 2 = opuesto)
- **Serialización**: float32 → raw bytes vía `struct.pack` (384 floats × 4 bytes = 1 536 bytes)
- **Deserialización**: bytes → float32 vía `np.frombuffer`

Funciones de serialización:

```python
def serialize_f32(vector: np.ndarray) -> bytes:
    """Pack a float32 vector into raw bytes for sqlite-vec.
    A 384-dim vector → 1 536 bytes."""
    return struct.pack(f"{len(vector)}f", *vector.flatten())


def deserialize_f32(data: bytes, dim: int = 384) -> np.ndarray:
    """Unpack raw bytes from sqlite-vec back into a float32 vector."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
```

El almacenamiento usa `INSERT OR REPLACE`, así que re-almacenar un embedding para la misma entidad sobrescribe el anterior sin crear duplicados.

### Patrón Singleton y Lazy Load

```python
class EmbeddingEngine:
    _instance: "EmbeddingEngine | None" = None

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
```

La carga diferida funciona en dos niveles:

1. **Nivel del singleton**: la instancia se crea solo cuando alguien llama a `get_instance()` por primera vez
2. **Nivel del módulo**: en `server.py`, la importación de `mcp_memory.embeddings` es lazy (dentro de `_get_engine()`), no en el ámbito del módulo

Si los archivos del modelo (`model.onnx` + `tokenizer.json`) no están presentes en `~/.cache/mcp-memory-v2/models/`, el motor se marca como `available=False`.

#### Archivos del modelo

```
~/.cache/mcp-memory-v2/models/
├── model.onnx                # Modelo ONNX exportado (~465 MB)
├── tokenizer.json            # HuggingFace fast tokenizer
├── tokenizer_config.json     # Configuración del tokenizer
└── special_tokens_map.json   # Mapeo de tokens especiales
```

### Búsqueda KNN

```
query (texto libre)
  │
  ▼
engine.encode([query], task="query") → float32[384] (L2-normalizado)
  │
  ▼
serialize_f32(vector)           → 1536 bytes
  │
  ▼
sqlite-vec: WHERE embedding MATCH ? ORDER BY distance LIMIT N
  │
  ▼
[entity_id, distance] pairs     → get_entity_by_id() + get_observations()
  │
  ▼
resultado: [{name, entityType, observations, distance, limbic_score, scoring}, ...]
```

Los resultados se ordenan por distancia ascendente (los más similares primero). La distancia es cosine distance (0 = idéntico). El límite por defecto es 10, configurable vía el parámetro `limit`.

---

## Sistema Límbico — Scoring Dinámico

### Qué es

El Sistema Límbico es una capa de scoring dinámico que se ejecuta **sobre los resultados KNN** de `search_semantic`, mejorando el ranking y exponiendo métricas de scoring. El nombre viene de la metáfora biológica: así como el sistema límbico del cerebro asigna valencia emocional, facilita el olvido y fortalece asociaciones, este sistema asigna importancia (salience), aplica decaimiento temporal y detecta co-ocurrencias entre entidades.

La API de `search_semantic` extiende el formato de salida para incluir scoring límbico. Además del formato original `{name, entityType, observations, distance}`, cada resultado ahora incluye `limbic_score` (score compuesto) y `scoring` (desglose de componentes).

### Las 3 capacidades

| Capacidad | Qué hace | Metáfora biológica |
|---|---|---|
| **Salience** | Entidades muy accedidas y bien conectadas suben en el ranking | Valencia emocional — lo importante se recuerda mejor |
| **Decaimiento temporal** | Entidades no accedidas recientemente bajan gradualmente | Olvido — lo que no se usa se desvanece |
| **Co-ocurrencia** | Entidades que aparecen juntas frecuentemente se refuerzan mutuamente | Asociación — lo que se activa junto se vincula |

### Fórmula de scoring

```
score(e, q) = cosine_sim(q, e) × (1 + β_sal × importance(e)) × temporal_factor(e) × (1 + γ · cooc_boost(e, R))
```

Donde:

- `cosine_sim(q, e) = max(0, 1 - distance)` — la similitud coseno pura del KNN
- `importance(e)` — importancia estructural de la entidad (ver sub-fórmula)
- `temporal_factor(e)` — factor de decaimiento temporal (ver sub-fórmula)
- `cooc_boost(e, R)` — refuerzo por co-ocurrencia con otros resultados (ver sub-fórmula)

#### Sub-fórmula: importance(e)

```
importance(e) = [log₂(1 + access_count) / log₂(1 + max_access)] × (1 + β_deg × min(degree, D_max) / D_max)
```

Combina dos señales:
- **Access frequency**: normalizada con logaritmo y dividida entre el máximo del set de candidatos (rango `[0, 1]`). Una entidad accedida 10 veces cuando el máximo es 20 no recibe la mitad, sino `log₂(11)/log₂(21) ≈ 0.77`.
- **Graph degree**: cuántas relaciones tiene la entidad, normalizada y capped en `D_MAX`. Las entidades centrales del grafo son más relevantes.

#### Sub-fórmula: temporal_factor(e)

```
temporal_factor(e) = max(TEMPORAL_FLOOR, exp(-LAMBDA_HOURLY × Δt_hours))
```

- `Δt_hours` = horas desde el último acceso (o desde `created_at` si nunca se accedió)
- `LAMBDA_HOURLY = 0.0001` → half-life ≈ 290 días (`ln(2)/0.0001 ≈ 6931 horas`)
- `TEMPORAL_FLOOR = 0.1` → el decaimiento nunca baja de 0.1 (el conocimiento se degrada pero no se destruye)
- Una entidad accedida hace 1 hora: factor ≈ 0.9999. Hace 30 días: factor ≈ 0.928. Hace 1 año: factor ≈ 0.407

#### Sub-fórmula: cooc_boost(e, R)

```
cooc_boost(e, R) = Σ_{r ∈ R, r ≠ e} log₂(1 + co_count(e, r))
```

Suma logarítmica de co-ocurrencias entre la entidad `e` y cada otra entidad `r` en el set de candidatos. El logaritmo suaviza la contribución: 10 co-ocurrencias no es 10× mejor que 1, sino `log₂(11)/log₂(2) ≈ 3.46×`.

### Constantes tuneables

| Constante | Valor | Propósito |
|---|---|---|
| `BETA_SAL` | `0.5` | Peso del boost por salience (importance). Un valor de 0.5 significa que una entidad con importance=1.0 recibe un boost de 1.5× sobre la similitud pura |
| `BETA_DEG` | `0.15` | Peso del degree dentro de la fórmula de importance. Bajo a propósito — el grado es una señal secundaria |
| `D_MAX` | `15` | Cap de relaciones para normalizar degree. Entidades con >15 relaciones no reciben boost adicional |
| `LAMBDA_HOURLY` | `0.0001` | Tasa de decaimiento temporal por hora. Half-life ≈ 290 días (`ln(2)/0.0001 ≈ 6931 horas`) |
| `GAMMA` | `0.1` | Peso del co-occurrence boost. Un valor de 0.1 significa que 5 co-ocurrencias con log₂ suman ~0.24× de boost |
| `EXPANSION_FACTOR` | `3` | Factor de sobre-recuperación KNN. Si `limit=10`, se recuperan 30 candidatos para re-ranking |
| `TEMPORAL_FLOOR` | `0.1` | Piso mínimo del decaimiento temporal. El conocimiento se degrada pero nunca cae por debajo de este valor |

### Schema: tablas nuevas

Las tablas `entity_access` y `co_occurrences` ya están documentadas en la sección [Modelo de Datos](#modelo-de-datos). Su DDL:

```sql
CREATE TABLE entity_access (
    entity_id    INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    access_count INTEGER NOT NULL DEFAULT 1,
    last_access  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE co_occurrences (
    entity_a_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    co_count     INTEGER NOT NULL DEFAULT 1,
    last_co      TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (entity_a_id, entity_b_id)
);
```

### Flujo completo: encode → re-rank → track

```
1. Encode query     → engine.encode([query], task="query") → float[384]
2. KNN 3×           → search_embeddings(query, limit × 3) → [{entity_id, distance}]
3. Fetch metadata   → get_access_data(), get_entity_degrees(), get_co_occurrences()
4. Re-rank          → rank_candidates() → compute limbic_score per candidate → sort → top-K
5. Build output     → {name, entityType, observations, distance, limbic_score, scoring}
6. Record signals   → record_access(top-K ids) + record_co_occurrences(top-K ids) (search + open_nodes)
```

El paso 6 (grabación de señales) ocurre después de construir la respuesta, es *best-effort*, y no afecta el resultado retornado.

### Módulo: scoring.py

El motor de scoring vive en `src/mcp_memory/scoring.py` (~179 líneas). Expone:

| Función | Propósito |
|---|---|
| `rank_candidates()` | Entry point principal — recibe resultados KNN + metadata, retorna top-K re-rankeados |
| `compute_importance()` | Calcula importance(e) a partir de access_count, max_access y degree |
| `compute_temporal_factor()` | Calcula temporal_factor(e) a partir de last_access / created_at |
| `compute_cooc_boost()` | Calcula cooc_boost(e, R) a partir del mapa de co-ocurrencias |

Las constantes están a nivel de módulo y son directamente editables.

### Transparencia de API

El Sistema Límbico extiende la API de `search_semantic` de forma backward-compatible:

- **Formato de entrada**: `search_semantic(query, limit)` — sin cambios
- **Formato de salida**: cada resultado incluye campos nuevos además de los originales:
  ```json
  {
    "results": [{
      "name": "...",
      "entityType": "...",
      "observations": ["..."],
      "distance": 0.42,
      "limbic_score": 0.67,
      "scoring": {
        "importance": 0.85,
        "temporal_factor": 0.99,
        "cooc_boost": 1.23
      }
    }]
  }
  ```
- **Campo `distance`**: sigue siendo la distancia coseno original del KNN (no el limbic score)
- **Campo `limbic_score`**: score compuesto que determina el orden de los resultados
- **Campo `scoring`**: desglose de los tres componentes del limbic score
- **Orden**: por limbic score descendente en vez de distancia coseno ascendente
- **Co-occurrence tracking**: se registra tanto en `search_semantic` como en `open_nodes` (cuando se abren 2+ entidades juntos)

---

## Migración desde Anthropic

### Propósito

La tool `migrate` permite importar datos existentes desde el formato JSONL que usa el servidor Anthropic MCP Memory original. Esto facilita la transición sin pérdida de datos.

### Formato JSONL

El archivo de entrada es JSONL (una línea JSON por registro). Existen dos tipos de registro:

**Entidad**:

```json
{"type": "entity", "name": "Sesión 2026-03-21", "entityType": "Sesion", "observations": ["Decisión: construir MCP Memory v2"]}
```

**Relación**:

```json
{"type": "relation", "from": "MCP Memory v2", "to": "FastMCP", "relationType": "usa"}
```

Campos obligatorios:

| Tipo | Campo | Descripción |
|---|---|---|
| entity | `type` | `"entity"` |
| entity | `name` | Nombre único de la entidad |
| entity | `entityType` | Tipo de la entidad (default: `"Generic"`) |
| entity | `observations` | Lista de strings (puede estar vacía) |
| relation | `type` | `"relation"` |
| relation | `from` | Nombre de la entidad origen |
| relation | `to` | Nombre de la entidad destino |
| relation | `relationType` | Tipo de relación |

### Proceso de Migración

El proceso se ejecuta en cuatro fases secuenciales:

1. **Lectura y parseo**: el archivo se lee línea por línea. Cada línea se parsea como JSON; si falla, se registra un warning y se continúa.
2. **Importación de entidades**: cada entidad se inserta o actualiza (upsert) por nombre. Las observaciones se mergen: solo se añaden las que no existan ya.
3. **Importación de relaciones**: las relaciones solo se crean si **ambas entidades** existen. Si falta alguna, la relación se salta.
4. **Generación de embeddings en batch**: si el motor está disponible, se generan embeddings para todas las entidades importadas al final.

### Idempotencia

La migración es **idempotente**: ejecutarla múltiples veces no duplica datos.

| Operación | Mecanismo de idempotencia |
|---|---|
| Insertar entidad | `upsert_entity` usa `ON CONFLICT(name) DO UPDATE` |
| Añadir observación | `add_observations` consulta existencia antes de insertar |
| Crear relación | `create_relation` captura `IntegrityError` (constraint `UNIQUE`) |
| Almacenar embedding | `INSERT OR REPLACE` sobre `rowid` sobrescribe si existe |

### Tratamiento de errores

La migración está diseñada para ser tolerante a fallos:

- **Líneas corruptas**: se saltan con un warning (se cuentan como `errors`)
- **Entidades sin nombre**: se saltan (se cuentan como `skipped`)
- **Relaciones con campos faltantes**: se saltan (se cuentan como `skipped`)
- **Relaciones con entidades inexistentes**: se saltan (se cuentan como `skipped`)
- **Fallos en generación de embeddings individuales**: se registran como warning, no detienen la migración

### Resultado

```json
{
    "entities_imported": 32,
    "relations_imported": 37,
    "errors": 0,
    "skipped": 2
}
```

- `entities_imported`: registros tipo `"entity"` procesados exitosamente
- `relations_imported`: relaciones efectivamente creadas
- `errors`: líneas que fallaron al parsear JSON o al procesar
- `skipped`: registros omitidos por campos faltantes, entidades no encontradas o tipo desconocido

---

## Consideraciones y Gotchas

### vec0 no soporta CASCADE

Las tablas virtuales de sqlite-vec **no participan** en las cláusulas `ON DELETE CASCADE` de SQLite. Al eliminar una entidad, sus observaciones y relaciones se borran por CASCADE, pero **no** su embedding.

El método `delete_entities_by_names` resuelve esto internamente:

```python
# 1. Delete embeddings (vec0 has no CASCADE support)
self.db.execute(
    f"DELETE FROM entity_embeddings WHERE rowid IN ({id_placeholders})", ids
)

# 2. Delete entities (CASCADE takes care of observations & relations)
self.db.execute(
    f"DELETE FROM entities WHERE id IN ({id_placeholders})", ids
)
```

Si eliminas entidades directamente con SQL manual, asegúrate de limpiar los embeddings primero para evitar registros huérfanos.

### Error "cannot start a transaction"

sqlite-vec puede fallar intermitentemente con `"cannot start a transaction"` durante operaciones de eliminación. Es un **bug conocido** de sqlite-vec relacionado con cómo las tablas virtuales interactúan con el sistema de transacciones de SQLite.

En la práctica, reintentar la operación suele resolverlo. El código captura este error y lo registra como warning sin fallar.

### search_semantic requiere modelo descargado

La tool `search_semantic` es la única que necesita el motor de embeddings. Si el modelo ONNX no está descargado, la tool retorna un error claro. Las **10 herramientas restantes** funcionan correctamente sin el modelo.

### Lazy load del motor de embeddings

El servidor MCP arranca en ~1 segundo porque **no carga el modelo al inicio**. La arquitectura lazy tiene dos capas:

1. **Import lazy**: en `server.py`, el módulo `mcp_memory.embeddings` no se importa en el ámbito del módulo. La importación ocurre dentro de `_get_engine()`.
2. **Instanciación lazy**: `EmbeddingEngine.get_instance()` crea el singleton solo en la primera llamada.

Consecuencia práctica:

- **Primera llamada a `search_semantic`**: tarda ~3-5 segundos adicionales mientras carga el modelo
- **Llamadas subsiguientes**: responden en milisegundos (el motor ya está en memoria)
- **Arranque del servidor**: siempre rápido, independientemente de si el modelo está descargado

### WAL mode y concurrencia

SQLite se configura con **WAL mode** y un timeout de busy waiting de 5 segundos:

| Operación | Comportamiento |
|---|---|
| Lecturas concurrentes | Permitidas (WAL permite múltiples readers simultáneos) |
| Escrituras | Secuenciales (single writer) |
| Lock contention | Los lectores esperan hasta 5s (`busy_timeout`) por un lock de escritura |
| Cache | 64 MB en memoria para reducir I/O |

En el contexto de MCP, donde las llamadas a tools son secuenciales, este modelo es adecuado.

### Embeddings no incrementales

Cada vez que las observaciones de una entidad cambian, el embedding se **regenera completamente** desde cero. El texto de entrada incluye un *snapshot completo* de todas las observaciones actuales.

**Implicaciones**:

- **Consistencia**: el embedding siempre refleja el estado actual, sin artefactos de actualizaciones parciales
- **Costo**: cada actualización dispara una codificación ONNX completa (~5ms en CPU para un vector individual)
- **Sobrescritura**: `INSERT OR REPLACE` en vec0 asegura que no se acumulan versiones antiguas
