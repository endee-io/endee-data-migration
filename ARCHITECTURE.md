# System Architecture — Endee Migration Tool

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENDEE MIGRATION TOOL                                │
│                                                                             │
│   CLI (migrate.py)  ──►  MigrationPipeline  ──►  Endee Vector DB           │
│        │                       │                                            │
│        │              ┌────────┴────────┐                                   │
│        │              │                 │                                   │
│        ▼              ▼                 ▼                                   │
│   .env / args      Producer          Consumer                               │
│   (config)      (BaseSource)      (BaseTarget)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              CONFIGURATION LAYER                            ║
║                                                                             ║
║  .env file / CLI args                                                       ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  ║
║  │   Source Config  │  │  Target Config   │  │    Pipeline Config       │  ║
║  │                  │  │                  │  │                          │  ║
║  │ SOURCE_URL       │  │ TARGET_URL       │  │ BATCH_SIZE      = 1000   │  ║
║  │ SOURCE_API_KEY   │  │ TARGET_API_KEY   │  │ UPSERT_SIZE     = 100    │  ║
║  │ SOURCE_COLLECTION│  │ TARGET_COLLECTION│  │ MAX_QUEUE_SIZE  = 5      │  ║
║  │ SOURCE_PORT      │  │ SPACE_TYPE       │  │ CHECKPOINT_FILE          │  ║
║  │ SOURCE_TYPE      │  │ TARGET_TYPE      │  │ RESUME                   │  ║
║  │   dense|hybrid   │  │   dense|hybrid   │  │                          │  ║
║  │ SPARSE_TEXT_FIELD│  │ SPARSE_MODEL     │  └──────────────────────────┘  ║
║  │ SPARSE_ALGO      │  │ M, EF_CONSTRUCT  │                                ║
║  │ FROM_DB          │  │ PRECISION        │                                ║
║  └──────────────────┘  └──────────────────┘                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ENTRYPOINT LAYER                               ║
║                                                                             ║
║  migrate.py                                                                 ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │  1. Parse args + env vars                                              │ ║
║  │  2. Validate type combinations (startup — before any connections)      │ ║
║  │     ├─ dense  → hybrid  requires SPARSE_ALGO                          │ ║
║  │     ├─ hybrid → dense   → error (unsupported)                         │ ║
║  │     ├─ SPARSE_ALGO + qdrant/milvus requires SPARSE_TEXT_FIELD         │ ║
║  │     └─ SPARSE_ALGO + target_type=dense  → error                       │ ║
║  │  3. Registry lookup                                                    │ ║
║  │     SOURCE_REGISTRY[(from_db, source_type)] → SourceClass             │ ║
║  │     TARGET_REGISTRY[(to_db,   target_type)] → TargetClass             │ ║
║  │  4. Instantiate source, target, checkpoint, pipeline                  │ ║
║  │  5. pipeline.run()                                                     │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                             ║
║  SOURCE_REGISTRY                     TARGET_REGISTRY                       ║
║  ┌──────────────────────────────┐    ┌──────────────────────────────────┐  ║
║  │ (milvus, dense)  → Milvus   │    │ (endee, dense)  → EndeeTarget   │  ║
║  │ (milvus, hybrid) → Milvus   │    │ (endee, hybrid) → EndeeTarget   │  ║
║  │ (qdrant, dense)  → Qdrant   │    └──────────────────────────────────┘  ║
║  │ (qdrant, hybrid) → Qdrant   │                                          ║
║  │ (chroma, dense)  → Chroma   │                                          ║
║  └──────────────────────────────┘                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
╔═══════════════════════╗  ╔═══════════════╗  ╔═══════════════════════════════╗
║     SOURCE LAYER      ║  ║  CORE LAYER   ║  ║       TARGET LAYER           ║
╚═══════════════════════╝  ╚═══════════════╝  ╚═══════════════════════════════╝
```

---

## Source Layer

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                               SOURCE LAYER                                  ║
║                                                                             ║
║   BaseSource (abstract — core/base_source.py)                               ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  connect()          establish DB connection                          │  ║
║   │  detect_schema()    inspect collection → build RowSchema             │  ║
║   │  iterate_batches()  async generator → yields (rows, cursor, timing) │  ║
║   │  close()            optional teardown                                │  ║
║   │  from_args()        factory — reads CLI args namespace               │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
║                    │                                                         ║
║       ┌────────────┼────────────┐                                           ║
║       ▼            ▼            ▼                                           ║
║  ┌──────────┐ ┌──────────┐ ┌──────────────────────────────────────────┐    ║
║  │  Milvus  │ │  Qdrant  │ │              ChromaDenseSource           │    ║
║  │          │ │          │ │                                          │    ║
║  │ Dense    │ │ Dense    │ │  • HttpClient or PersistentClient        │    ║
║  │ Source   │ │ Source   │ │  • Integer offset cursor                 │    ║
║  │          │ │          │ │  • dense-only OR dense→hybrid            │    ║
║  │ Hybrid   │ │ Hybrid   │ │    via SPARSE_ALGO                      │    ║
║  │ Source   │ │ Source   │ │  • uses native `documents` for text     │    ║
║  └──────────┘ └──────────┘ └──────────────────────────────────────────┘    ║
║                                                                             ║
║  Milvus specifics           Qdrant specifics                                ║
║  ┌──────────────────────┐   ┌────────────────────────────────────────────┐  ║
║  │ • QueryIterator      │   │ • scroll API with UUID cursor              │  ║
║  │ • int offset cursor  │   │ • named vectors: "dense", "sparse_keywords"│  ║
║  │ • HNSW from index    │   │ • HNSW from collection config             │  ║
║  │   params (M, ef_con) │   │ • Payload stored as arbitrary JSON keys   │  ║
║  │ • Payload = fields   │   └────────────────────────────────────────────┘  ║
║  └──────────────────────┘                                                   ║
║                                                                             ║
║  Dense→Hybrid path (Milvus / Qdrant)                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  SPARSE_ALGO set?                                                    │   ║
║  │  YES → load encoder via SparseEncoderFactory                        │   ║
║  │       → SPARSE_TEXT_FIELD required (which payload key has the text)  │   ║
║  │       → detect_schema() appends SPARSE_VECTOR slot to RowSchema     │   ║
║  │       → _convert_records() batch-encodes texts → fills sparse slot  │   ║
║  │  NO  → dense-only RowSchema                                          │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Core / Pipeline Layer

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           CORE / PIPELINE LAYER                             ║
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                     MigrationPipeline.run()                         │    ║
║  │                                                                     │    ║
║  │  1. source.connect()                                                │    ║
║  │  2. target.connect()                                                │    ║
║  │  3. schema = source.detect_schema()   ◄── RowSchema built here      │    ║
║  │  4. target.setup_index(schema)        ◄── index created here        │    ║
║  │  5. asyncio.run(_run_async())                                       │    ║
║  │     ├── asyncio.Queue(maxsize=MAX_QUEUE_SIZE)  ← back-pressure      │    ║
║  │     ├── _producer(queue)  ─────────────────────────────────────┐   │    ║
║  │     └── _consumer(queue)  ─────────────────────────────────────┘   │    ║
║  │  6. source.close() / target.close()                                 │    ║
║  │  7. print report                                                    │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
║                                                                             ║
║   PRODUCER (async)                         CONSUMER (async)                ║
║   ┌──────────────────────────┐             ┌──────────────────────────┐    ║
║   │                          │             │                          │    ║
║   │  source.iterate_batches()│             │  queue.get()             │    ║
║   │          │               │             │       │                  │    ║
║   │          ▼               │             │       ▼                  │    ║
║   │  (rows, cursor, timing)  │             │  target.upsert_batch()   │    ║
║   │          │               │             │       │                  │    ║
║   │          ▼               │  bounded    │       ▼                  │    ║
║   │    queue.put(batch) ─────┼────queue───►│  checkpoint.update()     │    ║
║   │                          │  max=5      │       │                  │    ║
║   │  puts None sentinel      │             │       ▼                  │    ║
║   │  when source exhausted   │             │  tqdm progress bar       │    ║
║   └──────────────────────────┘             └──────────────────────────┘    ║
║                                                                             ║
║                                                                             ║
║   SCHEMA — the shared contract                                              ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  RowSchema                                                           │  ║
║   │  ├── fields: List[FieldSchema]  ← ordered slot definitions          │  ║
║   │  │   ├── SLOT 0  name="id"         type=STRING        role=ID       │  ║
║   │  │   ├── SLOT 1  name="embedding"  type=DENSE_VECTOR  role=DENSE    │  ║
║   │  │   ├── SLOT 2  name="sparse_vec" type=SPARSE_VECTOR role=SPARSE   │  ║
║   │  │   └── SLOT N  name="payload"    type=JSON          role=METADATA  │  ║
║   │  ├── dimension           int    — dense vector size                 │  ║
║   │  ├── space_type          str    — "cosine" | "l2" | "ip"           │  ║
║   │  ├── is_hybrid           bool   — True if SPARSE_VECTOR slot exists │  ║
║   │  └── canonical_precision str    — "float32" | "float16" | ...      │  ║
║   │                                                                     │  ║
║   │  MigrationRow(arity)                                                │  ║
║   │  └── fields: List[Any]  ← positional values, no names inside       │  ║
║   │      set_field(pos, value)   ← source writes                       │  ║
║   │      get_field(pos)          ← target reads                        │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
║                                                                             ║
║   CHECKPOINT                                                                ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  migration_checkpoint.json                                          │  ║
║   │  ├── processed_count  int   — total records upserted so far        │  ║
║   │  ├── last_offset      Any   — opaque cursor (int/UUID/etc.)        │  ║
║   │  ├── batch_number     int   — last completed batch                 │  ║
║   │  └── completed        bool  — True = source fully drained          │  ║
║   │                                                                     │  ║
║   │  Written after EVERY successfully upserted batch.                  │  ║
║   │  On resume, pipeline skips already-processed records.              │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Target Layer

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                               TARGET LAYER                                  ║
║                                                                             ║
║   BaseTarget (abstract — core/base_target.py)                               ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  connect()        establish connection to target DB                  │  ║
║   │  setup_index()    create/get index from RowSchema                   │  ║
║   │  upsert_batch()   async — write one batch, return (bool, timing)    │  ║
║   │  close()          optional teardown                                  │  ║
║   │  from_args()      factory — reads CLI args namespace                │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
║                              │                                              ║
║                              ▼                                              ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │                        EndeeTarget                                   │  ║
║   │                                                                      │  ║
║   │  setup_index(schema)                                                 │  ║
║   │  ├── resolve slot positions from FieldRole (ID/DENSE/SPARSE/META)  │  ║
║   │  ├── validate filter_fields against metadata field names           │  ║
║   │  ├── check precision rank (no upgrade allowed)                     │  ║
║   │  ├── target_type == "hybrid"  → create_index(sparse_model=...)     │  ║
║   │  └── target_type == "dense"   → create_index(no sparse params)     │  ║
║   │                                                                      │  ║
║   │  upsert_batch(records, schema)                                       │  ║
║   │  ├── _to_endee() per record                                         │  ║
║   │  │   ├── id     = row[pk_slot]                                      │  ║
║   │  │   ├── vector = row[dense_slot]                                   │  ║
║   │  │   ├── sparse_indices/values = row[sparse_slot]  (hybrid only)   │  ║
║   │  │   └── payload split into filter{} and meta{}                    │  ║
║   │  │       (filter_fields controls which keys go to filter)           │  ║
║   │  ├── chunk into upsert_chunk_size pieces (default 100)             │  ║
║   │  ├── asyncio.gather() — all chunks in parallel                     │  ║
║   │  └── exponential backoff retry — 3 attempts (1s → 2s → 4s)        │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
║                              │                                              ║
║                              ▼                                              ║
║              ┌──────────────────────────────────┐                          ║
║              │           Endee API               │                          ║
║              │  POST /api/v1/indexes/{name}/upsert│                         ║
║              │  Dense:  {id, vector, filter, meta}│                         ║
║              │  Hybrid: {id, vector,              │                         ║
║              │           sparse_indices,          │                         ║
║              │           sparse_values,           │                         ║
║              │           filter, meta}            │                         ║
║              └──────────────────────────────────┘                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Sparse Encoder Layer

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           SPARSE ENCODER LAYER                              ║
║              (only active when SPARSE_ALGO is set in env)                   ║
║                                                                             ║
║  BaseSparseEncoder (interface_sparse_encoder.py)                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  encode(text: str)              → {"indices": [...], "values": [...]}│   ║
║  │  encode_batch(texts: List[str]) → List[{"indices", "values"}]        │   ║
║  │  build_sparse_field()           → FieldSchema(SPARSE_VECTOR role)    │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                              │                                              ║
║              ┌───────────────┴──────────────────┐                          ║
║              ▼                                   ▼                          ║
║   ┌─────────────────────┐            ┌─────────────────────────────────┐   ║
║   │     EndeeBM25       │            │    (future encoders here)       │   ║
║   │                     │            │                                 │   ║
║   │ algo key:           │            │  Add to:                        │   ║
║   │  "endee/bm25"       │            │  1. concrete_sparse_encoders.py │   ║
║   │                     │            │  2. factory_sparse_encoder.py   │   ║
║   │ backend:            │            │  3. --sparse_algo choices in    │   ║
║   │  endee-model        │            │     migrate.py                  │   ║
║   │  SparseModel        │            │                                 │   ║
║   │  ("endee/bm25")     │            │  No source/target changes needed│   ║
║   └─────────────────────┘            └─────────────────────────────────┘   ║
║                                                                             ║
║   SparseEncoderFactory (factory_sparse_encoder.py)                          ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  _REGISTRY: {"endee/bm25": EndeeBM25, ...}                          │  ║
║   │  create(algorithm)  → BaseSparseEncoder instance                    │  ║
║   │  register(key, cls) → add new encoder without touching source code  │  ║
║   └──────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow — Dense Migration

```
  Source DB                  Pipeline                     Endee
  (Milvus/Qdrant/Chroma)                                  Index
       │                         │                          │
       │  connect()              │                          │
       │◄────────────────────────│                          │
       │  detect_schema()        │                          │
       │──────────────────────── ► RowSchema ──────────────►│  setup_index()
       │                         │                          │
       │  iterate_batches()      │                          │
       │ ── batch 0 ──────────► queue ──────────────────── ►│  upsert_batch()
       │ ── batch 1 ──────────► queue ──────────────────── ►│  upsert_batch()
       │ ── batch N ──────────► queue ──────────────────── ►│  upsert_batch()
       │  (cursor saved to checkpoint after each batch)      │
       │                         │                          │
       │  None sentinel          │                          │
       │ ─────────────────────► queue (done)                │
```

---

## Data Flow — Dense → Hybrid Migration (with Sparse Algo)

```
  Source DB            SparseEncoderFactory         Pipeline           Endee
  (dense vectors          EndeeBM25                                    Hybrid
   + text payload)                                                      Index
       │                       │                       │                 │
       │  connect()            │                       │                 │
       │◄──────────────────────────────────────────────│                 │
       │  detect_schema()      │                       │                 │
       │──────────────────────────────────────────────►│                 │
       │  SPARSE_ALGO set?     │                       │                 │
       │  YES → load encoder   │                       │                 │
       │──────────────────────►│                       │                 │
       │                       │  encoder ready        │                 │
       │                       │◄──────────────────────│                 │
       │  schema: dense + sparse slots                  │                 │
       │──────────────────────────────────────────────►│ setup_index()  │
       │                                                │────────────────►│
       │  iterate_batches()    │                        │                 │
       │  fetch records        │                        │                 │
       │──────────────────────►│ encode_batch(texts)    │                 │
       │                       │────────────────────────│                 │
       │                       │  sparse vecs           │                 │
       │  MigrationRows (dense + sparse slots filled)   │                 │
       │──────────────────────────────────────────────►│ queue           │
       │                                               │──────────────── ►│
       │                                                │  upsert_batch() │
```

---

## Supported Migration Combinations

```
  SOURCE_TYPE │ TARGET_TYPE │ SPARSE_ALGO  │ SPARSE_TEXT_FIELD │ Status
 ─────────────┼─────────────┼──────────────┼───────────────────┼─────────────────
  dense       │ dense       │ not set      │ not needed        │  supported
  dense       │ hybrid      │ endee/bm25   │ required*         │  supported
  hybrid      │ hybrid      │ not set      │ not needed        │  supported
  hybrid      │ dense       │ any          │ any               │  blocked at start
  dense       │ hybrid      │ not set      │ any               │  blocked at start
  any         │ any         │ set          │ not set*          │  blocked at start
 ─────────────┴─────────────┴──────────────┴───────────────────┴─────────────────

  * SPARSE_TEXT_FIELD required for milvus/qdrant; NOT needed for chroma
    (chroma uses its native `documents` field automatically)
```

---

## File Layout

```
  endee-data-migration/
  │
  ├── migrate.py                      ← CLI entrypoint, registry, validation
  ├── constants.py                    ← shared constants (API paths, defaults)
  ├── .env                            ← runtime configuration
  │
  ├── core/
  │   ├── base_source.py              ← BaseSource abstract class
  │   ├── base_target.py              ← BaseTarget abstract class
  │   ├── pipeline.py                 ← MigrationPipeline (producer-consumer)
  │   ├── checkpoint.py               ← MigrationCheckpoint (resume support)
  │   ├── schema.py                   ← RowSchema, MigrationRow, FieldSchema
  │   ├── record.py                   ← (legacy) MigrationRecord, IndexConfig
  │   └── type_registry.py            ← space/precision mappings + rank table
  │
  ├── sources/
  │   ├── milvus_source.py            ← MilvusDenseSource, MilvusHybridSource
  │   ├── qdrant_source.py            ← QdrantDenseSource, QdrantHybridSource
  │   └── chroma_source.py            ← ChromaDenseSource
  │
  ├── targets/
  │   └── endee_target.py             ← EndeeTarget (dense + hybrid)
  │
  └── sparse_encoders/
      ├── interface_sparse_encoder.py ← BaseSparseEncoder abstract class
      ├── concrete_sparse_encoders.py ← EndeeBM25
      └── factory_sparse_encoder.py   ← SparseEncoderFactory registry
```
