# Low-Level Design — Endee Migration Tool

## 1. System Overview

The migration tool moves vector data from a source DB (Milvus, Qdrant, ChromaDB) into an Endee index.
It uses an **async producer-consumer pipeline** with a **canonical row format** to fully decouple sources from targets.

```
┌─────────────┐     RowSchema      ┌──────────────┐     MigrationRow     ┌──────────────┐
│   Source DB │ ──────────────────▶│   Pipeline   │ ────────────────────▶│  Endee Index │
│ (Milvus /   │   iterate_batches  │  (Producer / │   upsert_batch       │  (dense or   │
│  Qdrant /   │                    │   Consumer)  │                      │   hybrid)    │
│  ChromaDB)  │                    └──────────────┘                      └──────────────┘
└─────────────┘
```

---

## 2. Entry Point — migrate.py

```
migrate.py
│
├── _build_parser()
│   ├── Migration axes   : --from, --to, --source_type, --target_type
│   ├── Source args      : --source_url, --source_api_key, --source_collection, --source_port, --source_db, --filter_fields, --use_https
│   ├── Target args      : --target_url, --target_api_key, --target_collection
│   ├── Index config     : --space_type, --M, --ef_construct, --precision
│   ├── Performance      : --batch_size, --upsert_size, --max_queue_size
│   ├── Checkpoint       : --checkpoint_file, --resume
│   └── Sparse encoding  : --sparse_algo, --sparse_model, --sparse_text_field
│
├── SOURCE_REGISTRY  (from_db, source_type) → SourceClass
│   ├── ("milvus", "dense")  → MilvusDenseSource
│   ├── ("milvus", "hybrid") → MilvusHybridSource
│   ├── ("qdrant", "dense")  → QdrantDenseSource
│   ├── ("qdrant", "hybrid") → QdrantHybridSource
│   └── ("chroma", "dense")  → ChromaDenseSource
│
├── TARGET_REGISTRY  (to_db, target_type) → TargetClass
│   ├── ("endee", "dense")   → EndeeTarget
│   └── ("endee", "hybrid")  → EndeeTarget
│
└── main()
    ├── Validation rules
    │   ├── sparse_algo + target_type=dense           → ERROR
    │   ├── source_type=dense + target_type=hybrid + no sparse_algo → ERROR
    │   ├── source_type=hybrid + target_type=dense    → ERROR
    │   └── sparse_algo + from_db=qdrant/milvus + no sparse_text_field → ERROR
    │
    ├── _build_source(args) → source instance
    ├── _build_target(args) → target instance
    ├── MigrationCheckpoint(checkpoint_file)
    └── MigrationPipeline(source, target, checkpoint, batch_size, queue_size)
```

---

## 3. Core Abstractions

### 3.1 BaseSource (core/base_source.py)

```
BaseSource  (ABC)
│
├── connect()                                     abstract — establish DB connection
├── detect_schema() → RowSchema                   abstract — inspect collection, build canonical schema
├── iterate_batches(batch_size, cursor, schema)   abstract — async generator
│   yields: (List[MigrationRow], next_cursor, {"fetch": float, "src_transform": float})
├── close()                                       optional
└── from_args(args)                               classmethod factory
```

### 3.2 BaseTarget (core/base_target.py)

```
BaseTarget  (ABC)
│
├── connect()                                     abstract — establish DB connection
├── setup_index(schema: RowSchema)                abstract — get or create target index
├── upsert_batch(records, schema) → (bool, dict)  abstract — write batch, never raises
├── close()                                       optional
└── from_args(args)                               classmethod factory
```

### 3.3 Canonical Data Model (core/schema.py)

```
FieldType (Enum)
├── STRING | INT | FLOAT | BOOL
├── DENSE_VECTOR    — fixed-dim float list
├── SPARSE_VECTOR   — {indices, values} dict
└── JSON            — arbitrary metadata dict

FieldRole (Enum)
├── ID              — unique record identifier
├── DENSE_VECTOR    — ANN-indexed vector
├── SPARSE_VECTOR   — hybrid search sparse weights
└── METADATA        — everything else (filter/meta split done by target)

FieldSchema
├── name:       str
├── field_type: FieldType
├── role:       FieldRole
└── dimension:  Optional[int]    — only for DENSE_VECTOR

RowSchema
├── fields:               List[FieldSchema]
├── space_type:           str               — "cosine" | "l2" | "ip"
├── dimension:            int
├── is_hybrid:            bool
├── canonical_precision:  str               — "float32" | "float16" | "int16" | "int8" | "binary"
│
├── get_primary_key()     → Optional[FieldSchema]
├── get_dense_vector()    → Optional[FieldSchema]
├── get_sparse_vector()   → Optional[FieldSchema]
├── get_metadata_fields() → List[FieldSchema]
├── index_of(name)        → int   (-1 if not found)
└── require_index_of(name)→ int   (raises if missing)

MigrationRow
├── fields: List[Any]   — positional slot list, no names
├── set_field(pos, value)
├── get_field(pos) → Any
└── arity → int
```

---

## 4. Pipeline (core/pipeline.py)

```
MigrationPipeline
│
├── run()
│   ├── source.connect()
│   ├── target.connect()
│   ├── schema = source.detect_schema()       ← RowSchema built once here
│   ├── target.setup_index(schema)            ← index created/verified here
│   └── asyncio.run(_run_async())
│       └── asyncio.gather(
│               _producer(queue),
│               _consumer(queue, pbar)
│           )
│
├── _producer(queue)
│   └── async for rows, cursor, timings in source.iterate_batches():
│       └── queue.put({batch_number, rows, cursor, src_timings, enqueue_time})
│           [blocks when queue full — back-pressure]
│
├── _consumer(queue, pbar)
│   └── loop:
│       ├── batch = queue.get()
│       ├── success, tgt_timings = target.upsert_batch(rows, schema)
│       ├── if success  → checkpoint.update(batch_num, count, cursor)
│       └── if not success → stop_event.set(); break
│
└── _print_report()
    ├── MIGRATION COMPLETED SUCCESSFULLY  (failed == 0)
    ├── MIGRATION COMPLETED WITH ERRORS   (failed > 0)
    ├── MIGRATION INTERRUPTED             (SIGINT/SIGTERM)
    └── MIGRATION FAILED                  (producer exception)

Queue
└── Bounded asyncio.Queue(maxsize=max_queue_size)
    ├── Producer blocks on put() when full     → back-pressure
    └── Sentinel value None signals consumer to stop
```

---

## 5. Source Connectors

### 5.1 Milvus

```
MilvusBaseSource  (BaseSource)
│
├── __init__(url, token, collection, db, port)
├── connect()          — MilvusClient(uri, token, db_name), auto-fixes protocol
├── _load_collection() — load_collection() + poll get_load_state() every 5s (timeout 300s)
│
├── detect_schema() → RowSchema
│   ├── describe_collection() → walk fields
│   ├── SLOT 0:    ID         (PK field)
│   ├── SLOT 1:    DENSE_VECTOR (FLOAT_VECTOR | FLOAT16_VECTOR | BINARY_VECTOR)
│   ├── SLOT 2:    SPARSE_VECTOR (SPARSE_FLOAT_VECTOR, if present)
│   ├── LAST SLOT: JSON payload (all remaining fields bundled)
│   ├── Reads metric_type from describe_index() → resolve_space()
│   └── Reads field dtype → resolve_precision()
│
├── _decode_vector(raw, field_type) → List[float]
│   ├── bytes + FLOAT16 → np.frombuffer(float16).astype(float32)
│   └── bytes + FLOAT32 → np.frombuffer(float32)
│
├── _convert_records(milvus_records) → (List[MigrationRow], float)
│   ├── SLOT 0: str(pk_field)
│   ├── SLOT 1: _decode_vector(dense_field)
│   ├── SLOT 2: {indices, values} from sparse_field (if hybrid)
│   └── LAST:  {meta_field: value, ...}  — all non-vector non-pk fields
│
├── iterate_batches(batch_size, cursor, schema)
│   ├── Creates QueryIterator (no 16384 offset cap)
│   ├── Skips cursor records on resume
│   └── Yields (rows, count_cursor, timings)
│
└── _validate_schema(sparse_fields)  — override in subclass

MilvusDenseSource  (MilvusBaseSource)
│
├── __init__(..., sparse_algo, sparse_text_field)
├── connect()          — super().connect() + load encoder if sparse_algo set
│
├── _validate_schema(sparse_fields)
│   └── ERROR if sparse_fields present (use MilvusHybridSource)
│
├── detect_schema()    — super() + if sparse_algo:
│   ├── peek at 1 record via query() to verify sparse_text_field exists
│   ├── append SPARSE_VECTOR slot to schema.fields
│   └── schema.is_hybrid = True
│
├── _convert_records() — if sparse_algo:
│   ├── extract texts from rec[sparse_text_field]
│   ├── ERROR if all texts empty
│   ├── encoder.encode_batch(texts) → sparse_embs
│   └── fill sparse slot per row
│
└── from_args(args)

MilvusHybridSource  (MilvusBaseSource)
│
├── _validate_schema(sparse_fields)
│   └── ERROR if no sparse_fields (use MilvusDenseSource)
│
└── from_args(args)
```

### 5.2 Qdrant

```
QdrantBaseSource  (BaseSource)
│
├── __init__(url, collection, api_key, port, use_https)
├── connect()          — QdrantClient(url, port, api_key, https)
│
├── detect_schema() → RowSchema
│   ├── get_collection() → read vectors_config + sparse_vectors config
│   ├── SLOT 0:    ID
│   ├── SLOT 1:    DENSE_VECTOR (named or unnamed)
│   ├── SLOT 2:    SPARSE_VECTOR (if sparse_vectors config present)
│   ├── LAST SLOT: JSON payload (pt.payload dict — everything)
│   ├── Distance enum → resolve_space()
│   └── _extract_qdrant_precision_key(qcfg) → resolve_precision()
│       ├── Handles: none, scalar(int8), binary, turbo(bits4/bits2/bits1.5/bits1)
│       └── ERROR for product quantization (unsupported)
│
├── _convert_records(points) → (List[MigrationRow], float)
│   ├── SLOT 0: str(pt.id)
│   ├── SLOT 1: dense vector (named or unnamed from pt.vector)
│   ├── SLOT 2: {indices, values} from sparse (if hybrid)
│   └── LAST:  pt.payload dict
│
├── iterate_batches(batch_size, cursor, schema)
│   ├── scroll(collection, limit, offset=cursor, with_payload=True, with_vectors=True)
│   ├── 5-attempt retry with exponential backoff (2^attempt seconds)
│   └── Yields (rows, next_uuid_cursor, timings)
│
└── _validate_schema(sparse_vectors)  — override in subclass

QdrantDenseSource  (QdrantBaseSource)
│
├── __init__(..., sparse_algo, sparse_text_field)
├── connect()          — super() + load encoder if sparse_algo set
├── _validate_schema() — ERROR if has_sparse (use QdrantHybridSource)
│
├── detect_schema()    — super() + if sparse_algo:
│   ├── scroll(limit=1) to verify sparse_text_field in payload
│   ├── append SPARSE_VECTOR slot
│   └── schema.is_hybrid = True
│
├── _convert_records() — if sparse_algo:
│   ├── texts = [pt.payload.get(sparse_text_field) for pt in points]
│   ├── ERROR if all empty
│   ├── encoder.encode_batch(texts)
│   └── fill sparse slot per row
│
└── from_args(args)

QdrantHybridSource  (QdrantBaseSource)
│
├── _validate_schema() — ERROR if no sparse_vectors (use QdrantDenseSource)
└── from_args(args)
```

### 5.3 ChromaDB

```
ChromaDenseSource  (BaseSource)
│
├── __init__(url, collection, api_key, source_path,
│            store_document_in_meta, sparse_algo, canonical_precision)
│
├── connect()
│   ├── PersistentClient(path) if source_path set
│   ├── HttpClient(host, port, headers) otherwise
│   └── if sparse_algo: _load_encoder() + _validate_documents_exist()
│
├── _load_encoder()
│   └── SparseEncoderFactory.create(sparse_algo)
│
├── _validate_documents_exist()
│   └── get(limit=5, include=["documents"]) → ERROR if no documents
│
├── detect_schema() → RowSchema
│   ├── get(limit=1, include=["embeddings"]) → detect dimension
│   ├── SLOT 0:    ID
│   ├── SLOT 1:    DENSE_VECTOR (dim auto-detected)
│   ├── SLOT 2:    SPARSE_VECTOR (only if sparse_algo set)
│   └── LAST SLOT: JSON payload (metadata + optional document text)
│
├── _convert_records(result) → (List[MigrationRow], float)
│   ├── SLOT 0: str(id)
│   ├── SLOT 1: list(embedding[i])
│   ├── SLOT 2: {indices, values} from _encode_sparse(documents) if sparse
│   └── LAST:  {**metadata, document: text} if store_document_in_meta
│
├── _encode_sparse(documents, count) → list[dict]
│   ├── ERROR if no documents or all empty
│   └── encoder.encode_batch(doc_texts)
│
├── iterate_batches(batch_size, cursor, schema)
│   ├── get(limit, offset=cursor, include=[embeddings, metadatas, documents])
│   ├── 5-attempt retry, 60s timeout
│   └── Yields (rows, next_int_cursor, timings)
│
└── from_args(args)
    └── canonical_precision = args.precision or "float32"
```

---

## 6. Target Connector

### EndeeTarget (targets/endee_target.py)

```
EndeeTarget  (BaseTarget)
│
├── __init__(endee_url, endee_api_key, index_name, upsert_chunk_size,
│            sparse_model, filter_fields, space_type, M, ef_construct,
│            precision, target_type)
│
├── connect()
│   ├── Endee(token=api_key)
│   ├── client.set_base_url(url + ENDEE_V1_API) if url provided
│   └── client.list_indexes()  — smoke-test connectivity
│
├── setup_index(schema: RowSchema)
│   ├── Validate filter_fields against schema metadata field names
│   ├── Precision check:
│   │   ├── endee_rank > source_rank → ERROR (upgrade not allowed)
│   │   └── endee_rank < source_rank → WARNING (downgrade detected)
│   ├── Resolve slot positions from RowSchema roles:
│   │   ├── _pk_slot     = index_of(primary_key.name)
│   │   ├── _dense_slot  = index_of(dense_vector.name)
│   │   ├── _sparse_slot = index_of(sparse_vector.name)  if hybrid
│   │   └── _payload_slots = [index_of(f.name) for f in metadata_fields]
│   ├── client.get_index(index_name) → return if exists
│   └── client.create_index(name, dimension, space_type, M, ef_con, precision,
│                            [sparse_model if target_type=hybrid])
│
├── _to_endee(record, schema) → dict
│   ├── {"id": str(pk_slot)}
│   ├── {"vector": dense_slot}
│   ├── {"sparse_indices": ..., "sparse_values": ...}  if sparse_slot >= 0
│   └── {"filter": {...}, "meta": {...}}
│       └── payload split: field in filter_fields → filter, else → meta
│
├── upsert_batch(records, schema) → (bool, timings)
│   ├── [records → _to_endee()] → endee_records
│   ├── Split into chunks of upsert_chunk_size
│   ├── asyncio.gather(*[_upsert_chunk(c) for c in chunks])
│   ├── Retry failed chunks: exponential backoff 1s → 2s → 4s (max 3 attempts)
│   └── Return (False, timings) if chunk fails after all retries
│
└── from_args(args)
```

---

## 7. Sparse Encoding Subsystem

```
BaseSparseEncoder  (ABC)
├── build_sparse_field() → FieldSchema     abstract
├── encode(text) → {"indices": [...], "values": [...]}   abstract
└── encode_batch(texts) → [{"indices", "values"}, ...]   optional (default: loop encode)

SparseEncoderFactory
├── _REGISTRY: dict[str, type]
├── register(key, encoder_cls)
├── create(algorithm, **kwargs) → BaseSparseEncoder
│   └── _REGISTRY[algorithm.lower()](**kwargs)
└── Registered at import time:
    └── "endee/bm25" → EndeeBM25

EndeeBM25  (BaseSparseEncoder)
├── __init__(model_name="endee/bm25")
│   └── SparseModel(model_name=model_name)   ← endee-model SDK
├── build_sparse_field() → FieldSchema(name="sparse_vector", SPARSE_VECTOR, SPARSE_VECTOR)
├── encode(text) → wrap in list → model.embed([text])[0] → {indices, values}
└── encode_batch(texts) → model.embed(texts) → [{indices, values}, ...]
```

**Adding a new encoder (no source/target changes needed):**
```
1. concrete_sparse_encoders.py  — implement class MyEncoder(BaseSparseEncoder)
2. factory_sparse_encoder.py    — SparseEncoderFactory.register("my/algo", MyEncoder)
3. migrate.py                   — add "my/algo" to choices=[] on --sparse_algo arg
```

---

## 8. Checkpoint (core/checkpoint.py)

```
MigrationCheckpoint
│
├── __init__(filepath)
│   └── _load()  — reads JSON file, returns defaults if missing
│
├── State stored in JSON:
│   ├── processed_count  — total records successfully upserted (cumulative)
│   ├── last_offset      — opaque source cursor (int / UUID / None)
│   ├── batch_number     — last completed batch
│   └── completed        — True when source exhausted
│
├── update(batch_number, records_count, cursor)
│   └── processed_count += records_count; last_offset = cursor; _save()
├── mark_completed()     → completed = True; _save()
├── clear()              → reset all to defaults; _save()
├── get_processed_count()
├── get_last_cursor()
├── get_batch_number()
└── is_completed()
```

---

## 9. Type Registry (core/type_registry.py)

```
Canonical Space Constants
├── SPACE_COSINE = "cosine"
├── SPACE_L2     = "l2"
├── SPACE_IP     = "ip"
└── SPACE_MANHATTAN = "manhattan"

Canonical Precision Constants + Rank
├── PRECISION_FLOAT32 = "float32"   rank 4
├── PRECISION_FLOAT16 = "float16"   rank 3
├── PRECISION_INT16   = "int16"     rank 2
├── PRECISION_INT8    = "int8"      rank 1
└── PRECISION_BINARY  = "binary"    rank 0

Source → Canonical Mappings
├── MILVUS_PRECISION_MAPPING          DataType → canonical
├── MILVUS_TO_CANONICAL_SPACE_MAPPING metric_type → canonical
├── CHROMADB_PRECISION_MAPPING        chroma dtype → canonical
├── QDRANT_TO_CANONICAL_PRECISION_MAPPING  quant_config_key → canonical
│   (covers: none, scalar, int8, binary, turbo:bits4/2/1.5/1, binary variants)
└── QDRANT_TO_CANONICAL_SPACE_TYPE    Distance string → canonical

Canonical → Target Mappings
├── CANONICAL_TO_ENDEE_PRECISION_MAPPING  canonical → Endee Precision enum
└── CANONICAL_TO_ENDEE_SPACE_MAPPING      canonical → Endee space string

Helper Functions
├── resolve_space(mapping, raw)      → canonical string  (sys.exit on unknown)
└── resolve_precision(mapping, raw)  → canonical string  (sys.exit on unknown)
```

---

## 10. End-to-End Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  migrate.py main()                                                           │
│                                                                              │
│  1. Parse args + env vars                                                    │
│  2. Validate source/target type combos                                       │
│  3. source  = SOURCE_REGISTRY[(from_db, source_type)].from_args(args)       │
│  4. target  = TARGET_REGISTRY[(to_db, target_type)].from_args(args)         │
│  5. checkpoint = MigrationCheckpoint(file)                                   │
│  6. pipeline.run()                                                           │
└────────────────────────┬─────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MigrationPipeline.run()                                                     │
│                                                                              │
│  source.connect()   ──────────────────────────────────────────────────────▶ │
│  target.connect()   ──────────────────────────────────────────────────────▶ │
│                                                                              │
│  schema = source.detect_schema()                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RowSchema                                                          │    │
│  │  fields: [ID slot, DENSE_VECTOR slot, SPARSE_VECTOR? slot, PAYLOAD] │    │
│  │  space_type, dimension, is_hybrid, canonical_precision              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  target.setup_index(schema)  → get or create Endee index                    │
│                                                                              │
│  asyncio.gather(producer, consumer)                                          │
│                                                                              │
│  ┌─────────────┐   queue.put(batch)   ┌──────────────────────────────────┐  │
│  │  _producer  │ ──────────────────▶  │  _consumer                       │  │
│  │             │                      │                                  │  │
│  │  source     │   bounded Queue      │  target.upsert_batch(rows,schema)│  │
│  │  .iterate_  │   (max_queue_size)   │  → convert → chunk → parallel   │  │
│  │  batches()  │   ◀── back-pressure  │    upsert → retry on fail        │  │
│  │             │                      │  checkpoint.update() on success  │  │
│  └─────────────┘                      └──────────────────────────────────┘  │
│                                                                              │
│  source.close(); target.close()                                              │
│  _print_report()                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Supported Migration Combinations

| FROM_DB | SOURCE_TYPE | TARGET_TYPE | SPARSE_ALGO | SPARSE_TEXT_FIELD | Result                              |
|---------|-------------|-------------|-------------|-------------------|-------------------------------------|
| milvus  | dense       | dense       | —           | —                 | Dense → Dense                       |
| milvus  | dense       | hybrid      | endee/bm25  | text_field_name   | Dense + generated sparse → Hybrid   |
| milvus  | hybrid      | hybrid      | —           | —                 | Hybrid → Hybrid                     |
| qdrant  | dense       | dense       | —           | —                 | Dense → Dense                       |
| qdrant  | dense       | hybrid      | endee/bm25  | text_field_name   | Dense + generated sparse → Hybrid   |
| qdrant  | hybrid      | hybrid      | —           | —                 | Hybrid → Hybrid                     |
| chroma  | dense       | dense       | —           | —                 | Dense → Dense                       |
| chroma  | dense       | hybrid      | endee/bm25  | —                 | Dense + generated sparse → Hybrid   |
| any     | hybrid      | dense       | —           | —                 | ERROR — cannot drop sparse          |
| any     | dense       | hybrid      | —           | —                 | ERROR — SPARSE_ALGO required        |
