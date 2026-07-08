# Migration Tool — Architecture, LLD & UML Diagrams

---

## 1. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          VECTOR MIGRATION TOOL                                  │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────┐    ┌───────────────┐ │
│  │  .env / CLI  │───▶│           migrate.py             │───▶│  checkpoint   │ │
│  │   (config)   │    │  ┌─────────────────────────────┐ │    │  .json (disk) │ │
│  └──────────────┘    │  │ SOURCE_REGISTRY              │ │    └───────────────┘ │
│                      │  │  (milvus/dense)  → MilvusDense│ │                    │
│                      │  │  (milvus/hybrid) → MilvusHybrid│ │                   │
│                      │  │  (qdrant/dense)  → QdrantDense │ │                   │
│                      │  │  (qdrant/hybrid) → QdrantHybrid│ │                   │
│                      │  │  (chroma/dense)  → ChromaDense │ │                   │
│                      │  └─────────────────────────────┘ │    │                 │
│                      │  ┌─────────────────────────────┐ │    │                 │
│                      │  │ TARGET_REGISTRY              │ │    │                 │
│                      │  │  (endee/dense)  → EndeeTarget │ │                   │
│                      │  │  (endee/hybrid) → EndeeTarget │ │                   │
│                      │  └─────────────────────────────┘ │    │                 │
│                      └──────────────────────────────────┘    │                 │
│                                       │                       │                 │
│                                       ▼                       │                 │
│                      ┌──────────────────────────────────┐    │                 │
│                      │         MigrationPipeline         │◀───┘                │
│                      │                                   │                     │
│                      │  ┌────────────┐  asyncio.Queue   │                     │
│                      │  │ _producer  │ ────────────────▶│                     │
│                      │  │            │  (bounded, max=5) │                     │
│                      │  └────────────┘                   │                     │
│                      │        │                           │                     │
│                      │        ▼                           │                     │
│                      │  source.iterate_batches()          │                     │
│                      │                                   │                     │
│                      │  ┌────────────┐                   │                     │
│                      │  │ _consumer  │◀─────────────────▶│                     │
│                      │  │            │  target.upsert()   │                     │
│                      │  └────────────┘                   │                     │
│                      └──────────────────────────────────┘                      │
│                            │                    │                               │
│                            ▼                    ▼                               │
│              ┌─────────────────────┐  ┌─────────────────┐                      │
│              │   SOURCE (one of)   │  │  TARGET         │                      │
│              │                     │  │                 │                      │
│              │  Milvus  (19530)    │  │  Endee          │                      │
│              │  Qdrant  (6333)     │  │  (HTTP API)     │                      │
│              │  ChromaDB(8000)     │  │                 │                      │
│              └─────────────────────┘  └─────────────────┘                      │
│                            │                                                    │
│                  (optional) ▼                                                   │
│              ┌──────────────────────┐                                           │
│              │   SparseEncoder      │                                           │
│              │   (EndeeBM25)        │  ← only when dense→hybrid migration       │
│              │   TF-IDF on text     │                                           │
│              └──────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DATA FLOW — END TO END

```
  .env / CLI args
       │
       ▼
  migrate.py
  ├── Validate type combination (dense/hybrid)
  ├── Build source = SOURCE_REGISTRY[(from_db, source_type)].from_args(args)
  ├── Build target = TARGET_REGISTRY[(to_db, target_type)].from_args(args)
  └── MigrationPipeline(source, target, checkpoint).run()
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
   source.connect()  target.connect()  load checkpoint
          │
          ▼
   schema = source.detect_schema()
   ┌────────────────────────────────────────────────────┐
   │  RowSchema                                         │
   │  ┌──────┬───────────────┬──────┬────────────────┐ │
   │  │ slot │ name          │ role │ type           │ │
   │  ├──────┼───────────────┼──────┼────────────────┤ │
   │  │  0   │ id            │  PK  │ STRING         │ │
   │  │  1   │ embedding     │DENSE │ DENSE_VECTOR   │ │
   │  │  2   │ sparse_vector │SPARSE│ SPARSE_VECTOR  │ │ ← hybrid only
   │  │  N   │ payload       │META  │ JSON           │ │
   │  └──────┴───────────────┴──────┴────────────────┘ │
   │  dimension=1536  space=cosine  precision=float32   │
   └────────────────────────────────────────────────────┘
          │
          ▼
   target.setup_index(schema)
   ├── Validate precision (must be float32 from source)
   ├── Resolve slot positions from FieldRole
   ├── Get or create Endee index
   └── Cache pk_slot, dense_slot, sparse_slot, payload_slots
          │
          ▼
   ┌──────────────────────────────────────────────────────────┐
   │               Async Producer-Consumer Loop               │
   │                                                          │
   │  _producer ─────────────────────────────────────────┐   │
   │                                                      │   │
   │  source.iterate_batches(batch_size, cursor, schema)  │   │
   │       │                                              │   │
   │       │  [Milvus]  QueryIterator.next()              │   │
   │       │  [Qdrant]  client.scroll(offset=cursor)      │   │
   │       │  [Chroma]  collection.get(offset=cursor)     │   │
   │       │                                              │   │
   │       ▼                                              │   │
   │  _convert_records(raw_batch)                         │   │
   │       │                                              │   │
   │       │  For each record:                            │   │
   │       │    row[0] = str(id)                          │   │
   │       │    row[1] = vector (passed through as-is)    │   │
   │       │    row[2] = sparse (stored or generated)     │   │ ← if hybrid
   │       │    row[N] = {all other fields}               │   │
   │       │                                              │   │
   │       ▼                                              │   │
   │  [optional] encode_batch(texts) → sparse vectors     │   │ ← dense→hybrid
   │       │                                              │   │
   │       ▼                                              │   │
   │  queue.put({rows, next_cursor, timings}) ────────────┘   │
   │                                          asyncio.Queue   │
   │  _consumer ◀─────────────────────────────────────────    │
   │                                                          │
   │  target.upsert_batch(rows, schema)                       │
   │       │                                                  │
   │       ▼                                                  │
   │  _to_endee(row) → {"id", "vector",                       │
   │                     "sparse_indices", "sparse_values",   │
   │                     "filter": {...}, "meta": {...}}       │
   │       │                                                  │
   │       ▼                                                  │
   │  chunk into upsert_chunk_size pieces                     │
   │       │                                                  │
   │       ▼                                                  │
   │  asyncio.gather(upsert(chunk1), upsert(chunk2), ...)     │
   │       │                                                  │
   │       ├── success → checkpoint.update(batch, count, cur) │
   │       └── failure → retry 3× (1s→2s→4s backoff)         │
   │                     if all fail: stop pipeline           │
   └──────────────────────────────────────────────────────────┘
          │
          ▼
   source exhausted → checkpoint.mark_completed()
   print_report() → duration, records, throughput
```

---

## 3. LOW LEVEL DESIGN (LLD)

### 3a. MigrationRow & RowSchema (Core Data Model)

```
  RowSchema                              MigrationRow
  ┌─────────────────────────────────┐    ┌──────────────────────────────┐
  │ fields: List[FieldSchema]       │    │ fields: List[Any]            │
  │ dimension: int                  │    │                              │
  │ space_type: str                 │    │  [0] "doc_42"      ← PK     │
  │ is_hybrid: bool                 │    │  [1] [0.1,0.2,...] ← DENSE  │
  │ canonical_precision: str        │    │  [2] {indices,vals}← SPARSE │
  │                                 │    │  [3] {"text": ...} ← META   │
  │ get_primary_key() → FieldSchema │    │                              │
  │ get_dense_vector() → FieldSchema│    │ set_field(pos, val)          │
  │ get_sparse_vector() → FieldSchema    │ get_field(pos) → Any         │
  │ get_metadata_fields() → [...]   │    │ arity: int                   │
  │ index_of(name) → int            │    └──────────────────────────────┘
  └─────────────────────────────────┘

  FieldSchema
  ┌──────────────────────────────────────────────────────┐
  │ name: str          → "embedding"                     │
  │ field_type: FieldType                                │
  │   STRING | INT | FLOAT | BOOL | DENSE_VECTOR         │
  │   SPARSE_VECTOR | JSON                               │
  │ role: FieldRole                                      │
  │   ID | DENSE_VECTOR | SPARSE_VECTOR | METADATA       │
  │ dimension: Optional[int]  → 1536                     │
  └──────────────────────────────────────────────────────┘
```

### 3b. Source Connector Internals

```
  MilvusBaseSource
  ┌─────────────────────────────────────────────────────────────┐
  │ State                                                       │
  │   milvus_client          : MilvusClient                     │
  │   _schema                : RowSchema                        │
  │   _dense_slot            : int = 1                          │
  │   _sparse_slot           : int = -1 (or 2)                  │
  │   _payload_slot          : int = N                          │
  │   _dense_field_name      : str                              │
  │   _sparse_field_name     : str | None                       │
  │   _meta_field_names      : List[str]                        │
  │                                                             │
  │ detect_schema()                                             │
  │   _load_collection()                                        │
  │   describe_collection()                                     │
  │   for field in fields:                                      │
  │     if is_pk         → slot 0 (ID)                         │
  │     if FLOAT_VECTOR,                                        │
  │        FLOAT16_VECTOR,                                      │
  │        BFLOAT16_VECTOR,                                     │
  │        BINARY_VECTOR,                                       │
  │        INT8_VECTOR   → slot 1 (DENSE)                      │
  │                        precision = MILVUS_TO_WIRE_PRECISION │
  │                          FLOAT_VECTOR → float32             │
  │                          others      → raw_binary           │
  │     if SPARSE_FLOAT_VECTOR → slot 2 (SPARSE)               │
  │     else            → bundled into payload slot N           │
  │                                                             │
  │ iterate_batches(batch_size, initial_cursor, schema)         │
  │   iterator = query_iterator(collection, filter="", all)     │
  │   skip initial_cursor records if resuming                   │
  │   while True:                                               │
  │     batch = iterator.next()   [via executor]                │
  │     if empty: return                                        │
  │     yield _convert_records(batch), cursor, timings          │
  │                                                             │
  │ _convert_records(records)                                   │
  │   for rec in records:                                       │
  │     row[0] = str(rec[id_field])                             │
  │     row[1] = rec[dense_field]   ← passed through, no decode│
  │     row[2] = rec[sparse_field]  ← if hybrid                 │
  │     row[N] = {k: rec[k] for k in meta_field_names}         │
  └─────────────────────────────────────────────────────────────┘

  QdrantBaseSource
  ┌─────────────────────────────────────────────────────────────┐
  │ State                                                       │
  │   client             : QdrantClient                         │
  │   _dense_vector_name : str | None  (None = unnamed)         │
  │   _sparse_vector_name: str | None                           │
  │                                                             │
  │ detect_schema()                                             │
  │   collection_info = client.get_collection()                 │
  │   vectors_config:                                           │
  │     if dict → named vectors → pick first float vector       │
  │     if VectorParams → unnamed → always "vector"             │
  │   quantization: _extract_qdrant_quant_key()                 │
  │     → "none" | "scalar" | "binary" | "product" | "turbo"   │
  │     all map to float32 (Qdrant decompresses on scroll)      │
  │   sparse_vectors_config → detect sparse field if present    │
  │                                                             │
  │ iterate_batches(batch_size, initial_cursor, schema)         │
  │   offset = initial_cursor or None                           │
  │   while True:                                               │
  │     points, next_offset = scroll(offset, batch_size)        │
  │     yield _convert_records(points), next_offset, timings    │
  │     if next_offset is None: return                          │
  └─────────────────────────────────────────────────────────────┘

  ChromaDenseSource
  ┌─────────────────────────────────────────────────────────────┐
  │ State                                                       │
  │   client      : HttpClient | PersistentClient               │
  │   collection  : Collection                                  │
  │   _encoder    : BaseSparseEncoder | None                    │
  │                                                             │
  │ connect()                                                   │
  │   if source_path → PersistentClient(path)                   │
  │   else           → HttpClient(host, port, headers)          │
  │                                                             │
  │ detect_schema()                                             │
  │   peek 1 record → dimension = len(embeddings[0])            │
  │   precision always = float32 (Chroma always returns float32)│
  │                                                             │
  │ iterate_batches(batch_size, initial_cursor, schema)         │
  │   offset = initial_cursor or 0                              │
  │   while True:                                               │
  │     result = collection.get(offset=offset, limit=batch_size,│
  │                include=[embeddings, metadatas, documents])  │
  │     if empty: return                                        │
  │     rows = _convert_records(result)                         │
  │     offset += len(rows)                                     │
  │     yield rows, offset, timings                             │
  └─────────────────────────────────────────────────────────────┘
```

### 3c. Target Connector Internals

```
  EndeeTarget
  ┌─────────────────────────────────────────────────────────────┐
  │ State                                                       │
  │   _client         : Endee                                   │
  │   _index          : EndeeIndex                              │
  │   _pk_slot        : int                                     │
  │   _dense_slot     : int                                     │
  │   _sparse_slot    : int (-1 if dense-only)                  │
  │   _payload_slots  : List[int]                               │
  │   _payload_types  : List[FieldType]                         │
  │   _payload_names  : List[str]                               │
  │   filter_fields   : Set[str]                                │
  │                                                             │
  │ setup_index(schema)                                         │
  │   ① validate filter_fields ⊆ metadata field names          │
  │   ② precision check:                                        │
  │      source_precision != float32 → ERROR + sys.exit(1)      │
  │   ③ resolve slot positions from FieldRole                   │
  │   ④ try get_index() → already exists, skip create           │
  │      except NotFoundException → create_index(               │
  │        name, dimension, space_type, M, ef_con, precision,   │
  │        [sparse_model if hybrid])                            │
  │                                                             │
  │ upsert_batch(records, schema)                               │
  │   ① [records] → [endee_dicts] via _to_endee()               │
  │   ② split into chunks of upsert_chunk_size                  │
  │   ③ asyncio.gather(*[_upsert_chunk(c) for c in chunks])     │
  │   ④ collect failed chunks                                   │
  │   ⑤ retry each failed: 3 attempts, 1s→2s→4s backoff         │
  │   ⑥ return (all_success, {tgt_transform, upsert})           │
  │                                                             │
  │ _to_endee(row)                                              │
  │   {                                                         │
  │     "id":             row[pk_slot],                         │
  │     "vector":         row[dense_slot],                      │
  │     "sparse_indices": row[sparse_slot]["indices"],  (hybrid)│
  │     "sparse_values":  row[sparse_slot]["values"],   (hybrid)│
  │     "filter":         {k:v for k in filter_fields},         │
  │     "meta":           {k:v for k not in filter_fields}      │
  │   }                                                         │
  └─────────────────────────────────────────────────────────────┘
```

### 3d. Checkpoint & Resume

```
  migration_checkpoint.json
  ┌───────────────────────────────┐
  │ {                             │
  │   "processed_count": 5000,    │  total records written so far
  │   "last_offset": 5000,        │  opaque cursor (int or UUID)
  │   "batch_number": 5,          │  last completed batch index
  │   "completed": false          │  true when source exhausted
  │ }                             │
  └───────────────────────────────┘
            ▲               │
            │ update()      │ get_last_cursor()
            │ mark_done()   │ is_completed()
            │               ▼
  ┌────────────────────────────────────┐
  │         MigrationPipeline          │
  │                                    │
  │  startup:                          │
  │    if checkpoint.is_completed()    │
  │      → log "already done", exit    │
  │    initial_cursor = get_last_cursor│
  │    pbar.initial = get_processed()  │
  │                                    │
  │  after each batch:                 │
  │    checkpoint.update(              │
  │      batch_number,                 │
  │      len(rows),                    │
  │      next_cursor)                  │
  │                                    │
  │  on source exhausted:              │
  │    checkpoint.mark_completed()     │
  └────────────────────────────────────┘

  Cursor semantics per source:
  ┌─────────────┬─────────────────────────────────────────────┐
  │ Source      │ Cursor type & resume behavior               │
  ├─────────────┼─────────────────────────────────────────────┤
  │ Milvus      │ int (count) — skip first N records in loop  │
  │ Qdrant      │ UUID | None — pass as scroll offset         │
  │ ChromaDB    │ int (offset) — pass to collection.get()     │
  └─────────────┴─────────────────────────────────────────────┘
```

### 3e. Sparse Encoder System

```
  SparseEncoderFactory
  ┌───────────────────────────────────────────────┐
  │ _REGISTRY = {                                 │
  │   "endee/bm25" → EndeeBM25                    │
  │ }                                             │
  │                                               │
  │ create(algorithm) → BaseSparseEncoder         │
  │   None | "endee/bm25" → EndeeBM25()           │
  │   unknown            → ValueError             │
  └───────────────────────────────────────────────┘
                    │
                    ▼
  BaseSparseEncoder (ABC)
  ┌───────────────────────────────────────────────┐
  │ encode(text: str) → {"indices": [], "values"} │
  │ encode_batch(texts) → [{"indices","values"}]  │
  │ build_sparse_field() → FieldSchema            │
  └───────────────────────────────────────────────┘
                    │
                    ▼
  EndeeBM25
  ┌───────────────────────────────────────────────┐
  │ _model = SparseModel("endee/bm25")            │
  │                      (endee-model TF-IDF)     │
  │                                               │
  │ encode_batch(texts)                           │
  │   results = _model.embed(texts)               │
  │   return [{                                   │
  │     "indices": r.indices.tolist(),            │
  │     "values":  r.values.tolist()              │
  │   } for r in results]                         │
  └───────────────────────────────────────────────┘

  When used (dense→hybrid migration):
  ┌────────────────────────────────────────────────────┐
  │ Source._convert_records(batch)                     │
  │                                                    │
  │  texts = [rec[sparse_text_field] for rec in batch] │
  │  sparse_embs = encoder.encode_batch(texts)         │
  │                                                    │
  │  for i, rec in enumerate(batch):                   │
  │    row[2] = sparse_embs[i]                         │
  │             {"indices": [...], "values": [...]}     │
  └────────────────────────────────────────────────────┘
```

### 3f. Precision & Type Resolution

```
  MILVUS_TO_WIRE_PRECISION (milvus_source.py)
  ┌──────────────────────────────────────────┐
  │ FLOAT_VECTOR    → "float32"              │  Milvus returns List[float] natively
  │ FLOAT16_VECTOR  → "raw_binary"           │  ┐
  │ BFLOAT16_VECTOR → "raw_binary"           │  │ Milvus returns raw bytes
  │ BINARY_VECTOR   → "raw_binary"           │  │ → target will reject these
  │ INT8_VECTOR     → "raw_binary"           │  ┘
  └──────────────────────────────────────────┘
              │
              ▼ stored in RowSchema.canonical_precision
              │
  EndeeTarget.setup_index() precision check:
  ┌─────────────────────────────────────────────────────────────────┐
  │ if source_precision != "float32":                               │
  │   ERROR: "source uses quantized type, float32 required"         │
  │   sys.exit(1)                                                   │
  └─────────────────────────────────────────────────────────────────┘

  QDRANT_TO_WIRE_PRECISION (qdrant_source.py)
  ┌──────────────────────────────────────────┐
  │ "none"    → "float32"                    │  ┐
  │ "scalar"  → "float32"                    │  │ Qdrant always decompresses
  │ "binary"  → "float32"                    │  │ back to float32 on scroll
  │ "product" → "float32"                    │  │ → always passes Endee check
  │ "turbo"   → "float32"                    │  ┘
  └──────────────────────────────────────────┘

  CANONICAL_TO_ENDEE_PRECISION (endee_target.py)
  ┌──────────────────────────────────────────┐
  │ "float32" → Precision.FLOAT32            │
  │ "float16" → Precision.FLOAT16            │
  │ "int16"   → Precision.INT16              │
  │ "int8"    → Precision.INT8               │
  │ "binary"  → Precision.BINARY2            │
  └──────────────────────────────────────────┘
  (this is the TARGET index storage precision set at index creation)
```

---

## 4. UML CLASS DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        «abstract»  BaseSource                               │
│─────────────────────────────────────────────────────────────────────────────│
│ + connect() : void                                                          │
│ + detect_schema() : RowSchema                                               │
│ + iterate_batches(batch_size, cursor, schema) : AsyncGenerator               │
│ + close() : void                                                            │
│ + from_args(args) : BaseSource   «classmethod»                              │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │  inherits
          ┌────────────────┼───────────────────────────┐
          │                │                           │
          ▼                ▼                           ▼
┌──────────────────┐ ┌──────────────────┐ ┌────────────────────┐
│ MilvusBaseSource │ │ QdrantBaseSource │ │  ChromaDenseSource │
│──────────────────│ │──────────────────│ │────────────────────│
│ url: str         │ │ url: str         │ │ url: str           │
│ token: str       │ │ collection: str  │ │ collection: str    │
│ collection: str  │ │ api_key: str     │ │ api_key: str       │
│ db: str          │ │ use_https: bool  │ │ source_path: str   │
│ port: int        │ │                  │ │ sparse_algo: str   │
│                  │ │ connect()        │ │                    │
│ connect()        │ │ detect_schema()  │ │ connect()          │
│ detect_schema()  │ │ iterate_batches()│ │ detect_schema()    │
│ iterate_batches()│ │ _convert_records │ │ iterate_batches()  │
│ _convert_records │ │ _scroll()        │ │ _convert_records() │
│ _validate_schema │ │ _validate_schema │ │ from_args()        │
└────────┬─────────┘ └────────┬─────────┘ └────────────────────┘
         │ inherits            │ inherits
    ┌────┴────┐           ┌────┴────┐
    │         │           │         │
    ▼         ▼           ▼         ▼
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│ Milvus  │ │ Milvus   │ │ Qdrant  │ │ Qdrant   │
│ Dense   │ │ Hybrid   │ │ Dense   │ │ Hybrid   │
│ Source  │ │ Source   │ │ Source  │ │ Source   │
│─────────│ │──────────│ │─────────│ │──────────│
│sparse   │ │_validate │ │sparse   │ │_validate │
│_algo    │ │_schema() │ │_algo    │ │_schema() │
│sparse   │ │          │ │sparse   │ │          │
│_text_   │ │from_args │ │_text_   │ │from_args │
│field    │ │()        │ │field    │ │()        │
│_encoder │ │          │ │_encoder │ │          │
│         │ │          │ │         │ │          │
│connect()│ │          │ │connect()│ │          │
│detect_  │ │          │ │detect_  │ │          │
│schema() │ │          │ │schema() │ │          │
│_convert │ │          │ │_convert │ │          │
│from_args│ │          │ │from_args│ │          │
└─────────┘ └──────────┘ └─────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        «abstract»  BaseTarget                               │
│─────────────────────────────────────────────────────────────────────────────│
│ + connect() : void                                                          │
│ + setup_index(schema: RowSchema) : void                                     │
│ + upsert_batch(records, schema) : Tuple[bool, dict]                         │
│ + close() : void                                                            │
│ + from_args(args) : BaseTarget   «classmethod»                              │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ inherits
                           ▼
              ┌─────────────────────────────────┐
              │          EndeeTarget             │
              │─────────────────────────────────│
              │ endee_url: str                  │
              │ endee_api_key: str              │
              │ index_name: str                 │
              │ upsert_chunk_size: int          │
              │ sparse_model: str               │
              │ filter_fields: Set[str]         │
              │ space_type: str                 │
              │ M: int                          │
              │ ef_construct: int               │
              │ precision: str                  │
              │ target_type: str                │
              │ _pk_slot: int                   │
              │ _dense_slot: int                │
              │ _sparse_slot: int               │
              │ _payload_slots: List[int]        │
              │─────────────────────────────────│
              │ connect()                       │
              │ setup_index(schema)             │
              │ upsert_batch(records, schema)   │
              │ _to_endee(row) : dict           │
              │ _upsert_chunk(chunk) : void     │
              │ from_args() «classmethod»       │
              └─────────────────────────────────┘

┌─────────────────────────────────┐
│       MigrationPipeline         │
│─────────────────────────────────│
│ source: BaseSource              │◆──── uses ──── BaseSource
│ target: BaseTarget              │◆──── uses ──── BaseTarget
│ checkpoint: MigrationCheckpoint │◆──── uses ──── MigrationCheckpoint
│ fetch_batch_size: int           │
│ max_queue_size: int             │
│─────────────────────────────────│
│ run() : void                    │
│ _run_async() : coroutine        │
│ _producer(queue) : coroutine    │
│ _consumer(queue, pbar): coro    │
│ _signal_handler(sig, frame)     │
│ _print_report() : void          │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│     MigrationCheckpoint         │
│─────────────────────────────────│
│ filepath: str                   │
│ data: dict                      │
│─────────────────────────────────│
│ update(batch, count, cursor)    │
│ mark_completed()                │
│ clear()                         │
│ is_completed() : bool           │
│ get_processed_count() : int     │
│ get_last_cursor() : Any         │
│ get_batch_number() : int        │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│       RowSchema                 │
│─────────────────────────────────│
│ fields: List[FieldSchema]       │◆──── has many ──── FieldSchema
│ dimension: int                  │
│ space_type: str                 │
│ is_hybrid: bool                 │
│ canonical_precision: str        │
│─────────────────────────────────│
│ get_primary_key()               │
│ get_dense_vector()              │
│ get_sparse_vector()             │
│ get_metadata_fields()           │
│ index_of(name) : int            │
│ require_index_of(name) : int    │
│ total_fields : int              │
└─────────────────────────────────┘

┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       FieldSchema               │   │        MigrationRow             │
│─────────────────────────────────│   │─────────────────────────────────│
│ name: str                       │   │ fields: List[Any]               │
│ field_type: FieldType (enum)    │   │─────────────────────────────────│
│   STRING | INT | FLOAT | BOOL   │   │ set_field(pos, val)             │
│   DENSE_VECTOR | SPARSE_VECTOR  │   │ get_field(pos) : Any            │
│   JSON                          │   │ arity : int                     │
│ role: FieldRole (enum)          │   └─────────────────────────────────┘
│   ID | DENSE_VECTOR             │
│   SPARSE_VECTOR | METADATA      │
│ dimension: Optional[int]        │
└─────────────────────────────────┘

«abstract» BaseSparseEncoder
┌─────────────────────────────────┐
│ encode(text) : dict             │
│ encode_batch(texts) : List[dict]│
│ build_sparse_field(): FieldSchema│
└──────────────┬──────────────────┘
               │ inherits
               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│         EndeeBM25               │   │    SparseEncoderFactory         │
│─────────────────────────────────│   │─────────────────────────────────│
│ _model: SparseModel             │   │ _REGISTRY: dict                 │
│─────────────────────────────────│   │─────────────────────────────────│
│ encode(text) : dict             │   │ register(key, cls) «classmethod»│
│ encode_batch(texts) : List[dict]│   │ create(algo) : BaseSparseEncoder│
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 5. MIGRATION TYPE DECISION TREE

```
                    START
                      │
          ┌───────────┴──────────┐
          │  What source type?   │
          └─────────┬────────────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
     dense       hybrid        dense
  (no sparse   (has stored   + SPARSE_ALGO
   in source)   sparse)
        │           │            │
        ▼           ▼            ▼
  target_type?  target_type?  target_type
        │           │         MUST be
   ┌────┴──┐    ┌───┴────┐    hybrid
   ▼       ▼    ▼        ▼       │
 dense  hybrid dense   hybrid    │
   │       │    │         │      │
   │       │   ERROR   ✓ Use     ▼
   │       │  (cannot   Hybrid  ✓ Generate sparse
   │       │  drop      Source   from text field
   │       │  sparse)   + Endee    via EndeeBM25
   ▼       ▼            Target
  ✓ Dense ✓ Need
  Source   SPARSE_ALGO
  + Endee  → dense→hybrid
  Target    upgrade path


  Supported migration paths:
  ┌─────────────────────────────────────────────────────────┐
  │ milvus/dense  → endee/dense   : FLOAT_VECTOR only       │
  │ milvus/dense  → endee/hybrid  : + SPARSE_ALGO + TEXT    │
  │ milvus/hybrid → endee/hybrid  : has stored sparse        │
  │ qdrant/dense  → endee/dense   : any quant (→ float32)   │
  │ qdrant/dense  → endee/hybrid  : + SPARSE_ALGO + TEXT    │
  │ qdrant/hybrid → endee/hybrid  : has stored sparse        │
  │ chroma/dense  → endee/dense   : always float32           │
  │ chroma/dense  → endee/hybrid  : + SPARSE_ALGO           │
  └─────────────────────────────────────────────────────────┘
```

---

## 6. SEQUENCE DIAGRAM — FULL MIGRATION RUN

```
  CLI/Docker    migrate.py    Pipeline     Source        Target      Checkpoint

     │              │            │            │              │            │
     │──run──────▶  │            │            │              │            │
     │              │            │            │              │            │
     │              │─build──────▶            │              │            │
     │              │  source    │            │              │            │
     │              │─build──────────────────────▶           │            │
     │              │  target    │            │              │            │
     │              │─new(s,t,ck)▶            │              │            │
     │              │            │            │              │            │
     │              │            │─connect()─▶│              │            │
     │              │            │◀─ ok ──────│              │            │
     │              │            │─connect()───────────────▶│            │
     │              │            │◀─ ok ────────────────────│            │
     │              │            │            │              │            │
     │              │            │─detect_schema()──────────▶            │
     │              │            │  (inspect collection metadata)        │
     │              │            │◀─ RowSchema ─────────────             │
     │              │            │            │              │            │
     │              │            │─setup_index(schema)──────▶            │
     │              │            │  (create or get Endee index)          │
     │              │            │◀─ ok ────────────────────             │
     │              │            │            │              │            │
     │              │            │─load_checkpoint()──────────────────▶  │
     │              │            │◀─ cursor=N, count=N ────────────────  │
     │              │            │            │              │            │
     │              │            │─── asyncio.run(_run_async) ──────────▶│
     │              │            │            │              │            │
     │         ┌────│────────────│────────────│──────────────│───┐       │
     │         │    │  PRODUCER  │            │              │   │       │
     │         │    │            │─iterate_batches(sz,N,sch)─▶   │       │
     │         │    │            │            │              │   │       │
     │         │    │            │            │─QueryIterator│   │       │
     │         │    │            │            │  .next()     │   │       │
     │         │    │            │            │  (executor)  │   │       │
     │         │    │            │            │◀─ raw_batch  │   │       │
     │         │    │            │            │              │   │       │
     │         │    │            │            │─_convert_records()       │
     │         │    │            │            │  [optional encode_batch] │
     │         │    │            │            │◀─ rows, cursor           │
     │         │    │            │            │              │   │       │
     │         │    │            │─ queue.put({rows, cursor, timings})   │
     │         │    │            │  (repeat for each batch)  │   │       │
     │         │    │            │─ queue.put(None) ──EOF    │   │       │
     │         └────│────────────│────────────│──────────────│───┘       │
     │              │            │            │              │            │
     │         ┌────│────────────│────────────│──────────────│───┐       │
     │         │    │  CONSUMER  │            │              │   │       │
     │         │    │            │─ queue.get()              │   │       │
     │         │    │            │            │              │   │       │
     │         │    │            │────upsert_batch(rows, schema)─▶       │
     │         │    │            │            │  ┌────────────────────┐  │
     │         │    │            │            │  │ _to_endee(row) x N │  │
     │         │    │            │            │  │ chunk into pieces  │  │
     │         │    │            │            │  │ gather(upsert × K) │  │
     │         │    │            │            │  │ retry failed       │  │
     │         │    │            │            │  └────────────────────┘  │
     │         │    │            │◀─ (True, timings) ───────             │
     │         │    │            │                          │   │        │
     │         │    │            │─checkpoint.update(batch, count, cur)─▶│
     │         │    │            │            │              │   │       │
     │         │    │            │  (repeat for each batch)  │   │       │
     │         │    │            │            │              │   │       │
     │         │    │            │  receives None → mark_completed()─────▶│
     │         └────│────────────│────────────│──────────────│───┘       │
     │              │            │            │              │            │
     │              │            │─close() ──▶│              │            │
     │              │            │─close() ───────────────▶ │            │
     │              │            │─print_report()            │            │
     │◀─done─────── │            │            │              │            │
```

---

## 7. MODULE DEPENDENCY MAP

```
                         ┌──────────────┐
                         │  migrate.py  │  ← entry point
                         └──────┬───────┘
                                │ imports
           ┌────────────────────┼─────────────────────┐
           ▼                    ▼                      ▼
    ┌──────────────┐   ┌──────────────┐     ┌──────────────────┐
    │ sources/     │   │ targets/     │     │ core/pipeline.py │
    │ milvus_src   │   │ endee_target │     └────────┬─────────┘
    │ qdrant_src   │   └──────┬───────┘              │
    │ chroma_src   │          │                       │
    └──────┬───────┘          │                       │
           │                  │           ┌───────────┴──────────┐
           │                  │           ▼                      ▼
           │         ┌────────┴──────┐  ┌──────────────┐  ┌──────────────┐
           │         │  core/        │  │ core/        │  │ core/        │
           │         │  base_target  │  │ base_source  │  │ checkpoint   │
           │         └───────────────┘  └──────────────┘  └──────────────┘
           │
           │  all sources import:
           ├──────────────────────────────────────────────────┐
           ▼                                                  ▼
  ┌─────────────────┐                             ┌──────────────────────┐
  │ core/schema.py  │                             │ core/type_registry   │
  │                 │                             │                      │
  │ FieldType (enum)│                             │ SPACE_COSINE/L2/IP   │
  │ FieldRole (enum)│                             │ PRECISION_FLOAT32    │
  │ FieldSchema     │                             │ PRECISION_RAW_BINARY │
  │ RowSchema       │                             │ resolve_space()      │
  │ MigrationRow    │                             │ resolve_precision()  │
  └─────────────────┘                             └──────────────────────┘

           │ (optional, dense→hybrid only)
           ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   sparse_encoders/                       │
  │                                                          │
  │  factory_sparse_encoder.py                               │
  │    SparseEncoderFactory                                  │
  │      create(algo) → BaseSparseEncoder                    │
  │                                                          │
  │  interface_sparse_encoder.py                             │
  │    BaseSparseEncoder (ABC)                               │
  │                                                          │
  │  concrete_sparse_encoders.py                             │
  │    EndeeBM25 → SparseModel("endee/bm25") [endee-model]  │
  └──────────────────────────────────────────────────────────┘

  constants.py
  ┌─────────────────────────────────────────────────────────┐
  │ ENDEE_V2_API         : "/api/v2/"                       │
  │ DEFAULT_SPARSE_MODEL : "endee_bm25"                     │
  │ HNSW defaults        : M=16, EF_CONSTRUCT=128           │
  └─────────────────────────────────────────────────────────┘
```
