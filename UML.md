# UML Diagrams — Endee Migration Tool

---

## 1. Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      <<abstract>>                                        │
│                                       BaseSource                                         │
│─────────────────────────────────────────────────────────────────────────────────────────│
│ + connect()                                                                              │
│ + detect_schema() : RowSchema                                                            │
│ + iterate_batches(batch_size, cursor, schema)                                            │
│ + close()                                                                                │
│ + from_args(args) : BaseSource                                                           │
└─────────────────────────────┬───────────────────────────────────────────────────────────┘
                              │ extends
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  MilvusBaseSource│  │  QdrantBaseSource│  │   ChromaDenseSource  │
│──────────────────│  │──────────────────│  │──────────────────────│
│ url, token       │  │ url, api_key     │  │ url, collection      │
│ collection, db   │  │ port, use_https  │  │ api_key, source_path │
│ _dense_slot      │  │ _dense_slot      │  │ sparse_algo          │
│ _sparse_slot     │  │ _sparse_slot     │  │ canonical_precision  │
│ _payload_slot    │  │ _payload_slot    │  │ _encoder             │
│──────────────────│  │──────────────────│  │──────────────────────│
│ connect()        │  │ connect()        │  │ connect()            │
│ detect_schema()  │  │ detect_schema()  │  │ detect_schema()      │
│ _load_collection │  │ _convert_records │  │ _load_encoder()      │
│ _decode_vector() │  │ iterate_batches()│  │ _encode_sparse()     │
│ _convert_records │  │ _validate_schema │  │ _convert_records()   │
│ iterate_batches()│  └────────┬─────────┘  │ iterate_batches()    │
│ _validate_schema │           │ extends    │ from_args()          │
└────────┬─────────┘    ┌──────┴──────┐     └──────────────────────┘
         │ extends      │             │
    ┌────┴────┐   ┌─────┴──────┐ ┌───┴──────────┐
    │ Milvus  │   │  Qdrant    │ │   Qdrant     │
    │  Dense  │   │   Dense    │ │   Hybrid     │
    │ Source  │   │  Source    │ │  Source      │
    │─────────│   │────────────│ │──────────────│
    │sparse_  │   │ sparse_    │ │_validate_    │
    │  algo   │   │   algo     │ │  schema()    │
    │sparse_  │   │ sparse_    │ │from_args()   │
    │text_    │   │ text_field │ └──────────────┘
    │ field   │   │ _encoder   │
    │_encoder │   │────────────│
    │─────────│   │connect()   │
    │connect()│   │detect_     │
    │detect_  │   │  schema()  │
    │ schema()│   │_convert_   │
    │_convert_│   │  records() │
    │ records │   │_validate_  │
    │_validate│   │  schema()  │
    │from_args│   │from_args() │
    └─────────┘   └────────────┘

    ┌─────────┐
    │ Milvus  │
    │ Hybrid  │
    │ Source  │
    │─────────│
    │_validate│
    │from_args│
    └─────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      <<abstract>>                                        │
│                                       BaseTarget                                         │
│─────────────────────────────────────────────────────────────────────────────────────────│
│ + connect()                                                                              │
│ + setup_index(schema: RowSchema)                                                         │
│ + upsert_batch(records, schema) : (bool, dict)                                           │
│ + close()                                                                                │
│ + from_args(args) : BaseTarget                                                           │
└─────────────────────────────┬───────────────────────────────────────────────────────────┘
                              │ extends
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     EndeeTarget                                          │
│─────────────────────────────────────────────────────────────────────────────────────────│
│ + endee_url, endee_api_key, index_name                                                   │
│ + upsert_chunk_size, sparse_model, target_type                                           │
│ + space_type, M, ef_construct, precision                                                 │
│ + filter_fields : Set<str>                                                               │
│ - _pk_slot, _dense_slot, _sparse_slot, _payload_slots                                   │
│─────────────────────────────────────────────────────────────────────────────────────────│
│ + connect()           + setup_index(schema)      - _to_endee(record, schema)            │
│ + upsert_batch()      - _upsert_chunk(chunk)     + from_args(args)                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              Core Data Model                                   │
├─────────────────────┬──────────────────────┬──────────────────────────────────┤
│      RowSchema      │     FieldSchema       │       MigrationRow               │
│─────────────────────│──────────────────────│──────────────────────────────────│
│ fields: List        │ name: str            │ fields: List<Any>                │
│ space_type: str     │ field_type: FieldType│ set_field(pos, value)            │
│ dimension: int      │ role: FieldRole      │ get_field(pos): Any              │
│ is_hybrid: bool     │ dimension: int       │ arity: int                       │
│ canonical_precision │                      │                                  │
│─────────────────────│ FieldType (enum)     │ FieldRole (enum)                 │
│ get_primary_key()   │ STRING  DENSE_VECTOR │ ID           DENSE_VECTOR        │
│ get_dense_vector()  │ INT     SPARSE_VECTOR│ SPARSE_VECTOR  METADATA          │
│ get_sparse_vector() │ FLOAT   JSON         │                                  │
│ get_metadata_fields │ BOOL                 │                                  │
│ index_of(name)      │                      │                                  │
└─────────────────────┴──────────────────────┴──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Sparse Encoding                                  │
│                                                                      │
│  <<abstract>>                                                        │
│  BaseSparseEncoder                                                   │
│  ─────────────────────────────────────────────────────────────────  │
│  + build_sparse_field() : FieldSchema                                │
│  + encode(text) : {"indices": [...], "values": [...]}                │
│  + encode_batch(texts) : List<dict>                                  │
│         ▲ extends                                                    │
│         │                                                            │
│  EndeeBM25                       SparseEncoderFactory               │
│  ─────────────────────           ────────────────────────           │
│  - _model: SparseModel           - _REGISTRY: dict                  │
│  + encode(text)                  + register(key, cls)               │
│  + encode_batch(texts)           + create(algo) : Encoder           │
│                                                                      │
│  Registered: "endee/bm25" → EndeeBM25                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sequence Diagram — Full Migration Run

```
  User          migrate.py       MigrationPipeline     Source            Target         Checkpoint
   │                │                    │                │                  │               │
   │ docker up      │                    │                │                  │               │
   │───────────────▶│                    │                │                  │               │
   │                │ parse args         │                │                  │               │
   │                │ validate combos    │                │                  │               │
   │                │ from_args(args)───────────────────▶│                  │               │
   │                │ from_args(args)──────────────────────────────────────▶│               │
   │                │ MigrationPipeline()────────────────▶│                  │               │
   │                │ pipeline.run()─────▶│               │                  │               │
   │                │                    │                │                  │               │
   │         ═══════════════════════ Setup Phase ═══════════════════════════│               │
   │                │                    │                │                  │               │
   │                │                    │──connect()────▶│                  │               │
   │                │                    │◀── connected ──│                  │               │
   │                │                    │──connect()───────────────────────▶│               │
   │                │                    │◀── connected ────────────────────│               │
   │                │                    │──detect_schema()──▶│              │               │
   │                │                    │  [inspect collection]             │               │
   │                │                    │  [load encoder if sparse_algo]    │               │
   │                │                    │  [verify text field if needed]    │               │
   │                │                    │◀── RowSchema ──────│              │               │
   │                │                    │──setup_index(schema)──────────────▶│              │
   │                │                    │  [validate filter_fields]          │              │
   │                │                    │  [resolve slots from roles]        │              │
   │                │                    │  [create_index if not exists]      │              │
   │                │                    │◀── index ready ───────────────────│               │
   │                │                    │                │                  │               │
   │         ═══════════════════════ Data Flow Phase ════════════════════════│               │
   │                │                    │                │                  │               │
   │                │         asyncio.gather(producer, consumer)            │               │
   │                │                    │                │                  │               │
   │                │          ┌─────────┴──────────────────────────────────────────────┐   │
   │                │          │  loop until source exhausted                           │   │
   │                │          │                │                  │               │    │   │
   │                │          │  iterate_batches()──────────────▶│               │    │   │
   │                │          │  [fetch batch from DB]            │               │    │   │
   │                │          │  [encode_batch(texts) if sparse]  │               │    │   │
   │                │          │  [fill MigrationRow slots]        │               │    │   │
   │                │          │  ◀── (rows, cursor, timings) ─────│               │    │   │
   │                │          │                │                  │               │    │   │
   │                │          │  [queue.put(batch)]  [queue.get()]│               │    │   │
   │                │          │                │                  │               │    │   │
   │                │          │  upsert_batch(rows, schema)──────────────────────▶│   │   │
   │                │          │  [_to_endee() per row]            │               │    │   │
   │                │          │  [chunk + parallel upsert]        │               │    │   │
   │                │          │  [retry on fail: 1s→2s→4s]        │               │    │   │
   │                │          │  ◀── (True, timings) ─────────────────────────────│   │   │
   │                │          │                │                  │               │    │   │
   │                │          │  checkpoint.update(batch, count, cursor)───────────────▶│  │
   │                │          └────────────────────────────────────────────────────┘   │   │
   │                │                    │                │                  │           │   │
   │                │                    │──close()──────▶│                  │           │   │
   │                │                    │──close()───────────────────────── ▶│          │   │
   │                │                    │──mark_completed()─────────────────────────────▶│  │
   │                │                    │──print_report()│                  │           │   │
   │◀── exit 0 ─────│                    │                │                  │           │   │
```

---

## 3. Sequence Diagram — Dense → Hybrid (Sparse Generation)

```
  QdrantDenseSource        SparseEncoderFactory      EndeeBM25         SparseModel
  (MilvusDenseSource)              │                     │                  │
         │                         │                     │                  │
  ══════ connect() ══════════════════════════════════════════════════════════
         │                         │                     │                  │
         │──create("endee/bm25")──▶│                     │                  │
         │                         │──EndeeBM25()────────▶│                 │
         │                         │                     │──SparseModel()──▶│
         │◀── encoder instance ────│                     │                  │
         │                         │                     │                  │
  ══════ detect_schema() ═══════════════════════════════════════════════════
         │                         │                     │                  │
         │  super().detect_schema() → base RowSchema      │                  │
         │  scroll(limit=1) → peek at real record         │                  │
         │  verify payload[sparse_text_field] exists      │                  │
         │  !! ERROR if field missing or empty            │                  │
         │  append FieldSchema(SPARSE_VECTOR) to schema   │                  │
         │  schema.is_hybrid = True                       │                  │
         │                         │                     │                  │
  ══════ _convert_records() per batch ════════════════════════════════════════
         │                         │                     │                  │
         │  texts = [pt.payload[sparse_text_field] ...]  │                  │
         │  !! ERROR if all texts empty                   │                  │
         │──encode_batch(texts)────────────────────────▶│                  │
         │                         │                     │──model.embed()──▶│
         │                         │                     │◀─[embeddings]────│
         │◀─[{"indices","values"}]─────────────────────│                  │
         │                         │                     │                  │
         │  per record:                                   │                  │
         │  ├── row.set_field(0,  id)                     │                  │
         │  ├── row.set_field(1,  dense_vector)           │                  │
         │  ├── row.set_field(2,  {indices, values})      │                  │
         │  └── row.set_field(-1, payload_dict)           │                  │
```

---

## 4. State Diagram — Migration Pipeline

```
                        ┌─────────────────┐
                        │   pipeline.run() │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │     SETUP        │
                        │                 │
                        │ connect source  │
                        │ connect target  │
                        │ detect_schema() │
                        │ setup_index()   │
                        └────────┬────────┘
                                 │ all ready
                                 ▼
                 ┌──────────────────────────────┐
                 │         DATA FLOW             │◀─────────────────┐
                 │                              │                   │
                 │ fetch batch from source      │                   │
                 │ [encode sparse if needed]    │                   │
                 │ upsert batch to target       │                   │
                 │ checkpoint.update()          │                   │
                 └──────┬──────────┬────────────┘                   │
                        │          │                                │
              source     │          │ batch ok                       │
              exhausted  │          └────────────────────────────────┘
                        │                                  (next batch)
                        │
          ┌─────────────┼──────────────┬──────────────────┐
          ▼             ▼              ▼                   ▼
  ┌──────────────┐ ┌─────────┐ ┌───────────────┐  ┌────────────────┐
  │  COMPLETED   │ │ FAILED  │ │  INTERRUPTED  │  │ FAILED         │
  │              │ │         │ │               │  │ (producer err) │
  │ mark_        │ │ retries │ │ SIGINT/SIGTERM│  │                │
  │ completed()  │ │ exhausted│ │ progress saved│  │                │
  │              │ │         │ │               │  │                │
  │ "MIGRATION   │ │"COMPLETED│ │"MIGRATION     │  │"MIGRATION      │
  │  COMPLETED   │ │ WITH     │ │ INTERRUPTED"  │  │ FAILED"        │
  │  SUCCESSFULLY│ │ ERRORS"  │ │               │  │                │
  │              │ │         │ │ re-run to     │  │                │
  │  exit 0      │ │ exit 1  │ │ resume        │  │ exit 1         │
  └──────────────┘ └─────────┘ └───────────────┘  └────────────────┘
```

---

## 5. Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                           │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  migrate.py                                                               │  │
│   │  ─ parse args + env vars       ─ SOURCE_REGISTRY[(from_db,source_type)]  │  │
│   │  ─ validate type combos        ─ TARGET_REGISTRY[(to_db, target_type)]   │  │
│   └───────────┬──────────────────────────────────────────┬───────────────────┘  │
└───────────────│──────────────────────────────────────────│─────────────────────┘
                │ builds                                    │ builds
                ▼                                           ▼
┌────────────────────────────────┐         ┌──────────────────────────────────────┐
│        Pipeline Layer          │         │         Source Connectors             │
│                                │         │                                       │
│  ┌─────────────────────────┐   │         │  ┌────────────────┐                  │
│  │   MigrationPipeline     │   │         │  │MilvusDenseSource│◀── MilvusDB     │
│  │  ─ _producer(queue)     │   │         │  └────────────────┘                  │
│  │  ─ _consumer(queue)     │◀──┼─source──│  ┌─────────────────┐                │
│  │  ─ checkpoint.update()  │   │         │  │MilvusHybridSource│◀── MilvusDB    │
│  └─────────────────────────┘   │         │  └─────────────────┘                │
│                                │         │  ┌────────────────┐                  │
│  ┌─────────────────────────┐   │         │  │QdrantDenseSource│◀── QdrantDB     │
│  │  MigrationCheckpoint    │   │         │  └────────────────┘                  │
│  │  ─ save/load JSON        │   │         │  ┌──────────────────┐               │
│  └─────────────────────────┘   │         │  │QdrantHybridSource │◀── QdrantDB   │
└────────────────────────────────┘         │  └──────────────────┘               │
                                            │  ┌────────────────┐                  │
┌────────────────────────────────┐         │  │ChromaDenseSource│◀── ChromaDB     │
│        Target Connector        │         │  └────────────────┘                  │
│                                │         └──────────────────────────────────────┘
│  ┌─────────────────────────┐   │
│  │      EndeeTarget        │──────────────────────────────▶  Endee Index
│  │  ─ setup_index()        │   │
│  │  ─ upsert_batch()       │   │         ┌──────────────────────────────────────┐
│  │  ─ retry logic          │   │         │        Sparse Encoding                │
│  └─────────────────────────┘   │         │                                       │
└────────────────────────────────┘         │  SparseEncoderFactory                 │
                                            │  ─ _REGISTRY: {"endee/bm25": BM25}   │
┌────────────────────────────────┐         │  ─ create(algo) → encoder             │
│        Core Schema             │         │                                       │
│                                │         │  EndeeBM25                            │
│  RowSchema / FieldSchema       │         │  ─ SparseModel("endee/bm25")          │
│  MigrationRow                  │         │  ─ encode_batch(texts)                │
│  FieldType / FieldRole (enums) │         │                                       │
│  TypeRegistry (mappings)       │         │  Used by: MilvusDense, QdrantDense,   │
└────────────────────────────────┘         │           ChromaDense (when sparse)   │
                                            └──────────────────────────────────────┘
```

---

## 6. Activity Diagram — Validation in main()

```
      START
        │
        ▼
  ┌─────────────────────────────────────────┐
  │   Parse args + load env vars            │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
         sparse_algo set AND target_type == "dense"?
               │ YES                  │ NO
               ▼                      │
  ┌─────────────────────────┐         │
  │ ERROR: SPARSE_ALGO only │         │
  │ valid for hybrid target │         │
  │         exit 1          │         │
  └─────────────────────────┘         ▼
                     source_type == "dense" AND
                     target_type == "hybrid" AND
                     sparse_algo NOT set?
                           │ YES                  │ NO
                           ▼                      │
              ┌─────────────────────────┐         │
              │ ERROR: Set              │         │
              │ SPARSE_ALGO=endee/bm25  │         │
              │         exit 1          │         │
              └─────────────────────────┘         ▼
                                 source_type == "hybrid" AND
                                 target_type == "dense"?
                                       │ YES                  │ NO
                                       ▼                      │
                          ┌─────────────────────────┐         │
                          │ ERROR: Cannot drop        │         │
                          │ sparse vectors.           │         │
                          │ Set TARGET_TYPE=hybrid    │         │
                          │         exit 1            │         │
                          └─────────────────────────┘         ▼
                                               sparse_algo set AND
                                               from_db in (qdrant, milvus) AND
                                               sparse_text_field NOT set?
                                                     │ YES               │ NO
                                                     ▼                   │
                                        ┌──────────────────────┐         │
                                        │ ERROR: Set            │         │
                                        │ SPARSE_TEXT_FIELD=    │         │
                                        │ <field_name>          │         │
                                        │       exit 1          │         │
                                        └──────────────────────┘         │
                                                                          ▼
                                                               Build source + target
                                                               Build checkpoint
                                                               Run MigrationPipeline
                                                                          │
                                                                        END
```

---

## 7. Migration Combinations

```
FROM_DB   SOURCE_TYPE   TARGET_TYPE   SPARSE_ALGO   SPARSE_TEXT_FIELD   Result
───────   ───────────   ───────────   ───────────   ─────────────────   ──────────────────────────────────
milvus    dense         dense         —             —                   Dense → Dense
milvus    dense         hybrid        endee/bm25    <field_name>        Dense + generated sparse → Hybrid
milvus    hybrid        hybrid        —             —                   Hybrid → Hybrid
qdrant    dense         dense         —             —                   Dense → Dense
qdrant    dense         hybrid        endee/bm25    <field_name>        Dense + generated sparse → Hybrid
qdrant    hybrid        hybrid        —             —                   Hybrid → Hybrid
chroma    dense         dense         —             —                   Dense → Dense
chroma    dense         hybrid        endee/bm25    —  (uses documents) Dense + generated sparse → Hybrid
any       hybrid        dense         —             —                   ERROR — cannot drop sparse
any       dense         hybrid        —             —                   ERROR — SPARSE_ALGO required
any       dense         hybrid        endee/bm25    — (qdrant/milvus)   ERROR — SPARSE_TEXT_FIELD required
```
