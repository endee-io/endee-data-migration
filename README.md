# Endee Migration Tool

Migrate vector data from **Milvus**, **Qdrant**, or **ChromaDB** into **Endee**. Supports dense-only and hybrid (dense + sparse) collections. Ships as a Docker image with checkpoint-based resume for large migrations.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Source collection | Must exist and be accessible before running |
| API access | Endee API key must have write permission on the target index |
| Network reachability | The machine running the migration must be able to reach both the source DB and Endee |
| Source and Endee on separate servers | Do not run source DB and Endee on the same server |
| Milvus vector type | Source collection must use `FLOAT_VECTOR`. Quantized types (`FLOAT16_VECTOR`, `INT8_VECTOR`, `BFLOAT16_VECTOR`, `BINARY_VECTOR`) are not supported — see Limitations |

---

## Setup

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/endee-io/endee-data-migration.git
cd endee-data-migration/
```

**Step 2 — Create your `.env` file:**
```bash
cp .env.example .env
```

**Step 3 — Edit `.env` with your values** (see full reference below).

**Step 4 — Make scripts executable:**
```bash
chmod +x entrypoint.sh
chmod +x cmd.sh
```

**Step 5 — Run the migration:**
```bash
./cmd.sh
```

---

## Environment Variables

### Migration type (required)

| Variable | Required | Description | Example |
|---|---|---|---|
| `FROM_DB` | Yes | Source database type | `milvus` \| `qdrant` \| `chroma` |
| `TO_DB` | Yes | Target database type | `endee` |
| `SOURCE_TYPE` | Yes | Vector type of the source collection | `dense` \| `hybrid` |
| `TARGET_TYPE` | Yes | Vector type of the Endee index to create | `dense` \| `hybrid` |

### Source (required)

| Variable | Required | Description | Example |
|---|---|---|---|
| `SOURCE_URL` | Yes | Full URL of the source DB server | `http://192.168.1.10` |
| `SOURCE_PORT` | Yes | Port the source DB listens on | `19530` (Milvus) \| `6333` (Qdrant) \| `8000` (Chroma) |
| `SOURCE_COLLECTION` | Yes | Collection name in the source DB | `my_collection` |
| `SOURCE_API_KEY` | If auth enabled | API key or token for the source DB. Leave blank if auth is off. | `your-source-api-key` |
| `SOURCE_DB` | Milvus only | Milvus logical database name | `default` |
| `USE_HTTPS` | Qdrant only | Use HTTPS for Qdrant connection | `true` \| `false` |

### Target (required)

| Variable | Required | Description | Example |
|---|---|---|---|
| `TARGET_URL` | Yes | URL of your Endee server | `http://192.168.1.20:8080` |
| `TARGET_API_KEY` | Yes | Endee API key with write access | `your-endee-api-key` |
| `TARGET_COLLECTION` | Yes | Endee index name. Created automatically if it does not exist. | `my_index` |

### Index configuration (required)

| Variable | Required | Description | Example |
|---|---|---|---|
| `SPACE_TYPE` | Yes | Distance metric for the Endee index | `cosine` \| `l2` \| `ip` |
| `PRECISION` | Yes | Storage precision for vectors in Endee. See precision table below. | `float32` \| `int16` \| `int8` |
| `M` | Yes | HNSW bidirectional links per node. Higher = better recall, more memory. | `16` |
| `EF_CONSTRUCT` | Yes | HNSW build beam width. Higher = better graph quality, slower inserts. | `128` |

### Sparse / hybrid (required only for hybrid target)

| Variable | Required | Description | Example |
|---|---|---|---|
| `SPARSE_ALGO` | Yes, if `SOURCE_TYPE=dense` and `TARGET_TYPE=hybrid` | Encoder to generate sparse vectors on the fly from a text field | `endee/bm25` |
| `SPARSE_TEXT_FIELD` | Yes, if `SPARSE_ALGO` set (Milvus/Qdrant) | Field name in the source that contains text to encode. Not needed for Chroma (uses documents natively). | `text` \| `content` |
| `SPARSE_MODEL` | Yes, if `TARGET_TYPE=hybrid` | Sparse model name passed to Endee at index creation | `endee_bm25` \| `default` |

### Filtering (optional)

| Variable | Required | Description | Example |
|---|---|---|---|
| `FILTER_FIELDS` | No | Comma-separated metadata field names to store in Endee's `filter` slot for fast filtering. All other fields go to `meta`. Leave blank to put everything in `meta`. | `category,status,year` |

### Performance

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `1000` | Records fetched from source per batch. Lower if you hit memory limits. |
| `UPSERT_SIZE` | `100` | Records sent to Endee per upsert call. Lower if upserts time out. |
| `MAX_QUEUE_SIZE` | `5` | Max batches held in memory between producer and consumer. Controls memory pressure. |

### Checkpoint / resume

| Variable | Default | Description |
|---|---|---|
| `RESUME` | `true` | `true` = continue from last checkpoint. `false` = clear checkpoint and start fresh. |
| `CHECKPOINT_FILE` | `/app/data/checkpoints/migration.json` | Path where progress is saved. Do not change unless you know what you are doing. |

### Debug

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Set `true` to enable verbose debug logging. |

---

## Example `.env` — Milvus dense migration

```bash
FROM_DB=milvus
TO_DB=endee
SOURCE_TYPE=dense
TARGET_TYPE=dense

SOURCE_URL=http://192.168.1.10
SOURCE_PORT=19530
SOURCE_COLLECTION=my_collection
SOURCE_API_KEY=

TARGET_URL=http://192.168.1.20:8080
TARGET_API_KEY=your-endee-api-key
TARGET_COLLECTION=my_index

SPACE_TYPE=cosine
PRECISION=int16
M=16
EF_CONSTRUCT=128

BATCH_SIZE=1000
UPSERT_SIZE=100
MAX_QUEUE_SIZE=5
RESUME=true
DEBUG=false
```

## Example `.env` — Qdrant dense → hybrid (generate sparse from text)

```bash
FROM_DB=qdrant
TO_DB=endee
SOURCE_TYPE=dense
TARGET_TYPE=hybrid

SOURCE_URL=http://192.168.1.10
SOURCE_PORT=6333
SOURCE_COLLECTION=my_collection
SOURCE_API_KEY=your-qdrant-key
USE_HTTPS=false

TARGET_URL=http://192.168.1.20:8080
TARGET_API_KEY=your-endee-api-key
TARGET_COLLECTION=my_hybrid_index

SPACE_TYPE=cosine
PRECISION=float32
M=16
EF_CONSTRUCT=128

SPARSE_ALGO=endee/bm25
SPARSE_TEXT_FIELD=text
SPARSE_MODEL=endee_bm25

BATCH_SIZE=1000
UPSERT_SIZE=100
MAX_QUEUE_SIZE=5
RESUME=true
```

---

## Supported Migration Paths

| `FROM_DB` | `SOURCE_TYPE` | `TARGET_TYPE` | `SPARSE_ALGO` needed | Notes |
|---|---|---|---|---|
| `milvus` | `dense` | `dense` | No | Source must use `FLOAT_VECTOR` |
| `milvus` | `dense` | `hybrid` | Yes | Generates sparse from `SPARSE_TEXT_FIELD` |
| `milvus` | `hybrid` | `hybrid` | No | Uses stored sparse vectors from source |
| `qdrant` | `dense` | `dense` | No | Any Qdrant quantization — always returns float32 |
| `qdrant` | `dense` | `hybrid` | Yes | Generates sparse from `SPARSE_TEXT_FIELD` |
| `qdrant` | `hybrid` | `hybrid` | No | Uses stored named sparse vectors |
| `chroma` | `dense` | `dense` | No | Always float32 |
| `chroma` | `dense` | `hybrid` | Yes | Generates sparse from ChromaDB document text field |

**Invalid combinations (script exits immediately):**

| Combination | Reason |
|---|---|
| `SOURCE_TYPE=dense` + `TARGET_TYPE=hybrid` + no `SPARSE_ALGO` | No stored sparse in source and no encoder to generate them |
| `SOURCE_TYPE=hybrid` + `TARGET_TYPE=dense` | Cannot silently drop sparse vectors |
| `SPARSE_ALGO` set + `TARGET_TYPE=dense` | Sparse generation only makes sense for a hybrid target |
| `SPARSE_ALGO` set + `FROM_DB=milvus` or `qdrant` + no `SPARSE_TEXT_FIELD` | Need to know which field contains the text to encode |

---

## Precision

### Endee target precision options (`PRECISION`)

This is the storage precision Endee will use for the new index. The source data must arrive as `float32` (the script enforces this — see Limitations). Endee quantises internally at index creation time.

| `PRECISION` value | Endee storage | Memory use | Recall |
|---|---|---|---|
| `float32` | Full 32-bit float | Highest | Best |
| `float16` | 16-bit half float | High | Very good |
| `int16` | 16-bit integer | Medium | Good — recommended default |
| `int8` | 8-bit integer | Low | Acceptable |
| `binary` | 1-bit binary | Lowest | Lossy |

Recommended: `int16` for most migrations. Use `float32` to preserve full precision.

---

### Source wire precision — what each source actually sends

#### Milvus

The Milvus client returns different data depending on the field type. The script checks this at schema detection time.

| Milvus field type | Wire format sent to migration | Supported | Notes |
|---|---|---|---|
| `FLOAT_VECTOR` | `List[float]` (float32) | **Yes** | Only supported type |
| `FLOAT16_VECTOR` | Raw bytes (float16) | **No** | See Limitations |
| `BFLOAT16_VECTOR` | Raw bytes (bfloat16) | **No** | See Limitations |
| `INT8_VECTOR` | Raw bytes (int8) | **No** | See Limitations |
| `BINARY_VECTOR` | Raw bytes (1-bit) | **No** | See Limitations |

#### Qdrant

Qdrant always decompresses vectors back to `float32` on scroll regardless of how they are stored internally. All quantization types are therefore supported as source.

| Qdrant quantization config | Wire format sent to migration | Supported | Notes |
|---|---|---|---|
| None | float32 | Yes | No quantization |
| Scalar (int8) | float32 | Yes | Decompressed on scroll |
| Binary (1-bit) | float32 | Yes | Decompressed on scroll |
| Product quantization | float32 | Yes | Decompressed on scroll |
| Turbo quantization | float32 | Yes | Decompressed on scroll |

#### ChromaDB

ChromaDB always returns embeddings as `float32` lists. No quantization options exist on the source side.

| ChromaDB storage | Wire format sent to migration | Supported |
|---|---|---|
| Any (always float32) | float32 | Yes |

---

## Field Mapping

### How metadata is split into `filter` and `meta`

| `FILTER_FIELDS` setting | Source fields | Endee `filter` | Endee `meta` |
|---|---|---|---|
| Not set | All metadata / payload fields | `{}` (empty) | All fields |
| `category,status,year` | All metadata / payload fields | Only `category`, `status`, `year` | Everything else |

Only fields that exist in `FILTER_FIELDS` and are present in the source record end up in `filter`. Fields in `FILTER_FIELDS` that do not exist in the source are silently ignored. If you set an invalid `FILTER_FIELDS` value that does not match any metadata field name, the script exits before migration starts.

### Milvus → Endee field mapping

| Milvus concept | Endee field | How |
|---|---|---|
| Primary key (`is_primary=True`) | `id` | `str(record[pk_field])` |
| Dense vector (`FLOAT_VECTOR`) | `vector` | Passed through as-is (no decoding) |
| Sparse vector (`SPARSE_FLOAT_VECTOR`) | `sparse_indices`, `sparse_values` | Sorted index→weight dict split into two lists |
| All other fields (non-pk, non-vector) | `filter` or `meta` | Split by `FILTER_FIELDS` setting |

### Qdrant → Endee field mapping

| Qdrant concept | Endee field | How |
|---|---|---|
| Point ID | `id` | `str(point.id)` |
| Dense vector (named or unnamed) | `vector` | Passed through as-is |
| Sparse vector (named sparse field) | `sparse_indices`, `sparse_values` | Split from sparse vector dict |
| Payload | `filter` or `meta` | Split by `FILTER_FIELDS` setting |

### ChromaDB → Endee field mapping

| ChromaDB concept | Endee field | How |
|---|---|---|
| ID | `id` | As-is |
| Embedding | `vector` | Passed through as-is |
| Document | `meta.document` | Stored in meta when `STORE_DOCUMENT_IN_META=true` |
| Metadata | `filter` or `meta` | Split by `FILTER_FIELDS` setting |

---

## Checkpoint and Resume

The script saves progress to a JSON file after every successfully upserted batch. If the migration is interrupted (crash, network error, Ctrl+C), re-run the same command — it will skip already-processed records and continue from where it left off.

| Item | Value |
|---|---|
| Checkpoint file (inside container) | `/app/data/checkpoints/migration.json` |
| Checkpoint file (on host) | `./data/checkpoints/migration.json` |
| Resume (default) | `RESUME=true` — continue from checkpoint |
| Fresh start | `RESUME=false` — clear checkpoint, start from record 0 |

The checkpoint stores: total records processed, the last cursor position, the last batch number, and a completion flag. Cursor format differs by source: integer count for Milvus and ChromaDB, scroll UUID for Qdrant.

---

## Limitations

### Milvus quantized vector types are not supported

The Milvus client returns quantized vector types (`FLOAT16_VECTOR`, `INT8_VECTOR`, `BFLOAT16_VECTOR`, `BINARY_VECTOR`) as raw bytes — the original float32 values are lost after quantization and cannot be recovered. The script detects this at startup and exits with an explanation.

Only `FLOAT_VECTOR` (native float32) is supported as a source. If your Milvus collection uses a quantized type, re-embed your data as `FLOAT_VECTOR` before migrating.

### Single dense vector field per collection

If a Milvus or Qdrant collection has multiple dense vector fields, only the first detected one is migrated. Additional dense fields are ignored.

### No multivector support

Collections with multivector (array of vectors per record) are not supported. The script will reject them at schema detection.

### ChromaDB has no native sparse storage

ChromaDB does not store sparse vectors. The only way to produce a hybrid Endee index from a ChromaDB source is to set `SPARSE_ALGO=endee/bm25`, which generates sparse vectors on the fly from the document text field during migration.

### Sparse generation requires a text field

When using `SPARSE_ALGO=endee/bm25` with Milvus or Qdrant, the field named in `SPARSE_TEXT_FIELD` must exist and must contain non-empty text in every record. If a batch has all-empty text, the migration aborts.

### No schema migration

Only vectors and their associated metadata are migrated. Index-level configuration (HNSW parameters, space type, precision) must be set explicitly in `.env` — these are not automatically copied from the source.

### `hybrid` → `dense` downgrade is blocked

If your source collection is hybrid (has stored sparse vectors), you cannot migrate it to a dense-only Endee index. This would silently discard sparse data. Set `TARGET_TYPE=hybrid` to preserve sparse vectors.

### No support for Qdrant multivector (named vectors with multiple dense fields)

The Qdrant source connector picks the first detected dense vector config. If your collection has multiple named dense vector fields, only one is used and the others are dropped.
