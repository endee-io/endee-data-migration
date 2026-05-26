# Endee Migration Tool - Documentation

Migrate vector data from **Qdrant** / **Milvus** / **Chromadb** into **Endee**. The tool ships as a Docker image and supports dense-only and hybrid (dense + sparse) collections.

---



## Setup
### Prerequisites
| Requirement | Details |
|---|---|
| API access | API key should have data access permission of Collection or Index |
| Source Collection | Source DB collection must be present |
| Services Reachable |Make sure both Endee and Source DB services are up and reachable from machine where migration script will run.
| Source & Endee Server | Both Source and Endee should run on different servers |

---


### Setup Steps

**Step 1 - Clone Migration Script:**
```bash
git clone https://github.com/endee-io/endee-data-migration.git
cd endee-data-migration/
```

**Step 2 - Create .env from .env.example:**
```bash
cp .env.example .env
```

**Step 3 - Update .env file:**
```bash
# .env
MIGRATION_TYPE=milvus-to-endee-dense
SOURCE_URL=http://<milvus-server-ip>
SOURCE_PORT=19530
SOURCE_COLLECTION=my_collection # SOURCE DATABASE COLLECTION NAME FROM WHERE DATA NEEDS TO MIGRATE
SOURCE_DB=default # FOR MILVUS
TARGET_URL=http://<endee-server-ip>:8080 # IP OF ENDEE SERVER RUNNING ON
TARGET_API_KEY=your-endee-api-key # IF USING ENDEE DEV SERVER GET THE API KEY (NO NEED TO SET TARGET URL IN THAT CASE )
TARGET_COLLECTION=my_index # ENDEE INDEX NAME
RESUME=false # true IF YOU WANT TO RESUME THE MIGRATION
FILTER_FIELDS=field1,field2 # ADD FILTER_FIELDS ONLY IF YOU WANT FIELDS TO BE IN ENDEE FILTER AND NOT IN META
PRECISION=INT16
SPACE_TYPE=cosine # DISTANCE METRIC: cosine | l2 | ip
PRECISION=int16
M=16
EF_CONSTRUCT=128
SPACE_TYPE=cosine
```


**Step 4 - Make scripts executable:**
```bash
chmod +x entrypoint.sh
chmod +x cmd.sh
```

**Step 5 - Start Migration:**

```bash
./cmd.sh
```

Make sure:
- Port `6333` (Qdrant) or `19530` (Milvus) or `8000` (Chroma) is open on the source server.
- The Endee API port is open on the Endee server.
- The migration container has outbound access to both.

---

## Environment Variables Reference

| Variable | Should You Change? | Description | Example Value |
|---|---|---|---|
| `MIGRATION_TYPE` | Required | Selects which migration script to run. Must match your source DB and collection type. See available values below. | `milvus-to-endee-dense` |
| `SOURCE_URL` | Required | Full URL of your source Qdrant or Milvus server | `http://192.168.1.10` |
| `SOURCE_PORT` | Required | Port your source DB listens on | `6333` (Qdrant), `19530` (Milvus) |
| `SOURCE_API_KEY` | If auth enabled | API key or token for your source DB. Leave blank if auth is disabled. | `your-qdrant-api-key` |
| `SOURCE_COLLECTION` | Required | Name of the collection in the source DB to migrate from | `my_collection` |
| `TARGET_URL` | Required | URL of your Endee server | `http://192.168.1.20:8080` |
| `TARGET_API_KEY` | Required | Endee API key with write access | `your-endee-api-key` |
| `TARGET_COLLECTION` | Required | Name of the Endee index to create or write into. Created automatically if it does not exist. | `my_index` |
| `FILTER_FIELDS` | Optional | Comma-separated fields to store in Endee's `filter` slot for fast filtering. All other fields go to `meta`. Leave blank to put everything in `meta`. | `category,status,year` |
| `PRECISION` | Required | Vector storage precision in Endee. Explicitly setting it is recommended. See precision table below. | `INT16` |
| `SPACE_TYPE` | Required | Distance metric for the Endee index. Must match the metric used when the source collection was built. | `cosine` |
| `IS_MULTIVECTOR` | Do not change | Reserved for future use. Multivector mode is not currently supported. Always leave as `false`. | `false` |
| `M` | Required | Number of bidirectional links per node in the HNSW graph. Higher = better recall but slower inserts and more memory. Auto-read from source collection if not set. Cannot be zero or negative. | `16` |
| `EF_CONSTRUCT` | Required | Search beam width during index construction. Higher = better graph quality but slower inserts. Auto-read from source collection if not set. Cannot be zero or negative. | `128` |
| `BATCH_SIZE` | Required | Number of records fetched from the source DB per batch. Lower this if you hit memory limits. | `1000` |
| `UPSERT_SIZE` | Required | Number of records sent to Endee per upsert call. Lower this if upserts time out. | `1000` |
| `MAX_QUEUE_SIZE` | Required | Max number of fetched batches held in memory waiting to be upserted. Controls memory pressure between producer and consumer. | `5` |
| `RESUME` | Required | Set `true` to continue from the last saved checkpoint. Set `false` to start completely fresh (clears the checkpoint). | `true` |
| `CHECKPOINT_FILE` | Do not change | Path inside the container where progress is saved after every successful batch. Changing this breaks the resume feature. | `/app/data/checkpoints/migration.json` |
| `DEBUG` | Do Not Change | `true` to enable verbose debug logging. Leave `false` in normal operation. | `false` |


**Available `MIGRATION_TYPE` values:**


| Type | Source | Vector Mode |
|---|---|---|
| `milvus-to-endee-dense` | Milvus | Dense vectors only |
| `milvus-to-endee-hybrid` | Milvus | Dense + sparse vectors |
| `qdrant-to-endee-dense` | Qdrant | Dense vectors only |
| `qdrant-to-endee-hybrid` | Qdrant | Named dense (`dense`) + sparse (`sparse_keywords`) vectors |
| `chroma-to-endee-hybrid` | ChromaDB | Dense -> Hybrid (BM25 sparse generated via `endee/bm25`) |

Set `MIGRATION_TYPE` in your `.env`.
> **Note:** ChromaDB hybrid migration requires document text to be stored alongside embeddings in the source collection. If documents are missing, sparse BM25 vectors cannot be generated and the migration will exit with an error.


### Override precision manually
| Item | Value / Options | When to use |
|---|---|---|
| Override setting | Set `PRECISION` in `.env` | Use this to override auto-detected precision |
| Valid values | `FLOAT32`, `FLOAT16`, `INT16`, `INT8`, `BINARY` | Choose based on your accuracy and storage needs |
| Recommended option | `INT16` | Good balance of accuracy and storage efficiency when source has no explicit quantization |
| Full precision option | `FLOAT32` | Preserve full original precision |

### Override Space Type (`SPACE_TYPE`)

`SPACE_TYPE` is **required** for all migration types. It sets the distance metric of the Endee index.

| Value | Distance metric | When to use |
|---|---|---|
| `cosine` | Cosine similarity | Default for most embedding models; ChromaDB default |
| `l2` | Euclidean (L2) distance | Use when source collection was built with L2 |
| `ip` | Inner product | Use when source collection was built with dot-product / inner-product |


### Checkpoint and Resume

| Item | Details |
|---|---|
| Checkpoint behavior | The script saves progress to a JSON file after every successful batch. |
| Resume behavior | If migration is interrupted (network error, container restart, manual stop), re-run the same command with `Resume=true` to continue from the last saved point. |
| Checkpoint path (container) | `/app/data/checkpoints/migration.json` |
| Checkpoint path (host) | `./data/checkpoints/migration.json` |
| Volume mount required | `-v $(pwd)/data:/app/data` |



---

### FILTER FIELDS
| Source | Input fields | `FILTER_FIELDS` setting | Endee `filter` | Endee `meta` |
|---|---|---|---|---|
| Qdrant | Payload fields | Not set | Empty | All payload fields |
| Milvus | Metadata fields | Not set | Empty | All metadata fields |
| ChromaDB | Metadata fields | Not set | Empty | All metadata fields + optional `document` text |
| Qdrant / Milvus / ChromaDB | Payload or metadata | Set (for example: `category,status,year`) | Only fields listed in `FILTER_FIELDS` | Remaining fields not listed in `FILTER_FIELDS` |
---

### Qdrant -> Endee record field mapping


| Qdrant concept | Endee field | Mapping |
|---|---|---|
| Point ID (primary key) | `id` | `str(point.id)` |
| Dense vector | `vector` | `point.vector` from scroll (`with_vectors=True`). Expects a single dense vector; `--is_multivector` is rejected (script raises). |
| Sparse vector | `sparse_indices`, `sparse_values` | Used in hybrid flow when sparse data is present; not used in dense-only flow. |
| Payload | `filter`, `meta` | Keys listed in `FILTER_FIELDS` go to `filter`; every other payload key goes to `meta`. If `FILTER_FIELDS` is unset, the full payload goes to `meta` and `filter` is `{}`. |

---

### Milvus → Endee record field mapping

Rows are built in `convert_records` in the Milvus migration scripts. Schema detection walks `describe_collection` fields: primary key, vector field(s), and everything else as metadata. Endee uses the same keys as in the Qdrant flow (`id`, `vector`, `filter`, `meta`, and for hybrid, `sparse_indices` / `sparse_values` when applicable).

| Milvus concept | Endee field | Mapping |
|---|---|---|
| Primary key (`is_primary`) | `id` | `str(record[primary_key_field])` using the detected primary-key field name. |
| Dense vector | `vector` | Value is read from the first detected dense vector field and passed through `decode_vector` (bytes/list handling per field type). Additional dense vector fields in the same collection are not written to Endee by these scripts. |
| Sparse vector (`SPARSE_FLOAT_VECTOR`) | `sparse_indices`, `sparse_values` | Used in hybrid flow when sparse data exists. Sparse dict entries (index -> weight) are sorted and mapped to `sparse_indices` (ints) and `sparse_values` (floats). In dense-only flow, sparse fields are not used. |
| Non-vector fields (metadata columns) | `filter`, `meta` | Only non-primary, non-vector schema fields are eligible (`other_fields_meta`). If `FILTER_FIELDS` is set, listed keys go to `filter` and remaining eligible keys go to `meta`; if unset, eligible fields go to `meta` and `filter` is `{}`. |

---

### ChromaDB → Endee record field mapping

Rows are built in `_convert_batch` in the ChromaDB migration script. Sparse vectors are generated from document text using Endee's own BM25 model (`endee/bm25`) via the `endee-model` package — this is the only encoder compatible with indexes created with `sparse_model="endee_bm25"`.

| ChromaDB concept | Endee field | Mapping |
|---|---|---|
| Record ID | `id` | `str(doc_id)` |
| Embedding | `vector` | Dense embedding stored in the ChromaDB collection. Records with no embedding are skipped with a warning. |
| Document text | `sparse_indices`, `sparse_values` | BM25 sparse encoding via `SparseModel("endee/bm25").embed(documents)`. Records with no document text in the entire batch cause the migration to abort. |
| Document text | `meta["document"]` | Stored in `meta` alongside other metadata unless `--no-store-document` / `NO_STORE_DOCUMENT=true` is set. |
| Metadata fields | `filter`, `meta` | Keys listed in `FILTER_FIELDS` go to `filter`; every other metadata key goes to `meta`. If `FILTER_FIELDS` is unset, all metadata goes to `meta` and `filter` is `{}`. |

> **Space type for ChromaDB:** ChromaDB does not expose the collection distance metric in a reliably accessible way, so `SPACE_TYPE` must always be set explicitly for this migration type. Use the same metric you used when building the collection (`cosine` is the ChromaDB default).

---



### Precision Milvus and Endee DataType Mapping
---

| Milvus DataType | Endee Precision | Notes |
|---|---|---|
| `FLOAT_VECTOR` | `FLOAT32` | Default 32-bit float vectors |
| `FLOAT16_VECTOR` | `FLOAT16` | 16-bit half precision vectors |
| `BFLOAT16_VECTOR` | `FLOAT16` | Mapped to closest supported Endee precision |
| `BINARY_VECTOR` | `BINARY2` | Binary vectors |
<!-- | no quantization config | `INT16` | Default precision | -->

### Precision Qdrant and Endee DataType Mapping

| Qdrant Quantization Config | Endee Precision | Notes |
|---|---|---|
| `scalar` present | `INT8` | Qdrant scalar quantization is treated as int8 |
| `binary` present, no `query_encoding`, no explicit `encoding` | `BINARY2` | Supported binary quantization path |
| no quantization config | `INT16` | Default precision |
| quantization present but none of `scalar`/`binary`/`product` | `INT16` | Fallback/default path |
| `product` present | Not supported | Migration raises an error |
| `binary` with `query_encoding` | Not supported | Asymmetric quantization raises an error |
| `binary` with explicit `encoding` | Not supported | Invalid encoding raises an error |


---
Note: The script exits if quantization is not configured in the environment variables.

