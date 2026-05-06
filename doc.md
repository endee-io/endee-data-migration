# Endee Migration Tool - Documentation

Migrate vector data from **Qdrant** or **Milvus** into **Endee**. The tool ships as a Docker image and supports dense-only and hybrid (dense + sparse) collections.

---



## Setup
### Prerequisites
| Requirement | Details |
|---|---|
| API access | API key should have data access permission of Collection or Index |
| Source Collection | Source DB collection must be present |
| Services Reachable |Make sure both Endee and Source DB services are up and reachable from machine where migration script will run.
| Source & Endee Server | Both Source and Endee should run on different servers |
| Migration Script Server | If migration & endee running on same server and in containers both should be on same network |

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
TARGET_URL=http://<endee-server-ip>:8080 # IP OF ENDEE SERVER RUNNING ON
TARGET_API_KEY=your-endee-api-key # IF USING ENDEE DEV SERVER GET THE API KEY (NO NEED TO SET TARGET URL IN THAT CASE )
TARGET_COLLECTION=my_index # ENDEE INDEX NAME
CLEAR_CHECKPOINT=true # FALSE IF YOU WANT TO RESUME THE MIGRATION
FILTER_FIELDS=field1,field2 # ADD FILTER_FIELDS ONLY IF YOU WANT FIELDS TO BE IN ENDEE FILTER AND NOT IN META
PRECISION=INT16 
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
- Port `6333` (Qdrant) or `19530` (Milvus) is open on the source server.
- The Endee API port is open on the Endee server.
- The migration container has outbound access to both.

---


## Running the Migration

### Available Migration Types

| Type | Source | Vector Mode |
|---|---|---|
| `milvus-to-endee-dense` | Milvus | Dense vectors only |
| `milvus-to-endee-hybrid` | Milvus | Dense + sparse vectors |
| `qdrant-to-endee-dense` | Qdrant | Dense vectors only |
| `qdrant-to-endee-hybrid` | Qdrant | Named dense (`dense`) + sparse (`sparse_keywords`) vectors |

Pass the type as the first argument to the container, or set `MIGRATION_TYPE` in your `.env`.

### Override precision manually
| Item | Value / Options | When to use |
|---|---|---|
| Override setting | Set `PRECISION` in `.env` | Use this to override auto-detected precision |
| Valid values | `FLOAT32`, `FLOAT16`, `INT16`, `INT8`, `BINARY` | Choose based on your accuracy and storage needs |
| Recommended option | `INT16` | Good balance of accuracy and storage efficiency when source has no explicit quantization |
| Full precision option | `FLOAT32` | Preserve full original precision |


### Checkpoint and Resume

| Item | Details |
|---|---|
| Checkpoint behavior | The script saves progress to a JSON file after every successful batch. |
| Resume behavior | If migration is interrupted (network error, container restart, manual stop), re-run the same command with `--clear_checkpoint=False` to continue from the last saved point. |
| Checkpoint path (container) | `/app/data/checkpoints/migration.json` |
| Checkpoint path (host) | `./data/checkpoints/migration.json` |
| Volume mount required | `-v $(pwd)/data:/app/data` |

### FILTER FIELDS
| Source | Input fields | `FILTER_FIELDS` setting | Endee `filter` | Endee `meta` |
|---|---|---|---|---|
| Qdrant | Payload fields | Not set | Empty | All payload fields |
| Milvus | Metadata fields | Not set | Empty | All metadata fields |
| Qdrant / Milvus | Payload (Qdrant) / Metadata (Milvus) | Set (for example: `category,status,year`) | Only fields listed in `FILTER_FIELDS` | Remaining fields not listed in `FILTER_FIELDS` |
---

### Qdrant → Endee record field mapping


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

### Precision Milvus and Endee DataType Mapping
---

| Milvus DataType | Endee Precision | Notes |
|---|---|---|
| `FLOAT_VECTOR` | `FLOAT32` | Default 32-bit float vectors |
| `FLOAT16_VECTOR` | `FLOAT16` | 16-bit half precision vectors |
| `BFLOAT16_VECTOR` | `FLOAT16` | Mapped to closest supported Endee precision |
| `BINARY_VECTOR` | `BINARY2` | Binary vectors |
| no quantization config | `INT16` | Default precision |


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


## Limitations
- No incremental migration - new records added to the source after migration started will not be picked up. Run a fresh migration with `--clear_checkpoint` as `True` to re-migrate.
- No real-time sync - this is a one-time, point-in-time copy only.
- Do not run the source database and Endee on the same server - resource contention will degrade performance and risk crashes.
- Multivector mode is not supported.
- Only Qdrant and Milvus are supported as sources.
- Only Endee is supported as the target.
---

## Infrastructure Requirements

### Migration + Endee on the same server

These are the combined requirements when both the migration container and Endee are running on the same machine.

| Dataset Size | RAM | CPU | Disk |
|---|---|---|---|
| 1 million records | 8 GB | 4 cores | 20 GB |
| 10 million records | 32 GB | 8 cores | 50 GB |

Disk usage covers the Endee index data, checkpoint files, and container images. Most of the RAM and CPU is consumed by Endee itself, not the migration script.

---

### Migration running on a separate server

When the migration container runs on its own dedicated server (no Endee on that machine), the resource requirements are much lighter. The script uses an async producer-consumer queue with a default max size of 5 batches. Each batch holds 1,000 records. At float32 precision and dimension 768, one batch is roughly 3 MB in memory, so the queue holds at most ~15 MB of vector data at any time. The script is network-bound, not CPU or memory-bound.

| Dataset Size | RAM | CPU | Disk |
|---|---|---|---|
| 1 million records | 4 GB | 1-2 cores | 4 GB |
| 10 million records | 8 GB | 2 cores | 8 GB |

You can reduce memory further by lowering `BATCH_SIZE` and `MAX_QUEUE_SIZE` if needed.
