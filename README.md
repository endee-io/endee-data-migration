# Endee Migration Tool

Migrate vector collections from **Qdrant** or **Milvus** to **Endee** using a Dockerized producer-consumer pipeline with checkpoint resume support.

---

## Supported Migrations

| Migration Type | Command |
|---|---|
| Qdrant (Dense) → Endee | `qdrant-to-endee-dense` |
| Qdrant (Hybrid: Dense + Sparse) → Endee | `qdrant-to-endee-hybrid` |
| Milvus (Dense) → Endee | `milvus-to-endee-dense` |
| Milvus (Hybrid: Dense + Sparse) → Endee | `milvus-to-endee-hybrid` |

---

## Project Structure

```
.
├── cmd.sh                          # Build and run script
├── .env                            # Configuration (copy from .env.example)
├── data/
│   └── checkpoints/               # Checkpoint files (auto-created, mount as volume)
├── scripts/
│   ├── qdrant_to_endee_dense_migration.py
│   ├── qdrant_to_endee_hybrid_migration.py
│   ├── milvus_to_endee_dense_migration.py
│   └── milvus_to_endee_hybrid_migration.py
├── entrypoint.sh                   # Docker entrypoint
└── Dockerfile
```

---

## Quick Start

### 1. Configure your `.env` file

Copy and edit the environment file:

```bash
cp .env.example .env
```

Set your source database, target Endee credentials, and migration type. See [Configuration Reference](#configuration-reference) below for all options.

### 2. Build the Docker image

```bash
docker build -t vector-migration:latest .
```

### 3. Run the migration

Edit `cmd.sh` to select your migration type, then run:

```bash
bash cmd.sh
```

Or run directly with Docker:

```bash
docker run \
  --rm \
  --network vector-net \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  vector-migration:latest \
  qdrant-to-endee-hybrid
```

---

## Configuration Reference

All settings can be provided via `.env` file or as environment variables passed to Docker.

### Migration Type

```env
# Choose one:
# qdrant-to-endee-dense
# qdrant-to-endee-hybrid
# milvus-to-endee-dense
# milvus-to-endee-hybrid
MIGRATION_TYPE=qdrant-to-endee-dense
```

### Source — Qdrant

```env
SOURCE_URL=http://your-qdrant-host
SOURCE_PORT=6333
SOURCE_API_KEY=                      # Leave empty if no auth
SOURCE_COLLECTION=your_collection
USE_HTTPS=false
```

### Source — Milvus

```env
SOURCE_URL=http://your-milvus-host
SOURCE_PORT=19530
SOURCE_API_KEY=your_milvus_token     # Leave empty if no auth
SOURCE_COLLECTION=your_collection
IS_MULTIVECTOR=false
```

### Target — Endee

```env
TARGET_URL=http://your-endee-host:8080   # Omit for Endee Cloud
TARGET_API_KEY=your_endee_api_key
TARGET_COLLECTION=your_index_name
```

### Performance

```env
BATCH_SIZE=1000        # Records fetched per batch from source
UPSERT_SIZE=1000       # Records upserted per chunk to Endee
MAX_QUEUE_SIZE=5       # Max batches buffered in memory between producer and consumer
```

### Index Parameters (Milvus only — auto-detected from schema)

```env
# These are read automatically from Milvus collection schema.
# Override only if needed:
# SPACE_TYPE=cosine    # cosine | l2 | ip
# M=16
# EF_CONSTRUCT=128
```

### Checkpoint / Resume

```env
CHECKPOINT_FILE=/app/data/checkpoints/migration.json
# CLEAR_CHECKPOINT=true    # Uncomment to start fresh
```

### Filter Fields

```env
# Comma-separated list of payload fields to use as Endee filter fields.
# All other fields go to meta.
# Endee filter fields must be scalar types (str, int, float, bool).
# Lists and dicts must go to meta — do not include them here.
FILTER_FIELDS=company,region,sector,document_type
```

### Debug

```env
DEBUG=false    # Set true for verbose logging
```

---

## Full `.env` Example

```env
# ── Migration ────────────────────────────────────────────────────
MIGRATION_TYPE=qdrant-to-endee-hybrid

# ── Source (Qdrant) ──────────────────────────────────────────────
SOURCE_URL=http://35.207.217.185
SOURCE_PORT=6333
SOURCE_API_KEY=
SOURCE_COLLECTION=my_hybrid_collection
USE_HTTPS=false

# ── Target (Endee) ───────────────────────────────────────────────
TARGET_API_KEY=your_endee_api_key
TARGET_COLLECTION=my_endee_index

# ── Performance ──────────────────────────────────────────────────
BATCH_SIZE=1000
UPSERT_SIZE=1000
MAX_QUEUE_SIZE=5

# ── Filter fields ────────────────────────────────────────────────
FILTER_FIELDS=company,region,sector,document_type,page_number

# ── Checkpoint ───────────────────────────────────────────────────
CHECKPOINT_FILE=/app/data/checkpoints/migration.json

# ── Debug ────────────────────────────────────────────────────────
DEBUG=false
```

---

## cmd.sh Reference

```bash
#!/bin/bash
docker build -t vector-migration:latest .

docker run \
  --rm \
  --network vector-net \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  vector-migration:latest \
  qdrant-to-endee-hybrid   # ← change this line to switch migration type
  # qdrant-to-endee-dense
  # milvus-to-endee-hybrid
  # milvus-to-endee-dense
```

---

## Checkpoint & Resume

Migration progress is saved after every successfully upserted batch. If the migration is interrupted for any reason (network error, Ctrl+C, container restart), simply rerun the same command — it will resume from where it left off automatically.

```bash
# Resume from last checkpoint (default — just rerun):
bash cmd.sh

# Start fresh (discard checkpoint):
# Set in .env:
CLEAR_CHECKPOINT=true
```

The checkpoint file is stored at `CHECKPOINT_FILE` (default: `/app/data/checkpoints/migration.json`). Since `/app/data` is mounted as a volume, the checkpoint persists across container restarts.

**Checkpoint file example:**
```json
{
  "processed_count": 50000,
  "last_offset": "abc123-uuid-...",
  "batch_number": 50
}
```

---

## Filter Fields vs Meta Fields

Endee has two payload buckets per record:

| Bucket | Purpose | Allowed types |
|---|---|---|
| `filter` | Used for filtering search results | `str`, `int`, `float`, `bool` only |
| `meta` | Stored metadata, not filterable | Any type including `list`, `dict` |

**Important:** If any field in `FILTER_FIELDS` contains a `list` or `dict` value, Endee will reject the record with `MDBX_BAD_VALSIZE`. Always use scalar values in filter fields.

```env
# ✓ Safe — scalar fields
FILTER_FIELDS=company,region,sector,page_number

# ✗ Will fail — 'product' is a list in this dataset
FILTER_FIELDS=company,product
```

If `FILTER_FIELDS` is empty, all payload fields go to `filter`. Fields with non-scalar values should always be excluded from `FILTER_FIELDS` and will automatically land in `meta`.

---

## Architecture

Each migration runs a **producer-consumer pipeline** inside `asyncio`:

```
migrate()                     ← sync setup: connect, detect schema, create index
    └── asyncio.run(async_migrate())
            ├── async_producer()   ← fetches batches from source into bounded queue
            └── async_consumer()   ← reads from queue, upserts to Endee, saves checkpoint
```

- **Bounded queue** (`MAX_QUEUE_SIZE=5`) prevents memory overflow — producer pauses when queue is full.
- **All blocking SDK calls** (Qdrant scroll, Milvus query, Endee upsert) run in `loop.run_in_executor()` so the event loop is never frozen.
- **Parallel upsert** — chunks within a batch are upserted concurrently via `asyncio.gather()`.
- **Exponential backoff retry** — failed chunks are retried up to 3 times (1s, 2s, 4s).
- **Graceful shutdown** — `SIGINT`/`SIGTERM` (Ctrl+C or `docker stop`) saves checkpoint and exits cleanly.

---

## Troubleshooting

### Migration hangs after a failure

The consumer failed and the queue is full — the producer is blocked in `queue.put()`. Kill the container and rerun. The fix is to add a queue drain in the consumer's failure path (see source code comments).

To debug a hang:
```bash
# Inside the container
pip install py-spy
py-spy dump --pid 1
```

### `MDBX_BAD_VALSIZE` error

A filter field contains a non-scalar value (usually a `list`). Remove it from `FILTER_FIELDS` — it will go to `meta` instead.

```env
# If 'product' is a list:
FILTER_FIELDS=company,region,sector   # ← remove 'product'
```

### `Cannot allocate memory` on index creation

The Endee server is out of memory — too many indexes open. Delete unused indexes on the Endee server before retrying.

```bash
free -h          # check available RAM on Endee server
docker stats     # check container memory usage
```

### Qdrant client version warning

```
UserWarning: Qdrant client version 1.16.2 is incompatible with server version 1.13.6
```

Downgrade the client to match your server version, or add `check_compatibility=False` to the `QdrantClient` constructor. Migration will still work in most cases despite the warning.

### URL has trailing space

```
Failed to resolve 'your-host%20'
```

Check `TARGET_URL` or `SOURCE_URL` in `.env` for trailing whitespace.

---

## Requirements

- Docker
- Source database accessible from the Docker network (`--network vector-net` or host network)
- Endee instance running and accessible
- Sufficient disk space for checkpoint file (tiny — JSON, a few KB)
- Sufficient RAM for `MAX_QUEUE_SIZE × BATCH_SIZE` records in memory at once