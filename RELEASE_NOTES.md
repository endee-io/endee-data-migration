# Release notes — Vector migration scripts

**Tool:** Endee migration (Milvus & Qdrant → Endee)  
**Version (banner):** 1.0.0  
**Last updated:** March 2026  

This document summarizes notable changes and behavior for the Python migration scripts under `scripts/` and the Docker entrypoint.

---

## Supported migration types

| Type | Script | Description |
|------|--------|-------------|
| `milvus-to-endee-dense` | `milvus_to_endee_dense_migration.py` | Milvus dense vectors → Endee |
| `milvus-to-endee-hybrid` | `milvus_to_endee_hybrid_migration.py` | Milvus dense + sparse (hybrid) → Endee |
| `qdrant-to-endee-dense` | `qdrant_to_endee_dense_migration.py` | Qdrant dense vectors → Endee |
| `qdrant-to-endee-hybrid` | `qdrant_to_endee_hybrid_migration.py` | Qdrant dense + sparse (hybrid) → Endee |

---

## Highlights (recent generations)

### Async execution

- **Qdrant** migrations use an async pipeline (producer/consumer style with `asyncio`) for fetching and upserting.
- **Milvus** dense and hybrid migrations run their main migration path via **`asyncio.run(...)`** for consistent async I/O and batch processing.

### Filters, metadata, and payload defaults

- Payload / non-vector fields are mapped into **Endee `meta`** and **filter** fields according to script logic and optional **`FILTER_FIELDS`** (comma-separated). Only fields you list are promoted for filtering; the rest stay in metadata.
- **Change:** Default / sample payload handling was adjusted so that **payload data is placed in `meta` by default** rather than being treated as filter fields unless you explicitly configure filters (see commit *“default payload move to meta instead of filter”*).  
  **Action:** After upgrading, verify `FILTER_FIELDS` matches the fields you need for filtered queries in Endee.

### Vector precision & quantization

- Milvus field types are mapped to Endee **`Precision`** (e.g. float32, float16, binary) where applicable.
- Quantization-related handling was added/aligned with Endee expectations (see history: *“quantization handled”*).

### Checkpoints & resume

- JSON (or `orjson` where used) **checkpoint files** track progress (offsets, batch numbers, processed counts) so migrations can **resume** after interruption.
- Checkpoint path issues were addressed (see *“checkpoint file issue fixed”*). Prefer absolute paths or a stable volume mount (e.g. `./data`) in Docker.

### Configuration via environment

- Scripts load **`.env`** via `python-dotenv` / `dotenv` so **`SOURCE_*`**, **`TARGET_*`**, batch sizes, checkpoint path, debug flags, etc. can be set without long CLI strings.
- **Docker:** Migration type can be supplied as the **first CLI argument** or via **`MIGRATION_TYPE`** in the environment (CLI takes priority). The image no longer assumes a hard-coded migration type in the command line alone (see *“removed migration type from docker command”* — use env or explicit arg).

---

## CLI & Docker usage (short)

- **Scripts:** Run with `python scripts/<script>.py` and flags such as `--source_url`, `--source_collection`, `--target_url`, `--target_api_key`, `--target_collection`, `--batch_size`, `--checkpoint_file`, `--clear_checkpoint`, `--debug`, etc. (see each script’s `argparse` help).
- **Docker:** `docker run ... vector-migration <migration-type> [options]` or set `MIGRATION_TYPE` in `.env` / `--env-file`. See `entrypoint.sh` and project README for examples.

---

## Upgrade / migration notes

1. **Re-read `FILTER_FIELDS`:** If you relied on previous default mapping of payload keys to filters, compare with the new **meta-first** behavior and set `FILTER_FIELDS` explicitly for anything that must be filterable.
2. **Checkpoints:** If you change script versions mid-migration, keep the same **`CHECKPOINT_FILE`** path and verify offset semantics still match your source (Qdrant vs Milvus differ slightly in checkpoint fields).
3. **Dependencies:** Align `requirements.txt` with your runtime (e.g. `pymilvus`, `qdrant-client`, `endee` client version). Pin versions in production.

---

## Known limitations (typical)

- **Dense-only scripts** may reject or not support **multivector** mode — use the appropriate script and flags as documented in each file.
- Large collections: tune **`BATCH_SIZE`**, **`UPSERT_SIZE`**, and connection limits; monitor Endee and source DB rate limits.

---

## History (git summary)

| Area | Notes |
|------|--------|
| Initial scripts | Milvus/Qdrant → Endee data migration |
| Env | Variables loaded from `.env` |
| Checkpoint | Resume behavior and file handling fixes |
| Quantization | Alignment with Endee |
| Filters / meta | Explicit filter vs metadata mapping |
| Async | Qdrant async; Milvus dense & hybrid async |
| Docker | Migration type via env or argument |
| Payload default | Prefer **meta** unless filters are configured |

---

For detailed flags and defaults, run:

```bash
python scripts/milvus_to_endee_dense_migration.py --help
python scripts/qdrant_to_endee_hybrid_migration.py --help
# … etc.
```

Or with Docker:

```bash
docker run --rm vector-migration --help
docker run --rm vector-migration milvus-to-endee-dense --help
```
