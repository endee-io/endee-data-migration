# Adding a New Source or Target Connector

This guide covers exactly what code to write and where to register it.
No other files need to change.

---

## Table of Contents

1. [Adding a New Source](#1-adding-a-new-source)
2. [Adding a New Target](#2-adding-a-new-target)
3. [Adding a New Sparse Encoder](#3-adding-a-new-sparse-encoder)
4. [Key Types Reference](#4-key-types-reference)

---

## 1. Adding a New Source

### Files to create / edit

| Action | File |
|--------|------|
| Create | `sources/yourdb_source.py` |
| Edit   | `migrate.py` — add to `--from` choices + `SOURCE_REGISTRY` |

---

### Step 1 — Create `sources/yourdb_source.py`

```python
# sources/yourdb_source.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base_source import BaseSource
from core.schema import FieldRole, FieldSchema, FieldType, MigrationRow, RowSchema

logger = logging.getLogger(__name__)


class YourDBSource(BaseSource):
    """
    Source connector for YourDB.

    Cursor format: <describe what your cursor is — int offset, UUID token, etc.>

    RowSchema slot layout:
      SLOT 0  → ID           (STRING / INT)
      SLOT 1  → DENSE_VECTOR (DENSE_VECTOR)
      SLOT 2  → JSON payload (METADATA)
      # Add SPARSE_VECTOR slot between dense and payload if hybrid
    """

    def __init__(
        self,
        url:        str,
        collection: str,
        api_key:    str = "",
        # add any other connection params your DB needs
    ):
        self.url        = url
        self.collection = collection
        self.api_key    = api_key

        self._client: Any               = None
        self._schema: Optional[RowSchema] = None

        # track slot positions — set once in detect_schema()
        self._dense_slot:   int = -1
        self._sparse_slot:  int = -1   # leave -1 if dense-only
        self._payload_slot: int = -1

    # ── 1. connect ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Establish the DB connection. Called once before detect_schema().
        Store the client on self. Call sys.exit(1) on failure (pipeline expects it).
        """
        import sys
        try:
            # self._client = YourDBClient(url=self.url, api_key=self.api_key)
            # total = self._client.count(self.collection)
            logger.info(f"Connected to YourDB | collection='{self.collection}'")
        except Exception as e:
            logger.error(f"Failed to connect to YourDB: {e}")
            sys.exit(1)

    # ── 2. detect_schema ─────────────────────────────────────────────────────

    def detect_schema(self) -> RowSchema:
        """
        Inspect the collection, build and return a RowSchema.

        Rules:
          - Append FieldSchema objects in slot order (0, 1, 2, …).
          - Save slot indices on self (_dense_slot, _sparse_slot, _payload_slot).
          - Set is_hybrid=True only if a SPARSE_VECTOR field is in the schema.
          - Set canonical_precision to the source's quantization ("float32" if unknown).
          - Read HNSW params from collection metadata if available.
          - Call self._validate_schema() at the end for any extra checks.
        """
        schema_fields: List[FieldSchema] = []

        # --- SLOT 0: primary key ---
        schema_fields.append(FieldSchema(
            name       = "id",
            field_type = FieldType.STRING,   # or INT if your DB uses int ids
            role       = FieldRole.ID,
        ))

        # --- SLOT 1: dense vector ---
        dimension = self._detect_dimension()   # implement helper below
        schema_fields.append(FieldSchema(
            name       = "embedding",
            field_type = FieldType.DENSE_VECTOR,
            role       = FieldRole.DENSE_VECTOR,
            dimension  = dimension,
        ))
        self._dense_slot = 1

        # --- SLOT 2 (optional): sparse vector — only if source is hybrid ---
        # if source_has_sparse:
        #     schema_fields.append(FieldSchema(
        #         name       = "sparse_vector",
        #         field_type = FieldType.SPARSE_VECTOR,
        #         role       = FieldRole.SPARSE_VECTOR,
        #     ))
        #     self._sparse_slot = 2

        # --- LAST SLOT: metadata payload ---
        schema_fields.append(FieldSchema(
            name       = "payload",
            field_type = FieldType.JSON,
            role       = FieldRole.METADATA,
        ))
        self._payload_slot = len(schema_fields) - 1

        self._schema = RowSchema(
            fields              = schema_fields,
            dimension           = dimension,
            space_type          = "cosine",     # read from collection if available
            is_hybrid           = self._sparse_slot >= 0,
            canonical_precision = "float32",    # read from collection if available
        )
        return self._schema

    def _detect_dimension(self) -> int:
        """Peek at one record to get vector length."""
        # sample = self._client.get(self.collection, limit=1, include_vectors=True)
        # return len(sample[0].vector)
        raise NotImplementedError

    def _validate_schema(self):
        """Optional — raise ValueError for any schema-level checks."""

    # ── 3. _convert_records ───────────────────────────────────────────────────

    def _convert_records(self, raw_records: list) -> Tuple[List[MigrationRow], float]:
        """
        Convert a list of native DB records into MigrationRow objects.

        - Create MigrationRow(self._schema.total_fields) for each record.
        - Fill slots using row.set_field(slot_index, value).
        - Skip records that have no vector (log a warning).
        - Catch per-record exceptions, log them, and continue.
        - Return (rows, elapsed_seconds).
        """
        t0   = time.time()
        rows = []

        for record in raw_records:
            try:
                row = MigrationRow(self._schema.total_fields)

                # SLOT 0: ID — always stringify
                row.set_field(0, str(record.id))

                # SLOT 1: DENSE VECTOR
                vec = record.vector   # List[float]
                if vec is None:
                    logger.warning(f"Record '{record.id}' has no vector — skipping.")
                    continue
                row.set_field(self._dense_slot, list(vec))

                # SLOT 2 (hybrid only): SPARSE VECTOR
                # sparse = record.sparse_vector  # Dict[int, float]
                # if self._sparse_slot >= 0 and sparse:
                #     row.set_field(self._sparse_slot, sparse)

                # LAST SLOT: metadata as plain dict
                row.set_field(self._payload_slot, dict(record.payload or {}))

                rows.append(row)

            except Exception as e:
                logger.error(f"Error converting record '{record.id}': {e}")
                continue

        return rows, time.time() - t0

    # ── 4. iterate_batches ────────────────────────────────────────────────────

    async def iterate_batches(
        self,
        batch_size:     int,
        initial_cursor: Any,
        schema:         RowSchema,
    ):
        """
        Async generator — the pipeline calls this to stream data.

        Cursor contract:
          - initial_cursor is None on first run, or the value you last yielded.
          - Yield (rows, next_cursor, timing_dict) each iteration.
          - next_cursor is whatever you need to resume: int offset, UUID token, etc.
          - Yield next_cursor=None on the last batch to signal completion.
          - Use a `finally` block to close any open iterators.

        Timing dict keys (all float seconds):
          "fetch"         — time waiting for the DB response
          "src_transform" — time spent in _convert_records()
        """
        loop   = asyncio.get_running_loop()
        cursor = initial_cursor   # None = start from beginning

        logger.info(f"PRODUCER: starting YourDB fetch | cursor={cursor}")

        try:
            while True:
                t_fetch = time.time()

                # Fetch one page from your DB (run sync SDK in executor)
                result, next_cursor = await loop.run_in_executor(
                    None,
                    lambda c=cursor: self._fetch_page(c, batch_size),
                )
                fetch_time = time.time() - t_fetch

                if not result:
                    logger.info("PRODUCER: no more data")
                    return

                rows, transform_time = self._convert_records(result)

                yield rows, next_cursor, {
                    "fetch":         fetch_time,
                    "src_transform": transform_time,
                }

                if next_cursor is None or len(result) < batch_size:
                    # Reached the end
                    return

                cursor = next_cursor

        finally:
            pass   # close any open iterator handles here

    def _fetch_page(self, cursor: Any, batch_size: int):
        """
        Synchronous helper — runs in executor.
        Returns (records, next_cursor).
        next_cursor = None when there are no more pages.
        """
        # result = self._client.scroll(
        #     collection=self.collection,
        #     offset=cursor,
        #     limit=batch_size,
        #     with_vectors=True,
        #     with_payload=True,
        # )
        # next_cursor = result.next_page_offset   # None when done
        # return result.points, next_cursor
        raise NotImplementedError

    # ── 5. close (optional) ───────────────────────────────────────────────────

    def close(self) -> None:
        """Called by the pipeline after migration completes or on error."""
        # self._client.close()

    # ── 6. from_args (required) ───────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "YourDBSource":
        """
        Pull only the args your connector needs.
        Called by migrate.py — never instantiate connectors directly in migrate.py.

        Available args (from migrate.py argument parser):
          args.source_url          str   — SOURCE_URL
          args.source_api_key      str   — SOURCE_API_KEY
          args.source_collection   str   — SOURCE_COLLECTION
          args.source_port         int   — SOURCE_PORT (or None)
          args.source_db           str   — SOURCE_DB (Milvus database name)
          args.filter_fields       str   — comma-separated filterable field names
          args.use_https           bool  — USE_HTTPS
          args.source_type         str   — "dense" | "hybrid"
          args.sparse_algo         str   — SPARSE_ALGO (e.g. "endee/bm25") or None
          args.sparse_text_field   str   — SPARSE_TEXT_FIELD or None
          args.precision           str   — PRECISION or None
          args.batch_size          int   — BATCH_SIZE
        """
        return cls(
            url        = args.source_url,
            collection = args.source_collection,
            api_key    = args.source_api_key,
            # add other params as needed
        )
```

---

### Step 2 — Register in `migrate.py`

**Add `"yourdb"` to the `--from` choices:**

```python
p.add_argument(
    "--from", dest="from_db",
    choices=["milvus", "qdrant", "chroma", "yourdb"],   # ← add here
    ...
)
```

**Import your class and add to SOURCE_REGISTRY:**

```python
from sources.yourdb_source import YourDBSource   # ← add import

SOURCE_REGISTRY = {
    ("milvus", "dense"):   MilvusDenseSource,
    ("milvus", "hybrid"):  MilvusHybridSource,
    ("qdrant", "dense"):   QdrantDenseSource,
    ("qdrant", "hybrid"):  QdrantHybridSource,
    ("chroma", "dense"):   ChromaDenseSource,
    ("yourdb", "dense"):   YourDBSource,           # ← add entry
    # ("yourdb", "hybrid"): YourDBHybridSource,    # if hybrid is supported
}
```

**If your source needs extra CLI args**, add them to `_build_parser()` in the `Source` group:

```python
src.add_argument("--your_param", default=os.getenv("YOUR_PARAM"))
```

---

## 2. Adding a New Target

### Files to create / edit

| Action | File |
|--------|------|
| Create | `targets/yourdb_target.py` |
| Edit   | `migrate.py` — add to `--to` choices + `TARGET_REGISTRY` |

---

### Step 1 — Create `targets/yourdb_target.py`

```python
# targets/yourdb_target.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base_target import BaseTarget
from core.schema import FieldRole, MigrationRow, RowSchema

logger = logging.getLogger(__name__)


class YourDBTarget(BaseTarget):
    """
    Target connector for YourDB.

    Reads RowSchema roles (ID, DENSE_VECTOR, SPARSE_VECTOR, METADATA)
    to extract slots — never hardcodes field names.
    """

    def __init__(
        self,
        url:             str,
        api_key:         str = "",
        index_name:      str = "",
        target_type:     str = "dense",   # "dense" | "hybrid"
        upsert_chunk_size: int = 100,
        # add index creation params as needed
        space_type:      str          = "cosine",
        M:               Optional[int] = None,
        ef_construct:    Optional[int] = None,
        precision:       Optional[str] = None,
        sparse_model:    Optional[str] = None,
    ):
        self.url              = url
        self.api_key          = api_key
        self.index_name       = index_name
        self.target_type      = target_type
        self.upsert_chunk_size = upsert_chunk_size
        self.space_type       = space_type
        self.M                = M
        self.ef_construct     = ef_construct
        self.precision        = precision
        self.sparse_model     = sparse_model

        self._client: Any = None

        # slot indices — resolved once in setup_index() from RowSchema
        self._pk_slot:      int = -1
        self._dense_slot:   int = -1
        self._sparse_slot:  int = -1
        self._payload_slots: List[int] = []

    # ── 1. connect ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Establish the DB connection. Called once before setup_index().
        """
        # self._client = YourDBClient(url=self.url, api_key=self.api_key)
        logger.info(f"Connected to YourDB target | index='{self.index_name}'")

    # ── 2. setup_index ────────────────────────────────────────────────────────

    def setup_index(self, schema: RowSchema) -> None:
        """
        Create or verify the target index using RowSchema.

        - Resolve slot positions from schema roles (store on self).
        - Detect index params from schema (dimension, space_type, etc.).
        - Create the index if it does not exist.
        - Use self.target_type to decide dense vs hybrid index, NOT schema.is_hybrid.

        RowSchema role helpers:
          schema.get_primary_key()    → FieldSchema with role=ID
          schema.get_dense_vector()   → FieldSchema with role=DENSE_VECTOR
          schema.get_sparse_vector()  → FieldSchema with role=SPARSE_VECTOR (or None)
          schema.get_metadata_fields()→ List[FieldSchema] with role=METADATA
          schema.index_of("name")     → int slot position
        """
        # Resolve slots from roles
        pk     = schema.get_primary_key()
        dense  = schema.get_dense_vector()
        sparse = schema.get_sparse_vector()

        self._pk_slot    = schema.index_of(pk.name)     if pk     else 0
        self._dense_slot = schema.index_of(dense.name)  if dense  else 1
        self._sparse_slot= schema.index_of(sparse.name) if sparse else -1
        self._payload_slots = [schema.index_of(f.name) for f in schema.get_metadata_fields()]

        dimension    = schema.dimension
        M            = self.M or 16
        ef_construct = self.ef_construct or 128

        logger.info(
            f"Setting up index '{self.index_name}' | "
            f"type={self.target_type} | dim={dimension}"
        )

        # Create the index based on self.target_type
        if self.target_type == "hybrid":
            # self._client.create_hybrid_index(
            #     name=self.index_name,
            #     dimension=dimension,
            #     sparse_model=self.sparse_model or "default",
            #     ...
            # )
            pass
        else:
            # self._client.create_dense_index(
            #     name=self.index_name,
            #     dimension=dimension,
            #     ...
            # )
            pass

    # ── 3. upsert_batch ───────────────────────────────────────────────────────

    async def upsert_batch(
        self, records: List[MigrationRow], schema: RowSchema
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Write one batch to the target.

        Contract:
          - Never raise — catch all exceptions internally and return False.
          - Return (True, timing_dict)  on success.
          - Return (False, timing_dict) on unrecoverable failure (pipeline stops).
          - chunk records into self.upsert_chunk_size to respect API limits.
          - Timing dict key: "upsert" (float seconds).

        Read slots using the indices resolved in setup_index():
          record.get_field(self._pk_slot)      → id
          record.get_field(self._dense_slot)   → List[float]
          record.get_field(self._sparse_slot)  → Dict{"indices": [...], "values": [...]}
          record.get_field(self._payload_slots[i]) → dict (metadata)
        """
        t0 = time.time()
        loop = asyncio.get_running_loop()

        try:
            native_records = [self._to_native(r) for r in records]

            # Chunk and upsert
            for i in range(0, len(native_records), self.upsert_chunk_size):
                chunk = native_records[i : i + self.upsert_chunk_size]
                await loop.run_in_executor(
                    None,
                    lambda c=chunk: self._upsert_chunk(c),
                )

            return True, {"upsert": time.time() - t0}

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return False, {"upsert": time.time() - t0}

    def _to_native(self, record: MigrationRow) -> dict:
        """Convert one MigrationRow to your DB's native record format."""
        doc_id  = record.get_field(self._pk_slot)
        vector  = record.get_field(self._dense_slot)

        native = {
            "id":     str(doc_id),
            "vector": vector,
        }

        # Sparse vector (hybrid only)
        if self._sparse_slot >= 0:
            sparse = record.get_field(self._sparse_slot)
            if sparse:
                native["sparse_indices"] = sparse["indices"]
                native["sparse_values"]  = sparse["values"]

        # Metadata — merge all payload slots into one dict
        payload: dict = {}
        for slot in self._payload_slots:
            payload.update(record.get_field(slot) or {})
        native["metadata"] = payload

        return native

    def _upsert_chunk(self, chunk: list) -> None:
        """Synchronous upsert — runs in executor."""
        # self._client.upsert(index=self.index_name, records=chunk)
        raise NotImplementedError

    # ── 4. close (optional) ───────────────────────────────────────────────────

    def close(self) -> None:
        """Called after migration completes or on error."""
        # self._client.close()

    # ── 5. from_args (required) ───────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "YourDBTarget":
        """
        Pull only the args your connector needs.

        Available args (from migrate.py argument parser):
          args.target_url        str   — TARGET_URL
          args.target_api_key    str   — TARGET_API_KEY
          args.target_collection str   — TARGET_COLLECTION
          args.target_type       str   — "dense" | "hybrid"
          args.space_type        str   — SPACE_TYPE (default "cosine")
          args.M                 int   — HNSW M or None
          args.ef_construct      int   — HNSW ef_construct or None
          args.precision         str   — PRECISION or None
          args.sparse_model      str   — SPARSE_MODEL or None
          args.upsert_size       int   — UPSERT_SIZE
          args.filter_fields     str   — comma-separated filterable field names
        """
        return cls(
            url              = args.target_url,
            api_key          = args.target_api_key,
            index_name       = args.target_collection,
            target_type      = getattr(args, "target_type", "dense"),
            upsert_chunk_size= args.upsert_size,
            space_type       = args.space_type,
            M                = args.M,
            ef_construct     = args.ef_construct,
            precision        = args.precision,
            sparse_model     = getattr(args, "sparse_model", None),
        )
```

---

### Step 2 — Register in `migrate.py`

**Add `"yourdb"` to the `--to` choices:**

```python
p.add_argument(
    "--to", dest="to_db",
    choices=["endee", "yourdb"],   # ← add here
    ...
)
```

**Import your class and add to TARGET_REGISTRY:**

```python
from targets.yourdb_target import YourDBTarget   # ← add import

TARGET_REGISTRY = {
    ("endee",  "dense"):  EndeeTarget,
    ("endee",  "hybrid"): EndeeTarget,
    ("yourdb", "dense"):  YourDBTarget,            # ← add entry
    ("yourdb", "hybrid"): YourDBTarget,            # if hybrid is supported
}
```

---

## 3. Adding a New Sparse Encoder

Use this when you want to support a new algorithm for generating sparse vectors
(e.g. SPLADE, TF-IDF). No source or target files need to change.

### Files to create / edit

| Action | File |
|--------|------|
| Edit   | `sparse_encoders/concrete_sparse_encoders.py` — add your class |
| Edit   | `sparse_encoders/factory_sparse_encoder.py` — register it |
| Edit   | `migrate.py` — add to `--sparse_algo` choices |

**In `concrete_sparse_encoders.py`:**

```python
class MySparseEncoder(BaseSparseEncoder):

    def encode(self, text: str) -> dict:
        """
        Returns {"indices": List[int], "values": List[float]}.
        """
        ...

    def encode_batch(self, texts: List[str]) -> List[dict]:
        """
        Batch version — preferred when available (avoids Python loop overhead).
        Each element matches the format returned by encode().
        """
        return [self.encode(t) for t in texts]
```

**In `factory_sparse_encoder.py`:**

```python
from sparse_encoders.concrete_sparse_encoders import MySparseEncoder
SparseEncoderFactory.register("my_algo", MySparseEncoder)
```

**In `migrate.py`:**

```python
sparse.add_argument(
    "--sparse_algo",
    choices=["endee/bm25", "my_algo"],   # ← add here
    ...
)
```

---

## 4. Key Types Reference

### `RowSchema` — built by source, read by target

```
RowSchema
  .fields              List[FieldSchema]  — ordered slot definitions
  .dimension           int                — dense vector size
  .space_type          str                — "cosine" | "l2" | "ip"
  .is_hybrid           bool               — True if a SPARSE_VECTOR slot exists
  .canonical_precision str                — "float32" | "float16" | "int8" | "int16"

  .get_primary_key()           → FieldSchema | None
  .get_dense_vector()          → FieldSchema | None
  .get_sparse_vector()         → FieldSchema | None
  .get_metadata_fields()       → List[FieldSchema]
  .index_of(name)              → int  (-1 if not found)
  .require_index_of(name)      → int  (raises ValueError if not found)
  .total_fields                → int
```

### `FieldSchema` — one slot descriptor

```
FieldSchema
  .name        str        — field name (e.g. "id", "embedding")
  .field_type  FieldType  — STRING | INT | FLOAT | BOOL | DENSE_VECTOR | SPARSE_VECTOR | JSON
  .role        FieldRole  — ID | DENSE_VECTOR | SPARSE_VECTOR | METADATA
  .dimension   int | None — set only for DENSE_VECTOR fields
```

### `MigrationRow` — one data row

```
MigrationRow(arity: int)
  .set_field(pos: int, value: Any)   — source fills slots positionally
  .get_field(pos: int) → Any         — target reads slots by index
  .arity → int
```

### Sparse vector wire format

Sparse vectors flow as plain dicts between source and target:

```python
{
    "indices": [101, 4892, 12043],   # List[int]  — token ids
    "values":  [0.42, 0.17, 0.91],   # List[float] — weights
}
```

### `iterate_batches` yield format

```python
yield (
    rows,          # List[MigrationRow]
    next_cursor,   # Any — opaque; None signals end of data
    {
        "fetch":          1.23,   # seconds waiting for DB response
        "src_transform":  0.04,   # seconds in _convert_records()
    }
)
```

### `upsert_batch` return format

```python
return (
    True,          # bool — False stops the pipeline immediately
    {"upsert": 0.87},  # timing dict, seconds
)
```
