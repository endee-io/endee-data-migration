"""
sources/milvus_source.py
──────────────────────────────────────────────────────────────────────────────
Milvus source connectors (dense-only and hybrid).

MilvusDenseSource  — validates dense-only collections, rejects hybrid ones.
MilvusHybridSource — validates hybrid collections (dense+sparse), rejects dense-only.

Both share a common MilvusBaseSource that handles:
  - Connection (with protocol auto-fix)
  - Collection loading (with timeout)
  - Field schema detection → builds RowSchema
  - Vector byte decoding (FLOAT16, FLOAT32, BINARY)
  - QueryIterator-based async batch iteration (no 16384 offset cap)
  - Checkpoint-skip on resume (count-based)

Cursor format
─────────────
Milvus uses a COUNT cursor: the number of records processed so far.
On resume, the iterator skips that many records from the beginning.

RowSchema slot layout
──────────────────────
  SLOT 0        : ID  (STRING)
  SLOT 1        : DENSE_VECTOR
  SLOT 2        : SPARSE_VECTOR  (hybrid only)
  LAST SLOT     : JSON payload   (all remaining fields bundled)

Source never splits payload into filter/meta — that is the target's concern.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any, List, Optional, Tuple

from pymilvus import DataType, MilvusClient
from pymilvus import MilvusException

from core.base_source import BaseSource
from core.schema import FieldRole, FieldSchema, FieldType, MigrationRow, RowSchema
from core.type_registry import (
    SPACE_COSINE, SPACE_L2, SPACE_IP,
    PRECISION_FLOAT32, PRECISION_RAW_BINARY,
    resolve_space,
)

# ── Milvus → wire precision ───────────────────────────────────────────────────
# FLOAT_VECTOR: client returns List[float] natively — wire precision = float32.
# All other vector types: client returns raw bytes — wire precision = raw_binary.
# The target (Endee) checks this value and exits if it is not float32.
MILVUS_TO_WIRE_PRECISION: dict = {
    DataType.FLOAT_VECTOR:    PRECISION_FLOAT32,
    "float_vector":           PRECISION_FLOAT32,
    DataType.FLOAT16_VECTOR:  PRECISION_RAW_BINARY,
    "float16_vector":         PRECISION_RAW_BINARY,
    DataType.BFLOAT16_VECTOR: PRECISION_RAW_BINARY,
    "bfloat16_vector":        PRECISION_RAW_BINARY,
    DataType.BINARY_VECTOR:   PRECISION_RAW_BINARY,
    "binary_vector":          PRECISION_RAW_BINARY,
    DataType.INT8_VECTOR:     PRECISION_RAW_BINARY,
    "int8_vector":            PRECISION_RAW_BINARY,
}

# ── Milvus metric → canonical space ──────────────────────────────────────────
MILVUS_TO_CANONICAL_SPACE: dict[str, str] = {
    "cosine": SPACE_COSINE,
    "l2":     SPACE_L2,
    "ip":     SPACE_IP,
}



logger = logging.getLogger(__name__)

# # ── Milvus vector dtype → human-readable name (logging only) ─────────────────
# MILVUS_DTYPE_TO_PREC_NAME = {
#     DataType.FLOAT_VECTOR:    "float32",
#     DataType.FLOAT16_VECTOR:  "float16",
#     DataType.BINARY_VECTOR:   "binary",
# }
# MILVUS_STR_TO_PREC_NAME = {
#     "FLOAT_VECTOR":    "float32",
#     "FLOAT16_VECTOR":  "float16",
#     "BINARY_VECTOR":   "binary",
# }

# # ── Milvus metric type → normalised space_type string ────────────────────────
# MILVUS_METRIC_TO_SPACE = {
#     "COSINE": "cosine",
#     "L2":     "l2",
#     "IP":     "ip",
# }


# ══════════════════════════════════════════════════════════════════════════════
# Shared Milvus base
# ══════════════════════════════════════════════════════════════════════════════
class MilvusBaseSource(BaseSource):
    """
    Internal base class — all Milvus plumbing lives here.
      - detect_schema() builds and returns RowSchema
      - _convert_records() fills MigrationRow slots positionally
      - filter_fields is the target's concern, not the source's

    Subclasses only override _validate_schema().
    """

    DEFAULT_PORT = 19530

    def __init__(
        self,
        url:        str,
        token:      str,
        collection: str,
        db:         str = "default",
        port:       int = DEFAULT_PORT,
    ):
        self.url        = url
        self.token      = token
        self.db         = db
        self.collection = collection
        self.port       = port

        self.milvus_client: Optional[MilvusClient] = None
        self._schema:       Optional[RowSchema]    = None

        # slot positions — resolved once in detect_schema()
        self._dense_slot:        int = -1
        self._sparse_slot:       int = -1
        self._payload_slot:      int = -1

        # field names — needed for vector extraction per record
        self._dense_field_name:  Optional[str] = None
        self._sparse_field_name: Optional[str] = None
        self._meta_field_names:  List[str]   = []     # all non-vector, non-pk fields

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to Milvus...")
        uri = self.url
        if not uri.startswith(("http://", "https://", "tcp://", "unix://")):
            if uri.startswith("localhost") or uri.replace(".", "").replace(":", "").isdigit():
                uri = f"http://{uri}:{self.port}"
                logger.info(f"  Auto-added protocol: {uri}")
        try:
            self.milvus_client = MilvusClient(uri=uri, token=self.token, db_name=self.db)
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            sys.exit(1)

    # ── Collection loading ────────────────────────────────────────────────────

    def _load_collection(self):
        logger.info(f"Loading collection '{self.collection}' into memory...")
        try:
            state = str(self.milvus_client.get_load_state(
                collection_name=self.collection
            ).get("state", ""))
            logger.info(f"  Current load state: {state}")
            if state == "Loaded":
                logger.info("  Collection already loaded")
                return
            self.milvus_client.load_collection(collection_name=self.collection)
            for elapsed in range(5, 305, 5):
                time.sleep(5)
                state = str(self.milvus_client.get_load_state(
                    collection_name=self.collection
                ).get("state", ""))
                logger.info(f"  Load state after {elapsed}s: {state}")
                if state == "Loaded":
                    logger.info("  Collection loaded")
                    return
            logger.error(f"Collection did not load within 300s. Last state: {state}")
            sys.exit(1)
        except MilvusException as e:
            if "index not found" in str(e).lower():
                logger.error(
                    "Collection has no vector index and cannot be loaded. "
                    "Build an index first, then re-run the migration."
                )
            else:
                logger.error(f"Failed to load collection: {e}")
            sys.exit(1)

    # ── Schema detection ──────────────────────────────────────────────────────

    def detect_schema(self) -> RowSchema:
        """
        Reads Milvus collection fields.
        Builds RowSchema with field roles — source knows nothing about Endee.

        Slot layout:
          0        -> ID
          1        -> DENSE_VECTOR
          2        -> SPARSE_VECTOR (hybrid only)
          last     -> JSON payload (all meta fields bundled)
        """
        self._load_collection()

        desc = self.milvus_client.describe_collection(self.collection)
        logger.info(f"\n{'='*80}\nDetecting fields in: {self.collection}\n{'='*80}")

        schema_fields: List[FieldSchema] = []
        sparse_fields_raw = []
        meta_field_names  = []
        dimension         = None
        space_type        = "cosine"   # default if index read fails

        for field in desc.get("fields", []):
            name   = field.get("name")
            ftype  = field.get("type")
            is_pk  = field.get("is_primary", False)

            # ── SLOT 0: ID ─────────────────────────────────────────────────
            if is_pk:
                schema_fields.append(FieldSchema(
                    name       = name,
                    field_type = FieldType.STRING,
                    role       = FieldRole.ID,
                ))
                logger.info(f"  [PK]     {name} [{ftype}]")

            # ── SLOT 1: DENSE VECTOR ────────────────────────────────────────
            # All known vector types are accepted here; the target (Endee) will
            # reject non-float32 wire precision with a clean error at setup time.
            elif ftype in (
                DataType.FLOAT_VECTOR,    DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR, DataType.BINARY_VECTOR,
                DataType.INT8_VECTOR,
                "FLOAT_VECTOR",    "FLOAT16_VECTOR",
                "BFLOAT16_VECTOR", "BINARY_VECTOR",
                "INT8_VECTOR",
            ):

                if self._dense_field_name is not None:
                    # Already have a dense field — skip extras
                    continue

                params = field.get("params", {})
                dimension = params.get("dim") or field.get("dim")

                # Read metric type from index
                index_info = self.milvus_client.describe_index(self.collection, name) or {}
                raw_metric = index_info.get("metric_type", "COSINE").upper()
                space_type = resolve_space(MILVUS_TO_CANONICAL_SPACE, raw_metric)

                self._dense_field_name = name
                self._dense_slot       = 1   # always slot 1

                schema_fields.append(FieldSchema(
                    name       = name,
                    field_type = FieldType.DENSE_VECTOR,
                    role       = FieldRole.DENSE_VECTOR,
                    dimension  = dimension,
                ))

                wire_precision = MILVUS_TO_WIRE_PRECISION.get(ftype)
                if wire_precision is None:
                    raise ValueError(
                        f"Unknown Milvus vector field type '{ftype}' in field '{name}'. "
                        f"Add it to MILVUS_TO_WIRE_PRECISION in milvus_source.py first."
                    )
                logger.info(f"  [DENSE]  {name} [{ftype}], dim={dimension}, "
                            f"space={space_type}, wire_precision={wire_precision}")

            # ── SLOT 2 (hybrid): SPARSE VECTOR ─────────────────────────────
            elif ftype in (DataType.SPARSE_FLOAT_VECTOR, "SPARSE_FLOAT_VECTOR"):
                sparse_fields_raw.append(name)
                if self._sparse_field_name is None:
                    self._sparse_field_name = name
                    # slot index assigned after dense field is confirmed
                logger.info(f"  [SPARSE] {name} [{ftype}]")

            # ── metadata fields ─────────────────────────────────────────────
            else:
                meta_field_names.append(name)
                logger.info(f"  [META]   {name} [{ftype}]")

        if not self._dense_field_name:
            raise ValueError("No dense vector field found in collection")

        # Assign sparse slot AFTER dense (slot 1 is always dense)
        if self._sparse_field_name:
            schema_fields.append(FieldSchema(
                name       = self._sparse_field_name,
                field_type = FieldType.SPARSE_VECTOR,
                role       = FieldRole.SPARSE_VECTOR,
            ))
            self._sparse_slot = len(schema_fields) - 1

        # Bundle all metadata fields into a single JSON payload slot
        # Target decides how to split into filter/meta — source doesn't know
        self._meta_field_names = meta_field_names
        schema_fields.append(FieldSchema(
            name       = "payload",
            field_type = FieldType.JSON,
            role       = FieldRole.METADATA,
        ))
        self._payload_slot = len(schema_fields) - 1

        logger.info(f"\nPK: slot=0  Dense: slot={self._dense_slot}  "
                    f"Sparse: slot={self._sparse_slot}  Payload: slot={self._payload_slot}")
        logger.info(f"  meta fields bundled into payload: {meta_field_names}")

        self._schema = RowSchema(
            fields              = schema_fields,
            dimension           = dimension,
            space_type          = space_type,
            is_hybrid           = self._sparse_field_name is not None,
            canonical_precision = wire_precision,
        )

        # Subclass validates sparse presence/absence
        self._validate_schema(sparse_fields_raw)
        return self._schema

    def _validate_schema(self, sparse_fields: list):
        """Override in subclass."""

    # ── Record conversion ─────────────────────────────────────────────────────

    def _convert_records(
        self, milvus_records: list
    ) -> Tuple[List[MigrationRow], float]:
        """
        Converts Milvus records to MigrationRow.
        Fills slots POSITIONALLY — matches order in RowSchema.fields.
        All metadata fields are bundled into a single JSON payload dict.
        Returns (rows, transform_time_seconds).
        """
        t0     = time.time()
        schema = self._schema
        rows   = []

        for rec in milvus_records:
            try:
                row = MigrationRow(schema.total_fields)

                # SLOT 0: ID
                row.set_field(0, str(rec.get(self._schema.fields[0].name, "")))

                # SLOT 1: DENSE VECTOR — pass through as-is (no decoding)
                row.set_field(self._dense_slot, rec.get(self._dense_field_name, []))

                # SLOT 2 (hybrid): SPARSE VECTOR
                if self._sparse_slot >= 0:
                    sparse_data = rec.get(self._sparse_field_name, {})
                    if sparse_data and isinstance(sparse_data, dict):
                        sorted_items = sorted(sparse_data.items())
                        row.set_field(self._sparse_slot, {
                            "indices": [int(i)   for i, _ in sorted_items],
                            "values":  [float(v) for _, v in sorted_items],
                        })

                # LAST SLOT: full payload dict — target decides filter/meta split
                payload = {k: rec.get(k) for k in self._meta_field_names if k in rec}
                row.set_field(self._payload_slot, payload)

                rows.append(row)
            except Exception as e:
                import traceback
                logger.error(f"Error converting record: {e}")
                logger.error(traceback.format_exc())
                continue

        return rows, time.time() - t0

    # ── Scroll with retry ─────────────────────────────────────────────────────

    async def _next_batch(self, iterator, loop) -> list:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return await loop.run_in_executor(None, iterator.next)
            except Exception as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Iterator.next failed (attempt {attempt+1}/{max_retries}): "
                        f"{e}. Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Iterator.next failed after {max_retries} attempts: {e}")
                    raise

    # ── iterate_batches ───────────────────────────────────────────────────────

    async def iterate_batches(
        self, batch_size: int, initial_cursor: Any, schema: RowSchema
    ):
        """
        Async generator using Milvus QueryIterator (no 16384 offset cap).

        Cursor: int — total records processed before this run.
        On resume: skips `initial_cursor` records from the iterator head.
        Yields: (List[MigrationRow], next_cursor, {"fetch": float, "src_transform": float})
        """
        loop             = asyncio.get_running_loop()
        records_to_skip  = initial_cursor or 0
        fetched_this_run = 0

        logger.info(f"PRODUCER: creating Milvus QueryIterator, skip={records_to_skip}")
        try:
            iterator = await loop.run_in_executor(
                None,
                lambda: self.milvus_client.query_iterator(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["*"],
                    batch_size=batch_size,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to create Milvus QueryIterator: {e}")
            raise

        try:
            # Skip already-processed records (checkpoint resume)
            if records_to_skip > 0:
                skipped = 0
                logger.info(f"PRODUCER: skipping {records_to_skip} already-processed records")
                while skipped < records_to_skip:
                    batch = await self._next_batch(iterator, loop)
                    if not batch:
                        logger.warning("PRODUCER: ran out of records while skipping — already done?")
                        return
                    skipped += len(batch)
                logger.info(f"PRODUCER: skipped {skipped} records, resuming")

            while True:
                t_fetch = time.time()
                batch   = await self._next_batch(iterator, loop)
                fetch_time = time.time() - t_fetch

                if not batch:
                    logger.info("PRODUCER: no more data from Milvus")
                    return

                rows, transform_time = self._convert_records(batch)
                fetched_this_run    += len(rows)
                next_cursor          = records_to_skip + fetched_this_run

                yield rows, next_cursor, {"fetch": fetch_time, "src_transform": transform_time}

        finally:
            await loop.run_in_executor(None, iterator.close)
            logger.info("PRODUCER: Milvus iterator closed")

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Public connectors
# ══════════════════════════════════════════════════════════════════════════════

class MilvusDenseSource(MilvusBaseSource):
    """
    Milvus source for DENSE-ONLY collections.
    Raises ValueError if sparse vector fields are detected.
    Optionally generates sparse vectors from a text payload field when sparse_algo is set.
    """

    def __init__(self, *args, sparse_algo=None, sparse_text_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_algo       = sparse_algo
        self.sparse_text_field = sparse_text_field
        self._encoder          = None

    def connect(self):
        super().connect()
        if self.sparse_algo:
            from sparse_encoders.factory_sparse_encoder import SparseEncoderFactory
            self._encoder = SparseEncoderFactory.create(self.sparse_algo)
            logger.info(f"Sparse encoder loaded: {type(self._encoder).__name__} (algo='{self.sparse_algo}')")

    def _validate_schema(self, sparse_fields: list):
        if sparse_fields:
            raise ValueError(
                f"Collection '{self.collection}' is HYBRID "
                f"(sparse fields: {sparse_fields}).\n"
                f"Use MilvusHybridSource instead."
            )
        if self.sparse_algo:
            logger.info(f"Schema validated: dense collection — sparse will be generated via '{self.sparse_algo}' from field '{self.sparse_text_field}'")
        else:
            logger.info("Schema validated: dense-only collection")

    def detect_schema(self) -> RowSchema:
        schema = super().detect_schema()

        if not self.sparse_algo:
            return schema

        # Peek at one record to confirm the text field exists in actual data
        try:
            sample = self.milvus_client.query(
                collection_name=self.collection,
                filter="",
                output_fields=[self.sparse_text_field],
                limit=1,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to read field '{self.sparse_text_field}' from collection '{self.collection}': {e}\n"
                "Check that SPARSE_TEXT_FIELD matches a real field name in the collection."
            )

        if not sample or not sample[0].get(self.sparse_text_field):
            raise ValueError(
                f"Field '{self.sparse_text_field}' is empty or missing in collection '{self.collection}'.\n"
                "Cannot generate sparse vectors without text. "
                "Set SPARSE_TEXT_FIELD to a field that contains text, or remove SPARSE_ALGO to run a dense migration."
            )

        # Add sparse slot to schema so target creates a hybrid index
        from core.schema import FieldSchema, FieldType, FieldRole
        schema.fields.append(FieldSchema(
            name       = "sparse_vector",
            field_type = FieldType.SPARSE_VECTOR,
            role       = FieldRole.SPARSE_VECTOR,
        ))
        self._sparse_slot = len(schema.fields) - 1
        # Move payload slot one position forward
        self._payload_slot = self._sparse_slot + 1
        schema.fields.append(FieldSchema(
            name       = "payload",
            field_type = FieldType.JSON,
            role       = FieldRole.METADATA,
        ))
        schema.is_hybrid = True
        logger.info(f"  [SPARSE] will be generated via '{self.sparse_algo}' from field '{self.sparse_text_field}' (slot={self._sparse_slot})")
        return schema

    def _convert_records(self, milvus_records: list):
        if not self.sparse_algo:
            return super()._convert_records(milvus_records)

        t0   = time.time()
        rows = []

        texts = [
            (rec.get(self.sparse_text_field) or "") for rec in milvus_records
        ]
        if all(t == "" for t in texts):
            raise RuntimeError(
                f"All records in this batch have empty '{self.sparse_text_field}' — cannot generate sparse vectors."
            )

        if hasattr(self._encoder, "encode_batch"):
            sparse_embs = self._encoder.encode_batch(texts)
        else:
            sparse_embs = [self._encoder.encode(t) for t in texts]

        for i, rec in enumerate(milvus_records):
            try:
                row = MigrationRow(self._schema.total_fields)
                row.set_field(0, str(rec.get(self._schema.fields[0].name, "")))
                row.set_field(self._dense_slot, rec.get(self._dense_field_name, []))
                sp = sparse_embs[i]
                if sp and sp.get("indices"):
                    row.set_field(self._sparse_slot, sp)
                payload = {k: rec.get(k) for k in self._meta_field_names if k in rec}
                row.set_field(self._payload_slot, payload)
                rows.append(row)
            except Exception as e:
                import traceback
                logger.error(f"Error converting record: {e}")
                logger.error(traceback.format_exc())
                continue

        return rows, time.time() - t0

    @classmethod
    def from_args(cls, args):
        return cls(
            url              = args.source_url,
            token            = args.source_api_key,
            collection       = args.source_collection,
            db               = args.source_db,
            port             = args.source_port or 19530,
            sparse_algo      = getattr(args, "sparse_algo", None),
            sparse_text_field= getattr(args, "sparse_text_field", None),
        )


class MilvusHybridSource(MilvusBaseSource):
    """
    Milvus source for HYBRID collections (dense + sparse).
    Raises ValueError if no sparse vector fields are detected.
    """

    def _validate_schema(self, sparse_fields: list):
        if not sparse_fields:
            raise ValueError(
                f"Collection '{self.collection}' is DENSE-ONLY "
                f"(no sparse fields detected).\n"
                f"Use MilvusDenseSource instead."
            )
        logger.info(
            f"Schema validated: hybrid (dense={self._dense_field_name}, "
            f"sparse={self._sparse_field_name})"
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            url        = args.source_url,
            token      = args.source_api_key,
            collection = args.source_collection,
            db         = args.source_db,
            port       = args.source_port or 19530,
        )