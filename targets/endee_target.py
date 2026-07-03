"""
    targets/endee_target.py
    ============================
    Endee target connector — handles both dense-only and hybrid collections.

    Uses Endee API v2: Database → Collection → Index hierarchy.

    A single EndeeTarget works for both:
    - Dense  -> target_type=dense  -> creates a collection with one vector field
    - Hybrid -> target_type=hybrid -> creates a collection with vector + sparse fields

    MigrationRow -> Endee native format conversion (v2):
    Dense record:
        {"id": str, "fields": {"embedding": [...]}, "filter": {...}, "meta": {...}}
    Hybrid record:
        {"id": str, "fields": {"embedding": [...], "keywords": {"indices": [...], "values": [...]}},
        "filter": {...}, "meta": {...}}

    upsert_batch() splits the batch into upsert_chunk_size chunks, sends them
    all in parallel via asyncio.gather(), then retries any failures with
    exponential backoff (max 3 attempts).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from endee import Endee
from endee.exceptions import NotFoundException, ConflictException

from core.base_target import BaseTarget
from core.schema import FieldRole, FieldType, MigrationRow, RowSchema
import time
from constants import DEFAULT_SPARSE_MODEL, ENDEE_V1_API
from endee import Precision
from core.type_registry import (
    SPACE_COSINE, SPACE_L2, SPACE_IP,
    PRECISION_FLOAT32,
    PRECISION_FLOAT16, PRECISION_INT16, PRECISION_INT8, PRECISION_BINARY,
    resolve_space, resolve_precision,
)

# ── Canonical space → Endee space ────────────────────────────────────────────
CANONICAL_TO_ENDEE_SPACE: dict[str, str] = {
    SPACE_COSINE: "cosine",
    SPACE_L2:     "l2",
    SPACE_IP:     "ip",
}

# ── Canonical precision → Endee Precision enum ───────────────────────────────
CANONICAL_TO_ENDEE_PRECISION: dict[str, Precision] = {
    PRECISION_FLOAT32: Precision.FLOAT32,
    PRECISION_FLOAT16: Precision.FLOAT16,
    PRECISION_INT16:   Precision.INT16,
    PRECISION_INT8:    Precision.INT8,
    PRECISION_BINARY:  Precision.BINARY2,
}


logger = logging.getLogger(__name__)


class EndeeTarget(BaseTarget):
    """
    Endee sink connector.

    Parameters
    ----------
    endee_url        : Endee base URL (e.g. https://your-cluster.endee.io)
    endee_api_key    : Endee API key
    index_name       : Target index name
    upsert_chunk_size: Records per individual upsert call (internal chunking)
    sparse_model     : Sparse model identifier passed on hybrid index creation
    """

    def __init__(
        self,
        endee_url:              str,
        endee_api_key:          str,
        index_name:             str,
        upsert_chunk_size:      int = 100,
        sparse_model:           str = DEFAULT_SPARSE_MODEL,
        filter_fields:          Optional[str] = "", # FROM USER
        space_type:             str = "cosine",
        M:                      int = 16,
        ef_construct:           int = 128,
        precision:              Optional[str] = None,
        target_type:            str = "dense",

    ):
        self.endee_url              = endee_url
        self.endee_api_key          = endee_api_key
        self.index_name             = index_name
        self.upsert_chunk_size      = upsert_chunk_size
        self.sparse_model           = sparse_model
        self.space_type             = resolve_space(CANONICAL_TO_ENDEE_SPACE, space_type)
        self.M                      = M
        self.ef_construct           = ef_construct
        self.precision              = precision # CANONICAL PRECISION
        self.target_type            = target_type

        self._client:      Any      = None
        self._collection:  Any      = None

        self.filter_fields: Set[str] = (
            set(f.strip() for f in filter_fields.split(",") if f.strip())
            if filter_fields else set()
        )
        self._pk_slot:     int      = -1
        self._dense_slot:  int      = -1
        self._sparse_slot: int      = -1
        self._payload_slots: List[int] = []
        self._payload_types: List[FieldType] = []
        self._payload_names: List[str] = []

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to Endee...")
        self._client = Endee(token=self.endee_api_key)
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, ENDEE_V1_API)
            self._client.set_base_url(url)
            logger.info(f"  Base URL: {url}")

        # ── DEBUG: smoke-test before proceeding ──
        try:
            self._client.list_collections()
            logger.info(f"Endee reachable")
        except Exception as e:
            logger.error(f"  Endee unreachable after connect: {e}")
            raise
        logger.info("Connected to Endee")

    # ── setup_collection ─────────────────────────────────────────────────────

    def setup_collection(self, schema: RowSchema):
        """
            Creates or gets the Endee collection (v2 API).
            ALL params come from RowSchema — no hardcoded values, no separate IndexConfig.
            Role detection done here once — slots cached for fast row conversion.
        """
        # FILTER FIELDS VALIDATION
        if self.filter_fields:
            metadata_names = {f.name for f in schema.get_metadata_fields()}
            invalid = self.filter_fields - metadata_names
            if invalid:
                logger.error(
                    f"Invalid filter_fields: {invalid}\n"
                    f"Available metadata fields: {metadata_names}"
                )
                sys.exit(1)
            logger.info(f"  filter_fields validated: {self.filter_fields}")

        # RESOLVE SLOT POSITIONS
        pk_field         = schema.get_primary_key()
        dense_field      = schema.get_dense_vector()
        sparse_field     = schema.get_sparse_vector()
        meta_fields      = schema.get_metadata_fields()
        source_precision = schema.canonical_precision

        # PRECISION COMPATIBILITY CHECK
        # Endee only accepts float32 vectors on upsert (it quantises internally).
        # The Milvus source sets wire precision to float32 only for FLOAT_VECTOR;
        # all other Milvus vector types (float16, bfloat16, int8, binary) are
        # returned as raw bytes and mapped to raw_binary — those cannot be sent to Endee.
        if source_precision != PRECISION_FLOAT32:
            logger.error(
                f"Migration aborted: source collection uses a quantized vector type "
                f"(wire precision='{source_precision}'). "
                f"Quantized vectors (float16, bfloat16, int8, binary) cannot be migrated "
                f"because the original float32 values are lost after quantization. "
                f"Re-embed your data as FLOAT_VECTOR in Milvus and retry."
            )
            sys.exit(1)

        self.endee_precision = resolve_precision(CANONICAL_TO_ENDEE_PRECISION, self.precision)

        # REQUIRED FIELDS
        if pk_field is None:
            raise ValueError("RowSchema has no PRIMARY_KEY field — cannot create Endee collection")
        if dense_field is None:
            raise ValueError("RowSchema has no DENSE_VECTOR field — cannot create Endee collection")

        self.dimension   = dense_field.dimension
        self._pk_slot    = schema.index_of(pk_field.name)
        self._dense_slot = schema.index_of(dense_field.name)

        # optional: sparse
        if sparse_field:
            self._sparse_slot = schema.index_of(sparse_field.name)

        # metadata slots — precomputed for fast lookup in _to_endee
        for mf in meta_fields:
            self._payload_slots.append(schema.index_of(mf.name))
            self._payload_types.append(mf.field_type)
            self._payload_names.append(mf.name)

        logger.info(f"  Slot map: id={self._pk_slot}, dense={self._dense_slot}, "
                    f"sparse={self._sparse_slot}, meta={self._payload_slots}")

        # COLLECTION ALREADY EXISTS — reuse it
        try:
            self._collection = self._client.get_collection(self.index_name)
            logger.info(f"Collection already exists: {self.index_name}")
            return
        except NotFoundException:
            pass

        # BUILD FIELD DEFINITIONS FOR v2 create_collection
        fields = [
            {
                "name": "embedding",
                "type": "vector",
                "params": {
                    "dimension":  self.dimension,
                    "space_type": self.space_type,
                    "precision":  self.endee_precision,
                    "M":          self.M,
                    "ef_con":     self.ef_construct,
                },
            }
        ]

        if self.target_type == "hybrid":
            sparse_model = self.sparse_model or DEFAULT_SPARSE_MODEL
            fields.append({
                "name": "keywords",
                "type": "sparse",
                "params": {
                    "sparse_model": sparse_model,
                },
            })
            logger.info(f"Creating HYBRID collection '{self.index_name}' (sparse_model={sparse_model})")
        else:
            logger.info(f"Creating DENSE collection '{self.index_name}'")

        self._client.create_collection(name=self.index_name, fields=fields)
        self._collection = self._client.get_collection(self.index_name)
        logger.info(f"Created collection: {self.index_name}")

    # ── Record conversion ─────────────────────────────────────────────────────


    def _to_endee(self, record: MigrationRow, schema: RowSchema) -> dict:
        """Convert canonical MigrationRow to Endee v2 native dict format.
            Uses pre-resolved slot indexes — NO schema lookups per row (fast).
            Uses FieldRole — NO hardcoded attribute names like row.dense_vector.
            filter_fields split happens HERE — source never knew about it.

            v2 format:
            {
                "id": str,
                "fields": {
                    "embedding": [...],               # dense
                    "keywords":  {"indices": [...], "values": [...]}  # sparse (hybrid only)
                },
                "filter": {...},
                "meta":   {...}
            }
        """
        # primary key
        d: dict = {"id": str(record.get_field(self._pk_slot))}

        # vectors go inside the "fields" dict (v2)
        vector_fields: dict = {}
        vector_fields["embedding"] = record.get_field(self._dense_slot)

        # sparse vector (hybrid only)
        if self._sparse_slot >= 0:
            sp = record.get_field(self._sparse_slot)
            if sp:
                vector_fields["keywords"] = {
                    "indices": sp["indices"],
                    "values":  sp["values"],
                }

        d["fields"] = vector_fields

        # metadata — split payload into filter and meta
        filter_data: dict = {}
        meta_data:   dict = {}

        for slot, ftype, fname in zip(
            self._payload_slots, self._payload_types, self._payload_names
        ):
            value = record.get_field(slot)

            if ftype == FieldType.JSON:
                # whole payload dict — distribute keys into filter/meta
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in self.filter_fields:
                            filter_data[k] = v
                        else:
                            meta_data[k] = v

            elif ftype == FieldType.STRING:
                # individual typed field
                if fname in self.filter_fields:
                    filter_data[fname] = value
                else:
                    meta_data[fname] = value

            else:
                # numeric, bool etc. — always metadata
                meta_data[fname] = value

        d["filter"] = filter_data
        d["meta"]   = meta_data

        return d

    # ── Single chunk upsert ───────────────────────────────────────────────────

    async def _upsert_chunk(self, chunk: List[dict]):
        """
        Upsert one chunk to Endee.
        Wrapped in run_in_executor so the sync Endee SDK never blocks the event loop.
        Raises RuntimeError on falsy result (so gather() captures the failure).
        """
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(chunk),
        )
        if not result:
            raise RuntimeError(f"Endee upsert returned falsy for chunk of {len(chunk)} records")
        logger.debug(f"  Upserted chunk of {len(chunk)} records")

    # ── upsert_batch ──────────────────────────────────────────────────────────

    async def upsert_batch(
        self, records: List[MigrationRow], schema: RowSchema
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Convert records, split into chunks, upsert all in parallel,
        then retry failures with exponential backoff.

        Returns (success, {"tgt_transform": float, "upsert": float}).
        Never raises — errors are logged and False is returned.
        """
        t0 = time.time()
        endee_records = [self._to_endee(r, schema) for r in records]
        transform_time = time.time() - t0

        chunks = [
            endee_records[i: i + self.upsert_chunk_size]
            for i in range(0, len(endee_records), self.upsert_chunk_size)
        ]

        t1 = time.time()

        # Phase 1: all chunks in parallel
        results = await asyncio.gather(
            *[self._upsert_chunk(c) for c in chunks],
            return_exceptions=True,
        )

        # Phase 2: collect failures
        failed = [chunks[i] for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed:
            logger.warning(f"  {len(failed)}/{len(chunks)} chunks failed — retrying...")

        # Phase 3: retry with exponential backoff (1s → 2s → 4s)
        while failed:
            chunk     = failed.pop(0)
            succeeded = False
            for attempt in range(3):
                try:
                    await self._upsert_chunk(chunk)
                    succeeded = True
                    logger.info(f"  Retry {attempt+1} succeeded ({len(chunk)} records)")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"  Retry {attempt+1}/3 failed: {e}. Waiting {wait}s...")
                    await asyncio.sleep(wait)
            if not succeeded:
                logger.error(f"  Chunk of {len(chunk)} records failed after 3 retries")
                return False, {"tgt_transform": transform_time, "upsert": time.time() - t1}

        return True, {"tgt_transform": transform_time, "upsert": time.time() - t1}


    @classmethod
    def from_args(cls, args):
        return cls(
            endee_url         = args.target_url,
            endee_api_key     = args.target_api_key,
            index_name        = args.target_collection,
            upsert_chunk_size = args.upsert_size,
            space_type        = args.space_type,
            M                 = args.M,
            ef_construct      = args.ef_construct,
            precision         = args.precision,
            filter_fields     = args.filter_fields,
            sparse_model      = getattr(args, "sparse_model", DEFAULT_SPARSE_MODEL),
            target_type       = getattr(args, "target_type", "dense"),
        )

    def close(self):
        pass