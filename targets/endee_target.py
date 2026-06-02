"""
    sinks/endee_sink.py
    ============================
    Endee sink connector — handles both dense-only and hybrid indexes.

    A single EndeeSink works for both:
    - Dense  -> IndexConfig.is_hybrid=False -> creates a standard vector index
    - Hybrid -> IndexConfig.is_hybrid=True  -> creates a hybrid (dense+sparse) index

    MigrationRow -> Endee native format conversion:
    Dense record:
        {"id": str, "vector": [...], "filter": {...}, "meta": {...}}
    Hybrid record:
        {"id": str, "vector": [...], "sparse_indices": [...], "sparse_values": [...],
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
from endee.exceptions import NotFoundException
from endee import Precision

from core.base_target import BaseTarget
from core.schema import FieldRole, FieldType, MigrationRow, RowSchema
import time
from constants import DEFAULT_SPARSE_MODEL, ENDEE_V1_API
from core.type_registry import resolve_space, CANONICAL_TO_ENDEE_SPACE_MAPPING, CANONICAL_TO_ENDEE_PRECISION_MAPPING, resolve_precision,  PRECISION_RANK

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
        precision:              Precision = Precision.INT16,   

    ):
        self.endee_url              = endee_url
        self.endee_api_key          = endee_api_key
        self.index_name             = index_name
        self.upsert_chunk_size      = upsert_chunk_size
        self.sparse_model           = sparse_model
        self.space_type             = resolve_space(CANONICAL_TO_ENDEE_SPACE_MAPPING, space_type)
        self.M                      = M
        self.ef_construct           = ef_construct
        self.precision              = precision # CANONICAL PRECISION

        self._client: Any           = None
        self._index:  Any           = None

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
            indexes = self._client.list_indexes()
            logger.info(f"  Reachable — indexes: {indexes}")
        except Exception as e:
            logger.error(f"  Endee unreachable after connect: {e}")
            raise
        logger.info("Connected to Endee")

    # ── setup_index ───────────────────────────────────────────────────────────

    def setup_index(self, schema: RowSchema):
        """
            Creates or gets the Endee index.
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

        # RESOLVE SLOT POISTIONS
        pk_field     = schema.get_primary_key()
        dense_field  = schema.get_dense_vector()
        sparse_field = schema.get_sparse_vector()
        meta_fields  = schema.get_metadata_fields()
        target_canonical_precision = schema.canonical_precision

        # CHECK PRECISION DOWNGRADE
        source_db_precision_rank = PRECISION_RANK.get(target_canonical_precision)
        endee_db_precision_rank = PRECISION_RANK.get(self.precision)
        if endee_db_precision_rank > source_db_precision_rank:
            logger.error(f"Precision Upgrade Not Allowed: {target_canonical_precision} -> {self.precision}")
            sys.exit(1)
        elif endee_db_precision_rank < source_db_precision_rank:
            logger.warning(f"Precision Downgrade Detected: {target_canonical_precision} -> {self.precision}")

        self.endee_precision = resolve_precision(CANONICAL_TO_ENDEE_PRECISION_MAPPING, self.precision)

        self.dimension = dense_field.dimension
        # REQUIRED FIELDS
        if pk_field is None:
            raise ValueError("RowSchema has no PRIMARY_KEY field — cannot create Endee index")
        if dense_field is None:
            raise ValueError("RowSchema has no DENSE_VECTOR field — cannot create Endee index")

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

        # NO INDEX - CREATE NEW ONE
        try:
            self._index = self._client.get_index(self.index_name)
            logger.info(f"Index already exists: {self.index_name}")
            return
        except NotFoundException:
            pass

        kwargs = dict(
            name       = self.index_name,
            dimension  = self.dimension,         
            space_type = self.space_type,        # from RowSchema
            M          = self.M,                 # from RowSchema
            ef_con     = self.ef_construct,      # from RowSchema
            precision  = self.endee_precision,         # from RowSchema
        )
        if schema.is_hybrid:
            kwargs["sparse_model"] = self.sparse_model
            logger.info(f"Creating HYBRID index '{self.index_name}'")
        else:
            logger.info(f"Creating DENSE index '{self.index_name}'")

        self._client.create_index(**kwargs)
        self._index = self._client.get_index(self.index_name)
        logger.info(f"Created index: {self.index_name}")

    # ── Record conversion ─────────────────────────────────────────────────────


    def _to_endee(self, record: MigrationRow, schema: RowSchema) -> dict:
        """ Convert canonical MigrationRecord to Endee's native dict format.
            Converts MigrationRow to Endee native dict.
            Uses pre-resolved slot indexes — NO schema lookups per row (fast).
            Uses FieldRole — NO hardcoded attribute names like row.dense_vector.
            filter_fields split happens HERE — source never knew about it.
        """
        
        # primary key
        d = {"id": str(record.get_field(self._pk_slot))}

        # dense vector
        d["vector"] = record.get_field(self._dense_slot)

        # sparse vector (hybrid only)
        if self._sparse_slot >= 0:
            sp = record.get_field(self._sparse_slot)
            if sp:
                d["sparse_indices"] = sp["indices"]
                d["sparse_values"]  = sp["values"]

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
            lambda: self._index.upsert(chunk),
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
            sparse_model      = getattr(args, "sparse_model", DEFAULT_SPARSE_MODEL)
        )

    def close(self):
        pass