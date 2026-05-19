"""
sources/milvus_source.py
──────────────────────────────────────────────────────────────────────────────
Milvus source connectors (dense-only and hybrid).

MilvusDenseSource  — validates dense-only collections, rejects hybrid ones.
MilvusHybridSource — validates hybrid collections (dense+sparse), rejects dense-only.

Both share a common MilvusBaseSource that handles:
  - Connection (with protocol auto-fix)
  - Collection loading (with timeout)
  - Field schema detection
  - HNSW parameter detection
  - Vector byte decoding (FLOAT16, FLOAT32, BINARY)
  - QueryIterator-based async batch iteration (no 16384 offset cap)
  - Checkpoint-skip on resume (count-based)

Cursor format
─────────────
Milvus uses a COUNT cursor: the number of records processed so far.
On resume, the iterator skips that many records from the beginning.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import DataType, MilvusClient
from pymilvus import MilvusException

from core.base_source import BaseSource
from core.checkpoint  import MigrationCheckpoint
from core.record      import IndexConfig, MigrationRecord

logger = logging.getLogger(__name__)

# ── Precision mappings ────────────────────────────────────────────────────────
# Imported lazily so this file works without endee installed if only
# reading source docs.  The actual Precision enum is only needed in
# get_index_config(), which is called after connect().
def _endee_precision():
    from endee import Precision
    return Precision

MILVUS_DTYPE_TO_PRECISION_NAME = {
    DataType.FLOAT_VECTOR:    "float32",
    DataType.FLOAT16_VECTOR:  "float16",
    DataType.BFLOAT16_VECTOR: "float16",
    DataType.BINARY_VECTOR:   "binary",
}
MILVUS_STR_TO_PRECISION_NAME = {
    "FLOAT_VECTOR":    "float32",
    "FLOAT16_VECTOR":  "float16",
    "BFLOAT16_VECTOR": "float16",
    "BINARY_VECTOR":   "binary",
}
PRECISION_STR_TO_ENDEE = {
    "float32": None,   # filled lazily
    "float16": None,
    "int8":    None,
    "int16":   None,
    "binary":  None,
}
PRECISION_RANK = {}    # filled lazily

def _fill_precision_maps():
    """Populate PRECISION_STR_TO_ENDEE and PRECISION_RANK on first use."""
    from endee import Precision
    PRECISION_STR_TO_ENDEE.update({
        "float32": Precision.FLOAT32,
        "float16": Precision.FLOAT16,
        "int8":    Precision.INT8,
        "int16":   Precision.INT16,
        "binary":  Precision.BINARY2,
    })
    PRECISION_RANK.update({
        Precision.BINARY2:  0,
        Precision.INT8:     1,
        Precision.INT16:    2,
        Precision.FLOAT16:  3,
        Precision.FLOAT32:  4,
    })

PRECISION_NAMES = {}   # filled lazily alongside PRECISION_RANK

def _fill_precision_names():
    from endee import Precision
    PRECISION_NAMES.update({
        Precision.BINARY2:  "binary",
        Precision.INT8:     "int8",
        Precision.INT16:    "int16",
        Precision.FLOAT16:  "float16",
        Precision.FLOAT32:  "float32",
    })


def _resolve_precision(name: str):
    _fill_precision_maps()
    _fill_precision_names()
    return PRECISION_STR_TO_ENDEE[name]


def _validate_precision_downgrade(user_precision, source_precision):
    _fill_precision_maps()
    _fill_precision_names()
    source_rank = PRECISION_RANK.get(source_precision)
    if source_rank is None:
        logger.warning(
            f"Source precision not detected (got '{source_precision}'). "
            "Skipping downgrade check."
        )
        return
    user_rank = PRECISION_RANK[user_precision]
    if user_rank > source_rank:
        valid = ", ".join(
            PRECISION_NAMES[p]
            for p in PRECISION_RANK
            if PRECISION_RANK[p] <= source_rank and p in set(PRECISION_STR_TO_ENDEE.values())
        )
        raise ValueError(
            f"Precision upgrade not allowed.\n"
            f"  Source  : {PRECISION_NAMES[source_precision]}\n"
            f"  Requested: {PRECISION_NAMES[user_precision]}\n"
            f"  Valid choices: {valid}"
        )
    logger.info(
        f"Precision check passed: "
        f"{PRECISION_NAMES[source_precision]} (source) - {PRECISION_NAMES[user_precision]} (target)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared Milvus base (not exposed as a public connector)
# ══════════════════════════════════════════════════════════════════════════════
class MilvusBaseSource(BaseSource):
    """
    Internal base class with all Milvus plumbing.
    Subclasses only override _validate_schema().
    """

    DEFAULT_PORT = 19530
    DEFAULT_SPACE = "cosine"

    def __init__(
        self,
        url: str,
        token: str,
        collection: str,
        db: str = "default",
        port: int = DEFAULT_PORT,
        space_type: str = DEFAULT_SPACE,
        M: Optional[int] = None,
        ef_construct: Optional[int] = None,
        precision: Optional[str] = None,   # 'float32' | 'float16' | 'int8' | 'int16' | 'binary'
        filter_fields: str = "",
    ):
        self.url        = url
        self.token      = token
        self.db         = db
        self.collection = collection
        self.port       = port
        self.space_type = space_type
        self._user_M            = M
        self._user_ef_construct = ef_construct
        self._user_precision    = precision  # str or None

        self.filter_fields: set = (
            set(f.strip() for f in filter_fields.split(",") if f.strip())
            if filter_fields else set()
        )

        # Populated by detect_schema()
        self.milvus_client: Optional[MilvusClient] = None
        self.id_field_name:           Optional[str] = None
        self.dense_field_name:        Optional[str] = None
        self.dense_field_type               = None
        self.vectors_dimension:       Optional[int] = None
        self.sparse_field_name:       Optional[str] = None
        self.other_field_names:       List[str]     = []
        self.M:                       Optional[int] = None
        self.ef_construct:            Optional[int] = None
        self._resolved_precision              = None   # endee.Precision value

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to Milvus...")
        uri = self.url
        if not uri.startswith(("http://", "https://", "tcp://", "unix://")):
            if uri.startswith("localhost") or uri.replace(".", "").replace(":", "").isdigit():
                uri = f"http://{uri}:{self.port}"
                logger.info(f"Auto-added protocol: {uri}")
        try:
            self.milvus_client = MilvusClient(uri=uri, token=self.token, db_name=self.db)
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            sys.exit(1)

    # -- COLLECTION LOADING ------------------------------------------------------

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
                    logger.info("Collection loaded")
                    return
            logger.error(f"Collection did not load within 300s. Last state: {state}")
            sys.exit(1)
        except MilvusException as e:
            if "index not found" in str(e).lower():
                logger.error(
                    f"Collection has no vector index and cannot be loaded.\n"
                    f"Build an index first, then re-run the migration."
                )
            else:
                logger.error(f"Failed to load collection: {e}")
            sys.exit(1)

    # ── Schema detection ──────────────────────────────────────────────────────

    def detect_schema(self):
        self._load_collection()

        desc = self.milvus_client.describe_collection(self.collection)
        logger.info(f"\n{'='*80}\nDetecting fields in: {self.collection}\n{'='*80}")

        dense_fields, sparse_fields, other_fields = [], [], []
        id_field = None

        for field in desc.get("fields", []):
            name   = field.get("name")
            ftype  = field.get("type")
            is_pk  = field.get("is_primary", False)

            if is_pk:
                id_field = {"name": name, "type": ftype}
                self.id_field_name = name
                logger.info(f"  [PK]     {name} [{ftype}]")

            elif ftype in [DataType.BFLOAT16_VECTOR, "BFLOAT16_VECTOR"]:
                raise ValueError(
                    f"BFLOAT16_VECTOR not supported in field '{name}'. "
                    "Convert to FLOAT32 or FLOAT16 before migrating."
                )

            elif ftype in ["FLOAT_VECTOR", "FLOAT16_VECTOR", "BINARY_VECTOR",
                           DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR,
                           DataType.BINARY_VECTOR]:
                params = field.get("params", {})
                dim    = params.get("dim") or field.get("dim")

                # Try to read HNSW params from the index
                index_info = self.milvus_client.describe_index(self.collection, name) or {}
                source_M  = index_info.get("params", {}).get("M")
                source_ef = index_info.get("params", {}).get("efConstruction")

                if self.M is None:
                    self.M = self._user_M if self._user_M is not None else source_M
                if self.ef_construct is None:
                    self.ef_construct = self._user_ef_construct if self._user_ef_construct is not None else source_ef

                prec_name = (
                    MILVUS_DTYPE_TO_PRECISION_NAME.get(ftype)
                    or MILVUS_STR_TO_PRECISION_NAME.get(ftype)
                    or "int16"
                )
                source_precision = _resolve_precision(prec_name)

                dense_fields.append({"name": name, "type": ftype, "dim": dim, "precision": source_precision})

                if self.dense_field_name is None:
                    self.dense_field_name = name
                    self.dense_field_type = ftype
                    self.vectors_dimension = dim

                    if self._user_precision:
                        user_p = _resolve_precision(self._user_precision)
                        _validate_precision_downgrade(user_p, source_precision)
                        self._resolved_precision = user_p
                    else:
                        self._resolved_precision = source_precision

                logger.info(
                    f"  [DENSE]  {name} [{ftype}, dim={dim}, "
                    f"precision={_fill_precision_names() or PRECISION_NAMES.get(self._resolved_precision)}]"
                )

            elif ftype in ["SPARSE_FLOAT_VECTOR", DataType.SPARSE_FLOAT_VECTOR]:
                sparse_fields.append({"name": name, "type": ftype})
                if self.sparse_field_name is None:
                    self.sparse_field_name = name
                logger.info(f"  [SPARSE] {name} [{ftype}]")

            else:
                other_fields.append(name)
                logger.info(f"  [META]   {name} [{ftype}]")

        self.other_field_names = other_fields

        logger.info(f"\nPK: {self.id_field_name} | Dense: {len(dense_fields)} | "
                    f"Sparse: {len(sparse_fields)} | Meta: {len(other_fields)}")

        if not self.id_field_name:
            raise ValueError("No primary key field found in collection")
        if not self.dense_field_name:
            raise ValueError("No dense vector field found in collection")

        # Subclass decides whether sparse presence/absence is valid
        self._validate_schema(dense_fields, sparse_fields)

    def _validate_schema(self, dense_fields, sparse_fields):
        """Override in subclass to accept or reject the detected schema."""

    # ── Index config ──────────────────────────────────────────────────────────

    def get_index_config(self) -> IndexConfig:
        _fill_precision_maps()
        _fill_precision_names()
        return IndexConfig(
            dimension    = self.vectors_dimension,
            space_type   = self.space_type,
            M            = self.M,
            ef_construct = self.ef_construct,
            precision    = self._resolved_precision,
            is_hybrid    = self.sparse_field_name is not None,
        )

    # ── Vector decoding ───────────────────────────────────────────────────────

    def _decode_vector(self, raw, field_type) -> List[float]:
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], bytes):
                raw = raw[0]
            elif raw and isinstance(raw[0], (int, float)):
                return raw
            else:
                raw = raw[0] if raw else b""
        if isinstance(raw, bytes):
            if field_type in (DataType.FLOAT16_VECTOR, "FLOAT16_VECTOR"):
                return np.frombuffer(raw, dtype=np.float16).astype(np.float32).tolist()
            if field_type in (DataType.FLOAT_VECTOR, "FLOAT_VECTOR"):
                return np.frombuffer(raw, dtype=np.float32).tolist()
            # Fallback
            logger.warning(f"Unknown field type {field_type}, attempting float16 decode")
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).tolist()
        return raw  # already usable

    # ── Record conversion ─────────────────────────────────────────────────────

    def _convert_records(self, milvus_records) -> List[MigrationRecord]:
        records = []
        payload_fields = set(self.other_field_names)

        for rec in milvus_records:
            try:
                rid = str(rec.get(self.id_field_name, ""))
                raw = rec.get(self.dense_field_name, [])
                dense = self._decode_vector(raw, self.dense_field_type)

                if self.filter_fields:
                    filter_data = {k: v for k, v in rec.items() if k in self.filter_fields}
                    meta_data   = {k: v for k, v in rec.items()
                                   if k not in self.filter_fields and k in payload_fields}
                else:
                    filter_data = {}
                    meta_data   = {k: v for k, v in rec.items() if k in payload_fields}

                mr = MigrationRecord(
                    id           = rid,
                    dense_vector = dense,
                    filter_data  = filter_data,
                    meta_data    = meta_data,
                )

                # Sparse data (only set if field exists)
                if self.sparse_field_name:
                    sparse_data = rec.get(self.sparse_field_name, {})
                    if sparse_data and isinstance(sparse_data, dict):
                        sorted_items = sorted(sparse_data.items())
                        mr.sparse_indices = [int(i) for i, _ in sorted_items]
                        mr.sparse_values  = [float(v) for _, v in sorted_items]

                records.append(mr)
            except Exception as e:
                import traceback
                logger.error(f"Error converting record {rec.get(self.id_field_name,'?')}: {e}")
                logger.error(traceback.format_exc())
                continue  # skip bad record, keep migrating

        return records

    # ── iterate_batches ───────────────────────────────────────────────────────

    async def iterate_batches(self, batch_size: int, initial_cursor: Any):
        """
        Async generator using Milvus QueryIterator (no 16384 offset cap).

        Cursor: int — total records processed before this run.
        On resume: skips `initial_cursor` records from the iterator head.
        Yields: (List[MigrationRecord], new_cursor: int)
        """
        loop = asyncio.get_running_loop()
        records_to_skip = initial_cursor or 0
        fetched_this_run = 0

        logger.info("PRODUCER: creating Milvus QueryIterator")
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
                    batch = await loop.run_in_executor(None, iterator.next)
                    if not batch:
                        logger.warning("PRODUCER: ran out of records while skipping — already done?")
                        return
                    skipped += len(batch)
                logger.info(f"PRODUCER: skipped {skipped} records, resuming")

            while True:
                batch = await loop.run_in_executor(None, iterator.next)
                if not batch:
                    logger.info("PRODUCER: no more data from Milvus")
                    return

                records = self._convert_records(batch)
                fetched_this_run += len(records)
                next_cursor = records_to_skip + fetched_this_run
                yield records, next_cursor

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
    """

    def _validate_schema(self, dense_fields, sparse_fields):
        if sparse_fields:
            raise ValueError(
                f"Collection '{self.collection}' is HYBRID "
                f"(sparse fields: {[f['name'] for f in sparse_fields]}).\n"
                f"Use MilvusHybridSource instead."
            )
        logger.info("Schema validated: dense-only collection")


class MilvusHybridSource(MilvusBaseSource):
    """
    Milvus source for HYBRID collections (dense + sparse).
    Raises ValueError if no sparse vector fields are detected.
    Also computes sparse_dimension from a sample (needed by the sink).
    """

    def _validate_schema(self, dense_fields, sparse_fields):
        if not sparse_fields:
            raise ValueError(
                f"Collection '{self.collection}' is DENSE-ONLY "
                f"(no sparse fields detected).\n"
                f"Use MilvusDenseSource instead."
            )
        logger.info(
            f"✓ Schema validated: hybrid collection "
            f"(dense={self.dense_field_name}, sparse={self.sparse_field_name})"
        )

    def _detect_sparse_dimension(self) -> int:
        """Sample up to 1000 records to find max sparse index, add 10% buffer."""
        logger.info("Detecting sparse dimension from data sample...")
        max_idx = 0
        try:
            sample = self.milvus_client.query(
                collection_name=self.collection,
                filter="",
                output_fields=["*"],
                limit=1000,
                offset=0,
            )
            for rec in sample:
                sd = rec.get(self.sparse_field_name, {})
                if sd and isinstance(sd, dict):
                    local_max = max((int(i) for i in sd.keys()), default=0)
                    max_idx = max(max_idx, local_max)
            sparse_dim = int(max_idx * 1.1) + 100
            logger.info(f"  Max sparse index: {max_idx} → sparse_dim={sparse_dim}")
            return sparse_dim
        except Exception as e:
            logger.warning(f"Could not detect sparse dimension ({e}). Defaulting to 30000.")
            return 30000

    def get_index_config(self) -> IndexConfig:
        config = super().get_index_config()
        config.sparse_dimension = self._detect_sparse_dimension()
        return config