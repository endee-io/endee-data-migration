"""
sources/qdrant_source.py
──────────────────────────────────────────────────────────────────────────────
Qdrant source connectors (dense-only and hybrid).

QdrantDenseSource  — validates dense-only collections, rejects hybrid ones.
QdrantHybridSource — validates hybrid collections (dense+sparse), rejects dense-only.

Both share a common QdrantBaseSource that handles:
  - Connection
  - Collection config detection (dimension, space, HNSW, quantization - precision)
  - Dense field detection (named or unnamed vectors)
  - Sparse field detection
  - Scroll-based async batch iteration with retry

Cursor format
─────────────
Qdrant uses the scroll API's offset cursor (a UUID or None).
Cursor = None means "start from the beginning" (or "end of collection").
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from core.base_source import BaseSource
from core.record      import IndexConfig, MigrationRecord

logger = logging.getLogger(__name__)

# ── Space-type mapping ────────────────────────────────────────────────────────
QDRANT_TO_ENDEE_SPACE = {
    Distance.COSINE: "cosine",
    Distance.EUCLID: "l2",
    Distance.DOT:    "ip",
}

# ── Precision helpers ─────────────────────────────────────────────────────────
PRECISION_STR_TO_ENDEE: Dict[str, Any] = {}
PRECISION_RANK:         Dict[Any, int]  = {}
PRECISION_NAMES:        Dict[Any, str]  = {}

def _fill_precision_maps():
    if PRECISION_STR_TO_ENDEE:
        return
    from endee import Precision
    PRECISION_STR_TO_ENDEE.update({
        "float32": Precision.FLOAT32,
        "int8":    Precision.INT8,
        "int16":   Precision.INT16,
        "binary":  Precision.BINARY2,
    })
    PRECISION_RANK.update({
        Precision.BINARY2:  0,
        Precision.INT8:     1,
        Precision.INT16:    2,
        Precision.FLOAT32:  3,
    })
    PRECISION_NAMES.update({
        Precision.BINARY2:  "binary",
        Precision.INT8:     "int8",
        Precision.INT16:    "int16",
        Precision.FLOAT32:  "float32",
    })


def _resolve_precision(name: str):
    _fill_precision_maps()
    return PRECISION_STR_TO_ENDEE[name]


def _validate_precision_downgrade(user_precision, source_precision):
    _fill_precision_maps()
    source_rank = PRECISION_RANK.get(source_precision)
    if source_rank is None:
        logger.warning(f"Source precision not detected ('{source_precision}'). Skipping check.")
        return
    user_rank = PRECISION_RANK[user_precision]
    if user_rank > source_rank:
        valid = ", ".join(
            PRECISION_NAMES[p]
            for p in PRECISION_RANK
            if PRECISION_RANK[p] <= source_rank
        )
        raise ValueError(
            f"Precision upgrade not allowed.\n"
            f"  Source   : {PRECISION_NAMES[source_precision]}\n"
            f"  Requested: {PRECISION_NAMES[user_precision]}\n"
            f"  Valid    : {valid}"
        )
    logger.info(
        f"Precision check passed: "
        f"{PRECISION_NAMES[source_precision]} (source) → {PRECISION_NAMES[user_precision]} (target)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared Qdrant base
# ══════════════════════════════════════════════════════════════════════════════
class QdrantBaseSource(BaseSource):
    """
    Internal base class — all Qdrant plumbing lives here.
    Subclasses only override _validate_schema().
    """

    DEFAULT_SPACE = "cosine"

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str = "",
        port: Optional[int] = None,
        use_https: bool = False,
        space_type: Optional[str] = None,   # override detected space
        M: Optional[int] = None,
        ef_construct: Optional[int] = None,
        precision: Optional[str] = None,    # e.g. 'int16'
        filter_fields: str = "",
    ):
        self.url        = url
        self.collection = collection
        self.api_key    = api_key
        self.port       = port
        self.use_https  = use_https
        self._space_override    = space_type
        self._user_M            = M
        self._user_ef_construct = ef_construct
        self._user_precision    = precision

        self.filter_fields: set = (
            set(f.strip() for f in filter_fields.split(",") if f.strip())
            if filter_fields else set()
        )

        # Populated by detect_schema()
        self.qdrant_client:       Optional[QdrantClient] = None
        self.dimension:           Optional[int]           = None
        self.space_type:          Optional[str]           = None
        self.M:                   Optional[int]           = None
        self.ef_construct:        Optional[int]           = None
        self._resolved_precision                          = None
        self.dense_field_name:    Optional[str]           = None
        self.sparse_field_name:   Optional[str]           = None

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to Qdrant...")
        try:
            self.qdrant_client = QdrantClient(
                url=self.url,
                port=self.port,
                api_key=self.api_key,
                https=bool(self.use_https),
            )
            self.qdrant_client.get_collections()
            logger.info("✓ Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            sys.exit(1)

    # ── Schema detection ──────────────────────────────────────────────────────

    def detect_schema(self):
        logger.info(f"\n{'='*80}\nDetecting collection config: {self.collection}\n{'='*80}")
        _fill_precision_maps()

        info   = self.qdrant_client.get_collection(self.collection)
        params = info.config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors

        # ── Detect dense field ────────────────────────────────────────────────
        if isinstance(vectors, dict):
            # Named vectors → find first with a .size
            for name, cfg in vectors.items():
                if hasattr(cfg, "size") and cfg.size is not None:
                    self.dense_field_name = name
                    self.dimension  = cfg.size
                    qdrant_space    = cfg.distance
                    break
        elif vectors is not None:
            self.dense_field_name = "default"
            self.dimension  = vectors.size
            qdrant_space    = vectors.distance
        else:
            raise ValueError("No vector config found in collection")

        self.space_type = (
            self._space_override
            or QDRANT_TO_ENDEE_SPACE.get(qdrant_space, "cosine")
        )
        logger.info(f"  [DENSE]  field='{self.dense_field_name}', dim={self.dimension}, space={self.space_type}")

        # ── Detect sparse field ───────────────────────────────────────────────
        if sparse_vectors and isinstance(sparse_vectors, dict):
            for name in sparse_vectors:
                self.sparse_field_name = name
                logger.info(f"  [SPARSE] field='{name}'")
                break

        # ── HNSW params ───────────────────────────────────────────────────────
        source_M  = info.config.hnsw_config.m
        source_ef = info.config.hnsw_config.ef_construct
        self.M            = self._user_M            if self._user_M            is not None else source_M
        self.ef_construct = self._user_ef_construct if self._user_ef_construct is not None else source_ef
        logger.info(f"  HNSW: M={self.M}, ef_construct={self.ef_construct}")

        # ── Quantization → precision ──────────────────────────────────────────
        from endee import Precision
        if info.config.quantization_config:
            qcfg = dict(info.config.quantization_config)
            if "scalar" in qcfg:
                source_prec = Precision.INT8
            elif "binary" in qcfg:
                binary_cfg = qcfg.get("binary", {})
                if "query_encoding" in binary_cfg:
                    raise ValueError(f"Asymmetric binary quantization is not supported: {qcfg}")
                source_prec = Precision.BINARY2
            elif "product" in qcfg:
                raise ValueError(f"Product quantization is not supported: {qcfg}")
            else:
                source_prec = Precision.FLOAT32
        else:
            source_prec = Precision.FLOAT32

        if self._user_precision:
            user_p = _resolve_precision(self._user_precision)
            _validate_precision_downgrade(user_p, source_prec)
            self._resolved_precision = user_p
        else:
            self._resolved_precision = source_prec

        logger.info(f"  Precision: {PRECISION_NAMES.get(self._resolved_precision, self._resolved_precision)}")

        # Subclass validates sparse presence/absence
        self._validate_schema(vectors, sparse_vectors)

    def _validate_schema(self, vectors, sparse_vectors):
        """Override in subclass."""

    # ── Index config ──────────────────────────────────────────────────────────

    def get_index_config(self) -> IndexConfig:
        return IndexConfig(
            dimension    = self.dimension,
            space_type   = self.space_type,
            M            = self.M,
            ef_construct = self.ef_construct,
            precision    = self._resolved_precision,
            is_hybrid    = self.sparse_field_name is not None,
        )

    # ── Record conversion ─────────────────────────────────────────────────────

    def _convert_records(self, points) -> List[MigrationRecord]:
        records = []
        for pt in points:
            try:
                vec_data = pt.vector
                if isinstance(vec_data, dict):
                    dense  = vec_data.get(self.dense_field_name)
                    sparse = vec_data.get(self.sparse_field_name) if self.sparse_field_name else None
                else:
                    dense  = vec_data
                    sparse = None

                payload = pt.payload or {}
                if self.filter_fields:
                    filter_data = {k: v for k, v in payload.items() if k in self.filter_fields}
                    meta_data   = {k: v for k, v in payload.items() if k not in self.filter_fields}
                else:
                    filter_data = {}
                    meta_data   = payload

                mr = MigrationRecord(
                    id           = str(pt.id),
                    dense_vector = dense,
                    filter_data  = filter_data,
                    meta_data    = meta_data,
                )
                if sparse is not None:
                    mr.sparse_indices = list(sparse.indices)
                    mr.sparse_values  = list(sparse.values)

                records.append(mr)
            except Exception as e:
                import traceback
                logger.error(f"Error converting point {pt.id}: {e}")
                logger.error(traceback.format_exc())
                continue
        return records

    # ── Scroll with retry ─────────────────────────────────────────────────────

    async def _scroll(self, offset, batch_size: int, loop):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.qdrant_client.scroll(
                            collection_name=self.collection,
                            limit=batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=True,
                        ),
                    ),
                    timeout=60,
                )
                return result  # (points_batch, next_offset)
            except Exception as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    logger.warning(f"Scroll failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Scroll failed after {max_retries} attempts: {e}")
                    raise

    # ── iterate_batches ───────────────────────────────────────────────────────

    async def iterate_batches(self, batch_size: int, initial_cursor: Any):
        """
        Async generator using Qdrant scroll API.

        Cursor: scroll offset (UUID string or None).
        initial_cursor=None means start from the beginning.
        Yields: (List[MigrationRecord], next_cursor)
        """
        loop   = asyncio.get_running_loop()
        offset = initial_cursor   # None = start of collection

        logger.info(f"PRODUCER: starting Qdrant scroll from offset={offset}")

        while True:
            points_batch, next_offset = await self._scroll(offset, batch_size, loop)

            if not points_batch:
                logger.info("PRODUCER: no more data from Qdrant")
                return

            records = self._convert_records(points_batch)
            yield records, next_offset

            if next_offset is None:
                logger.info("PRODUCER: reached end of Qdrant collection (next_offset=None)")
                return

            offset = next_offset

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Public connectors
# ══════════════════════════════════════════════════════════════════════════════

class QdrantDenseSource(QdrantBaseSource):
    """
    Qdrant source for DENSE-ONLY collections.
    Raises ValueError if sparse vector fields are detected.
    """

    def _validate_schema(self, vectors, sparse_vectors):
        has_sparse = bool(sparse_vectors and isinstance(sparse_vectors, dict) and sparse_vectors)
        if has_sparse:
            raise ValueError(
                f"Collection '{self.collection}' is HYBRID "
                f"(sparse fields: {list(sparse_vectors.keys())}).\n"
                f"Use QdrantHybridSource instead."
            )
        logger.info("Schema validated: dense-only collection")


class QdrantHybridSource(QdrantBaseSource):
    """
    Qdrant source for HYBRID collections (dense + sparse).
    Raises ValueError if no sparse vector fields are detected.
    """

    def _validate_schema(self, vectors, sparse_vectors):
        has_sparse = bool(sparse_vectors and isinstance(sparse_vectors, dict) and sparse_vectors)
        # Dense-only = VectorParams (not a dict) AND no sparse
        is_dense_only = not isinstance(vectors, dict) and not has_sparse
        if is_dense_only:
            raise ValueError(
                f"Collection '{self.collection}' is DENSE-ONLY.\n"
                f"Use QdrantDenseSource instead."
            )
        if not has_sparse:
            raise ValueError(
                f"Collection '{self.collection}' has no sparse vector fields.\n"
                f"Use QdrantDenseSource instead."
            )
        logger.info(
            f"Schema validated: hybrid collection "
            f"(dense={self.dense_field_name}, sparse={self.sparse_field_name})"
        )