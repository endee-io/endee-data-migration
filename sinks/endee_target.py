"""
sinks/endee_sink.py
──────────────────────────────────────────────────────────────────────────────
Endee sink connector — handles both dense-only and hybrid indexes.

A single EndeeSink works for both:
  • Dense  → IndexConfig.is_hybrid=False → creates a standard vector index
  • Hybrid → IndexConfig.is_hybrid=True  → creates a hybrid (dense+sparse) index

MigrationRecord → Endee native format conversion:
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
from typing import Any, List

from endee import Endee
from endee.exceptions import NotFoundException

from core.base_sink import BaseSink
from core.record    import IndexConfig, MigrationRecord
import time

logger = logging.getLogger(__name__)

DEFAULT_SPARSE_MODEL = "bm25"
ENDEE_V1_API         = "/api/v1/"


class EndeeSink(BaseSink):
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
        endee_url: str,
        endee_api_key: str,
        index_name: str,
        upsert_chunk_size: int = 100,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
    ):
        self.endee_url        = endee_url
        self.endee_api_key    = endee_api_key
        self.index_name       = index_name
        self.upsert_chunk_size = upsert_chunk_size
        self.sparse_model     = sparse_model

        self._client: Any = None
        self._index:  Any = None

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to Endee...")
        self._client = Endee(token=self.endee_api_key)
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, ENDEE_V1_API)
            self._client.set_base_url(url)
            logger.info(f"  Base URL: {url}")
        logger.info(f"  Available indexes: {self._client.list_indexes()}")
        logger.info("✓ Connected to Endee")

    # ── setup_index ───────────────────────────────────────────────────────────

    def setup_index(self, config: IndexConfig):
        """Get or create the Endee index from IndexConfig."""
        try:
            self._index = self._client.get_index(self.index_name)
            logger.info(f"✓ Index already exists: {self.index_name}")
            return
        except NotFoundException:
            pass

        kwargs = dict(
            name       = self.index_name,
            dimension  = config.dimension,
            space_type = config.space_type,
            M          = config.M,
            ef_con     = config.ef_construct,
            precision  = config.precision,
        )

        if config.is_hybrid:
            kwargs["sparse_model"] = config.sparse_model or self.sparse_model
            logger.info(f"Creating HYBRID index '{self.index_name}' …")
            logger.info(f"  dense_dim   : {config.dimension}")
            logger.info(f"  sparse_model: {kwargs['sparse_model']}")
        else:
            logger.info(f"Creating DENSE index '{self.index_name}' …")
            logger.info(f"  dimension   : {config.dimension}")

        logger.info(f"  space_type  : {config.space_type}")
        logger.info(f"  M           : {config.M}")
        logger.info(f"  ef_construct: {config.ef_construct}")
        logger.info(f"  precision   : {config.precision}")

        self._client.create_index(**kwargs)
        self._index = self._client.get_index(self.index_name)
        logger.info(f"✓ Created {'HYBRID' if config.is_hybrid else 'DENSE'} index: {self.index_name}")

    # ── Record conversion ─────────────────────────────────────────────────────

    @staticmethod
    def _to_endee(record: MigrationRecord) -> dict:
        """Convert canonical MigrationRecord to Endee's native dict format."""
        d: dict = {
            "id":     record.id,
            "vector": record.dense_vector,
            "filter": record.filter_data,
            "meta":   record.meta_data,
        }
        if record.is_hybrid:
            d["sparse_indices"] = record.sparse_indices
            d["sparse_values"]  = record.sparse_values
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

    async def upsert_batch(self, records: List[MigrationRecord]) -> bool:
        """
        Convert records, split into chunks, upsert all in parallel,
        then retry failures with exponential backoff.

        Returns True on full success, False if any chunk exhausts retries.
        Never raises — errors are logged and False is returned.
        """
        t0 = time.time()
        endee_records = [self._to_endee(r) for r in records]
        transform_time = time.time() - t0
        logger.info(
            f"  [COMMON→ENDEE] {len(endee_records)} records converted "
            f"in {transform_time:.3f}s"
        )
        chunks = [
            endee_records[i: i + self.upsert_chunk_size]
            for i in range(0, len(endee_records), self.upsert_chunk_size)
        ]

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
                return False

        return True


    @classmethod
    def from_args(cls, args):
        return cls(
            endee_url         = args.target_url,
            endee_api_key     = args.target_api_key,
            index_name        = args.target_collection,
            upsert_chunk_size = args.upsert_size,
        )

    def close(self):
        pass