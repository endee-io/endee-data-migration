import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import numpy as np
from endee import Endee, Precision
from endee.exceptions import NotFoundException
from pymilvus import DataType, MilvusClient
from tqdm import tqdm
import argparse
from constants import *

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Precision mapping ───────────────────────────────────────────────
MILVUS_DTYPE_TO_ENDEE_PRECISION = {
    DataType.FLOAT_VECTOR:   Precision.FLOAT32,
    DataType.FLOAT16_VECTOR: Precision.FLOAT16,
    DataType.BINARY_VECTOR:  Precision.BINARY2,
}
MILVUS_STR_TO_ENDEE_PRECISION = {
    "FLOAT_VECTOR":    Precision.FLOAT32,
    "FLOAT16_VECTOR":  Precision.FLOAT16,
    "BFLOAT16_VECTOR": Precision.FLOAT16,
    "BINARY_VECTOR":   Precision.BINARY2,
}


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════
class MigrationCheckpoint:
    """Simple checkpoint for resume capability"""
    
    def __init__(self, checkpoint_file: str = CHECKPOINT_FILE):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load checkpoint from file"""

        exception_resposne = {
                PROCESSED_COUNT_KEY: DEFAULT_PROCESSED_COUNT,
                LAST_OFFSET_KEY: DEFAULT_LAST_OFFSET,
                BATCH_NUMBER_KEY: DEFAULT_BATCH_NUMBER
            }

        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                logger.info(f"✓ Loaded checkpoint: {data.get(PROCESSED_COUNT_KEY, DEFAULT_PROCESSED_COUNT)} records processed")
                return data
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh migration")
            return exception_resposne
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}, starting fresh")
            return exception_resposne
    
    def save(self):
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update(self, batch_number: int, records_count: int, offset: int):
        """Update checkpoint after successful batch"""
        self.data[PROCESSED_COUNT_KEY] += records_count
        self.data[BATCH_NUMBER_KEY] = batch_number
        self.data[LAST_OFFSET_KEY] = offset
        self.save()

    def get_last_offset(self) -> int:
        """Get the last processed offset"""
        return self.data.get(LAST_OFFSET_KEY, DEFAULT_LAST_OFFSET)
    
    def get_batch_number(self) -> int:
        """Get the last processed batch number"""
        return self.data.get(BATCH_NUMBER_KEY, DEFAULT_BATCH_NUMBER)
    
    def get_processed_count(self) -> int:
        """Get total processed records"""
        return self.data.get(PROCESSED_COUNT_KEY, DEFAULT_PROCESSED_COUNT)
    
    def clear(self):
        """Clear checkpoint for fresh start"""
        self.data = {
            PROCESSED_COUNT_KEY: DEFAULT_PROCESSED_COUNT,
            LAST_OFFSET_KEY: DEFAULT_LAST_OFFSET,
            BATCH_NUMBER_KEY: DEFAULT_BATCH_NUMBER
        }
        self.save()


# ══════════════════════════════════════════════════════════════════
# MIGRATOR
# ══════════════════════════════════════════════════════════════════
class AsyncHybridMilvusToEndeeMigrator:
    """
    Async producer-consumer migration: Milvus (hybrid dense+sparse) → Endee.

    Architecture:
        migrate()                        ← sync setup (connections, schema, index)
            └── asyncio.run(async_migrate())
                    └── asyncio.gather(
                            async_producer(queue),
                            async_consumer(queue, pbar)
                        )

    Both SDKs are synchronous. Every blocking call is wrapped in
    loop.run_in_executor() so the event loop is never frozen.
    """

    def __init__(
        self,
        milvus_url: str,
        milvus_token: str,
        milvus_collection: str,
        endee_url: str,
        endee_api_key: str,
        endee_index: str,
        milvus_port: int = DEFAULT_MILVUS_PORT,
        fetch_batch_size: int = DEFAULT_FETCH_BATCH_SIZE,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        space_type: str = DEFAULT_SPACE_TYPE,
        M: int = DEFAULT_M,
        ef_construct: int = DEFAULT_EF_CONSTRUCT,
        checkpoint_file: str = CHECKPOINT_FILE,
        filter_fields: str = "",
        is_multivector: bool = False,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ):
        self.milvus_url = milvus_url
        self.milvus_token = milvus_token
        self.milvus_collection = milvus_collection
        self.milvus_port = milvus_port
        self.endee_url = endee_url
        self.endee_api_key = endee_api_key
        self.endee_index_name = endee_index
        self.fetch_batch_size = fetch_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.filter_fields = (
            set(f.strip() for f in filter_fields.split(",") if f.strip())
            if filter_fields else set()
        )
        self.space_type = space_type
        self.M = M
        self.ef_construct = ef_construct
        self.precision = Precision.FLOAT32   # overwritten by detect_vector_fields()
        self.is_multivector = is_multivector
        self.max_queue_size = max_queue_size

        self.checkpoint = MigrationCheckpoint(checkpoint_file)
        self.interrupted = False          # plain bool — safe from signal handler

        # _stop_event created inside async_migrate() (must be in running loop)
        self._stop_event: Optional[asyncio.Event] = None

        # Field detection — populated by detect_vector_fields()
        self.vector_field_info: Optional[Dict] = None
        self.id_field_name: Optional[str] = None
        self.dense_vector_field_name: Optional[str] = None
        self.dense_vector_field_type = None          # BUG 6 fix: set properly later
        self.sparse_vector_field_name: Optional[str] = None
        self.vectors_dimension: Optional[int] = None
        self.sparse_dimension: Optional[int] = None

        # Clients — populated before asyncio.run()
        self.milvus_client: Optional[MilvusClient] = None
        self.endee_client: Optional[Endee] = None
        self.endee_index = None
        
        # Statistics
        self.stats = {
            FETCHED_KEY: 0,
            UPSERTED_KEY: 0,
            FAILED_KEY: 0,
            BATCHES_PROCESSED_KEY: 0,
            RECORDS_WITH_SPARSE_KEY: 0,
            RECORDS_WITHOUT_SPARSE_KEY: 0,
            START_TIME_KEY: None
        }

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Signal handler ──────────────────────────────────────────────
    def _signal_handler(self, signum, frame):
        logger.warning("\n" + "=" * 80)
        logger.warning("Received shutdown signal. Saving progress and stopping...")
        logger.warning("=" * 80)
        self.interrupted = True

    # ══════════════════════════════════════════════════════════════
    # SYNC SETUP — all called in migrate() BEFORE asyncio.run()
    # ══════════════════════════════════════════════════════════════
    def connect_milvus(self):
        logger.info("Connecting to Milvus...")
        try:
            uri = self.milvus_url
            if not uri.startswith(("http://", "https://", "tcp://", "unix://")):
                if uri.startswith("localhost") or uri.replace(".", "").replace(":", "").isdigit():
                    uri = f"http://{uri}:{self.milvus_port}"
                    logger.info(f"Added protocol to URI: {uri}")
            self.milvus_client = MilvusClient(uri=uri, token=self.milvus_token)
            self.endee_client.list_indexes()
            logger.info("✓ Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")

    def connect_endee(self):
        logger.info("Connecting to Endee...")
        self.endee_client = Endee(token=self.endee_api_key)
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, ENDEE_V1_API)
            self.endee_client.set_base_url(url)
            logger.info(f"Set Endee base URL: {url}")
        logger.info(f"Indexes: {self.endee_client.list_indexes()}")
        logger.info("✓ Connected to Endee")

    def detect_sparse_dimension(self) -> int:
        """Sample records to find maximum sparse index, add 10% buffer."""
        if not self.sparse_vector_field_name:
            return DEFAULT_SPARSE_DIMENSION_FALLBACK
        
        logger.info("\nDetecting sparse dimension from data...")
        
        max_index = 0
        try:
            sample = self.milvus_client.query(
                collection_name=self.milvus_collection,
                filter="",
                output_fields=["*"],
                limit=min(1000, self.fetch_batch_size),
                offset=0,
            )
            for record in sample:
                sparse_data = record.get(self.sparse_vector_field_name, {})
                if sparse_data and isinstance(sparse_data, dict):
                    indices = sparse_data.keys()
                    if indices:
                        max_index = max(max_index, max(int(i) for i in indices))
            sparse_dim = int(max_index * 1.1) + 100
            logger.info(f"  Max sparse index found : {max_index}")
            logger.info(f"  Sparse dimension set to: {sparse_dim}")
            return sparse_dim
        except Exception as e:
            logger.warning(f"Could not detect sparse dimension: {e}. Using default 30000.")
            return 30000

    def decode_vector(self, raw_vector, field_type) -> List[float]:
        """Decode Milvus vector bytes to float list."""
        logger.debug(
            f"decode_vector: type={field_type}, raw={type(raw_vector)}, "
            f"preview={str(raw_vector)[:80]}"
        )
        # Unwrap list wrapper if present
        if isinstance(raw_vector, list):
            if len(raw_vector) == 1 and isinstance(raw_vector[0], bytes):
                raw_bytes = raw_vector[0]
            elif raw_vector and isinstance(raw_vector[0], (int, float)):
                return raw_vector   # already float list
            else:
                raw_bytes = raw_vector[0] if raw_vector else b""
        elif isinstance(raw_vector, bytes):
            raw_bytes = raw_vector
        else:
            return raw_vector   # already usable (e.g. plain list from newer SDK)

        if field_type == DataType.FLOAT16_VECTOR:
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).tolist()
        elif field_type == DataType.BFLOAT16_VECTOR:
            raise ValueError(
                "BFLOAT16_VECTOR is not supported. "
                "Convert to FLOAT32 or FLOAT16 before migrating."
            )
        elif field_type == DataType.FLOAT_VECTOR:
            return np.frombuffer(raw_bytes, dtype=np.float32).tolist()
        else:
            logger.warning(f"Unknown field type {field_type}, attempting float16 decode")
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).tolist()

    def detect_vector_fields(self) -> Dict:
        """
        Auto-detect field names, types, and dimensions from Milvus schema.
        BUG 6 fix: sets self.dense_vector_field_type properly.
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"Detecting fields in collection: {self.milvus_collection}")
        logger.info("=" * 80 + "\n")

        desc = self.milvus_client.describe_collection(self.milvus_collection)
        logger.info(f"Schema: {desc}")

        dense_vector_fields, sparse_vector_fields, other_fields = [], [], []
        id_field = None

        logger.info("FIELDS DETECTED:")
        logger.info("-" * 80)

        for field in desc.get("fields", []):
            name = field.get("name")
            ftype = field.get("type")
            is_primary = field.get("is_primary", False)

            if is_primary:
                id_field = {"name": name, "type": ftype}
                self.id_field_name = name
                logger.info(f"✓ ID Field (Primary Key): '{name}' [{ftype}]")

            elif ftype in [DataType.BFLOAT16_VECTOR, "BFLOAT16_VECTOR"]:
                raise ValueError(
                    f"Unsupported vector type BFLOAT16_VECTOR in field '{name}'. "
                    "Convert to FLOAT32 or FLOAT16 before migrating."
                )

            elif ftype in [
                "FLOAT_VECTOR", "FLOAT16_VECTOR", "BINARY_VECTOR",
                DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BINARY_VECTOR,
            ]:
                index_info = self.milvus_client.describe_index(self.milvus_collection, name)
                params = field.get("params", {})
                dim = params.get("dim") or field.get("dim")
                precision = (
                    MILVUS_DTYPE_TO_ENDEE_PRECISION.get(ftype)
                    or MILVUS_STR_TO_ENDEE_PRECISION.get(ftype)
                    or Precision.FLOAT32
                )
                # Read HNSW params from index
                self.ef_construct = index_info.get("params", {}).get("efConstruction", self.ef_construct)
                self.M = index_info.get("params", {}).get("M", self.M)

                dense_vector_fields.append(
                    {"name": name, "type": ftype, "dimension": dim, "precision": precision}
                )
                if self.dense_vector_field_name is None:
                    self.dense_vector_field_name = name
                    self.dense_vector_field_type = ftype   # BUG 6 FIX
                    self.vectors_dimension = dim
                    self.precision = precision
                logger.info(f"✓ Dense Vector Field: '{name}' [{ftype}, dim={dim}]")

            elif ftype in ["SPARSE_FLOAT_VECTOR", DataType.SPARSE_FLOAT_VECTOR]:
                sparse_vector_fields.append({"name": name, "type": ftype})
                if self.sparse_vector_field_name is None:
                    self.sparse_vector_field_name = name
                logger.info(f"✓ Sparse Vector Field: '{name}' [{ftype}]")

            else:
                other_fields.append({"name": name, "type": ftype})
                logger.info(f"  • Metadata Field: '{name}' [{ftype}]")

        logger.info("-" * 80)
        is_hybrid = len(sparse_vector_fields) > 0
        logger.info(f"\nCollection Type: {'HYBRID (Dense + Sparse)' if is_hybrid else 'DENSE ONLY'}")
        logger.info(f"  Primary Key       : {id_field['name'] if id_field else 'NOT FOUND'}")
        logger.info(f"  Dense fields      : {[f['name'] for f in dense_vector_fields]}")
        logger.info(f"  Sparse fields     : {[f['name'] for f in sparse_vector_fields]}")
        logger.info(f"  Metadata fields   : {[f['name'] for f in other_fields]}")
        logger.info("\n" + "=" * 80 + "\n")

        self.vector_field_info = {
            "id_field": id_field,
            "dense_vector_fields": dense_vector_fields,
            "sparse_vector_fields": sparse_vector_fields,
            "other_fields_meta": other_fields,
            "is_hybrid": is_hybrid,
        }

        if not self.id_field_name:
            raise ValueError("No primary key field found in collection")
        if not self.dense_vector_field_name:
            raise ValueError("No dense vector field found in collection")

        return self.vector_field_info

    def get_or_create_endee_index(self):
        """Get or create Endee index (hybrid or dense-only)."""
        if not self.vectors_dimension:
            raise ValueError("Vector dimension not detected. Run detect_vector_fields() first.")

        is_hybrid = self.vector_field_info.get("is_hybrid", False)

        try:
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Index already exists: {self.endee_index_name}")
        except NotFoundException:
            if is_hybrid:
                self.sparse_dimension = self.detect_sparse_dimension()
                logger.info(f"Creating HYBRID index: {self.endee_index_name}")
                logger.info(f"  Dense dimension : {self.vectors_dimension}")
                logger.info(f"  Sparse dimension: {self.sparse_dimension}")
                logger.info(f"  Space type      : {self.space_type}")
                logger.info(f"  M               : {self.M}")
                logger.info(f"  ef_construct    : {self.ef_construct}")
                logger.info(f"  Precision       : {self.precision}")
                self.endee_client.create_index(
                    name=self.endee_index_name,
                    dimension=self.vectors_dimension,
                    space_type=self.space_type,
                    sparse_model="default",
                    M=self.M,
                    ef_con=self.ef_construct,
                    precision=self.precision,
                )
                logger.info(f"✓ Created HYBRID index: {self.endee_index_name}")
            else:
                logger.info(f"Creating DENSE index: {self.endee_index_name}")
                logger.info(f"  Dimension   : {self.vectors_dimension}")
                logger.info(f"  Space type  : {self.space_type}")
                logger.info(f"  M           : {self.M}")
                logger.info(f"  ef_construct: {self.ef_construct}")
                logger.info(f"  Precision   : {self.precision}")
                self.endee_client.create_index(
                    name=self.endee_index_name,
                    dimension=self.vectors_dimension,
                    space_type=self.space_type,
                    M=self.M,
                    ef_con=self.ef_construct,
                    precision=self.precision,
                )
                logger.info(f"✓ Created DENSE index: {self.endee_index_name}")

            self.endee_index = self.endee_client.get_index(self.endee_index_name)

    # ══════════════════════════════════════════════════════════════
    # ASYNC FETCH
    # ══════════════════════════════════════════════════════════════
    async def async_fetch_batch(self, offset: int) -> Tuple[List, int]:
        """
        Fetch one batch from Milvus using integer offset pagination.

        BUG 5 fix: milvus_client.query() returns a plain list, not a tuple.
        We compute next_offset = offset + len(results) ourselves.

        run_in_executor keeps the event loop free during the HTTP call.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.milvus_client.query(
                collection_name=self.milvus_collection,
                filter="",
                output_fields=["*"],
                limit=self.fetch_batch_size,
                offset=offset,
            ),
        )
        next_offset = offset + len(results) if results else offset
        return results or [], next_offset

    # ══════════════════════════════════════════════════════════════
    # CONVERT  (CPU — safe on event loop thread for typical batch sizes)
    # ══════════════════════════════════════════════════════════════
    def convert_records(self, milvus_records: List) -> List[Dict]:
        """Convert Milvus records to Endee hybrid format."""
        if not milvus_records:
            return []

        payload_field_names = set(
            f.get("name")
            for f in self.vector_field_info.get("other_fields_meta", [])
        )

        # Validate filter fields exist
        for field in self.filter_fields:
            if field not in payload_field_names:
                raise ValueError(f"Filter field '{field}' not found in schema metadata fields")

        records = []
        for record in milvus_records:
            try:
                record_id = str(record.get(self.id_field_name, ""))
                raw_dense = record.get(self.dense_vector_field_name, [])
                dense_vector = self.decode_vector(raw_dense, self.dense_vector_field_type)

                # Filter vs meta split
                if self.filter_fields:
                    filter_data = {k: v for k, v in record.items() if k in self.filter_fields}
                    meta_data = {
                        k: v for k, v in record.items()
                        if k not in self.filter_fields and k in payload_field_names
                    }
                else:
                    filter_data = {}
                    meta_data = {k: v for k, v in record.items() if k in payload_field_names}

                endee_record = {
                    ENDEE_ID_KEY: record_id,
                    ENDEE_VECTOR_KEY: dense_vector,
                    ENDEE_FILTER_KEY: filter_data,
                    ENDEE_META_KEY: {}  # All other fields go here
                }

                # Sparse vector handling
                if self.sparse_vector_field_name:
                    sparse_data = record.get(self.sparse_vector_field_name, {})
                    if sparse_data and isinstance(sparse_data, dict):
                        sorted_items = sorted(sparse_data.items())
                        indices = [int(idx) for idx, _ in sorted_items]
                        values = [float(val) for _, val in sorted_items]
                        if indices:
                            endee_record[ENDEE_SPARSE_INDICES_KEY] = indices
                            endee_record[ENDEE_SPARSE_VALUES_KEY] = values
                            self.stats[RECORDS_WITH_SPARSE_KEY] += 1
                        else:
                            self.stats[RECORDS_WITHOUT_SPARSE_KEY] += 1
                    else:
                        self.stats["records_without_sparse"] += 1
                records.append(endee_record)

            except Exception as e:
                logger.error(
                    f"Error converting record "
                    f"{record.get(self.id_field_name, 'unknown')}: {e}"
                )
                logger.error(traceback.format_exc())
                continue   # skip bad record, keep migrating

        return records

    # ══════════════════════════════════════════════════════════════
    # ASYNC UPSERT — single chunk
    # ══════════════════════════════════════════════════════════════
    async def async_upsert_chunk(self, chunk: List[Dict]) -> None:
        """
        BUG 2 fix: was calling self.endee_index.upsert() directly on the
        event loop thread (blocking). Now wrapped in run_in_executor.
        RAISES on failure so gather(return_exceptions=True) detects it.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.endee_index.upsert(chunk)
        )
        if not result:
            raise RuntimeError(f"Endee upsert returned falsy for chunk of {len(chunk)}")
        logger.debug(f"  Upserted chunk of {len(chunk)} records")

    # ══════════════════════════════════════════════════════════════
    # ASYNC UPSERT — full batch with parallelism + retry
    # ══════════════════════════════════════════════════════════════
    async def async_upsert_records(self, records: List[Dict]) -> bool:
        """
        BUG 2 fix: original was a plain sync for-loop. Now:
          - splits into chunks
          - upserts all chunks in parallel via gather()
          - retries each failed chunk with exponential backoff
        """
        if self.interrupted:
            return False

        chunks = [
            records[i: i + self.upsert_batch_size]
            for i in range(0, len(records), self.upsert_batch_size)
        ]

        # Phase 1: all chunks in parallel
        results = await asyncio.gather(
            *[self.async_upsert_chunk(c) for c in chunks],
            return_exceptions=True,
        )

        # Phase 2: collect failures
        failed_chunks = [
            chunks[i] for i, r in enumerate(results) if isinstance(r, Exception)
        ]
        if failed_chunks:
            logger.warning(f"  {len(failed_chunks)}/{len(chunks)} chunks failed — retrying...")

        # Phase 3: retry with exponential backoff (1s, 2s, 4s)
        while failed_chunks:
            chunk = failed_chunks.pop(0)
            succeeded = False
            for attempt in range(3):
                try:
                    await self.async_upsert_chunk(chunk)
                    succeeded = True
                    logger.info(f"  Retry {attempt + 1} succeeded ({len(chunk)} records)")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"  Retry {attempt + 1}/3 failed: {e}. Waiting {wait}s...")
                    await asyncio.sleep(wait)   # non-blocking — yields to event loop
            if not succeeded:
                logger.error(f"  Chunk of {len(chunk)} exhausted 3 retries")
                return False

        return True

    # ══════════════════════════════════════════════════════════════
    # PRODUCER
    # ══════════════════════════════════════════════════════════════
    async def async_producer(self, queue: asyncio.Queue):
        """
        Fetch batches from Milvus and put them into the queue.

        Two stop checks:
          (A) while condition — top of every iteration
          (B) post-put check — closes race window where consumer fails
              WHILE producer is suspended inside await queue.put()

        BUG 4 fix: on exception, set _stop_event BEFORE put(None).
        """
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        logger.info("PRODUCER: started")

        try:
            while not self.interrupted and not self._stop_event.is_set():
                logger.info(f"PRODUCER: fetching batch {batch_number} (offset={offset})")

                milvus_records, next_offset = await self.async_fetch_batch(offset)

                # End of collection
                if not milvus_records:
                    logger.info("PRODUCER: no more data — sending sentinel")
                    await queue.put(None)
                    break

                # Convert
                records = self.convert_records(milvus_records)
                self.stats["fetched"] += len(records)

                # Interrupt check (Ctrl+C arrived during fetch)
                if self.interrupted:
                    logger.info("PRODUCER: interrupted — sending sentinel")
                    await queue.put(None)
                    break

                # Put into queue — suspends here if queue is full (backpressure)
                await queue.put({
                    "batch_number": batch_number,
                    "records": records,
                    "next_offset": next_offset,
                })

                # Post-put stop check — consumer may have failed while we waited
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: stop event after queue.put() — exiting")
                    break

                # End of collection check (partial batch = last batch)
                if len(milvus_records) < self.fetch_batch_size:
                    logger.info(
                        f"PRODUCER: partial batch ({len(milvus_records)} < "
                        f"{self.fetch_batch_size}) — sending sentinel"
                    )
                    await queue.put(None)
                    break

                # Advance to next batch
                offset = next_offset
                batch_number += 1

        except Exception as e:
            logger.error(f"PRODUCER: exception — {e}")
            logger.error(traceback.format_exc())
            # BUG 4 fix: signal first, then sentinel
            self._stop_event.set()
            await queue.put(None)

        logger.info("PRODUCER: finished")

    # ══════════════════════════════════════════════════════════════
    # CONSUMER
    # ══════════════════════════════════════════════════════════════
    async def async_consumer(self, queue: asyncio.Queue, pbar: tqdm):
        """
        Get batches from queue and upsert to Endee.

        task_done() ORDER on failure — prevents deadlock (BUG 3 fix):

          WRONG:
            self._stop_event.set()
            break                   ← exits without task_done()
            # producer blocked in queue.put() → DEADLOCK forever

          CORRECT:
            self._stop_event.set()  # 1. signal producer to stop
            queue.task_done()       # 2. unblock producer from queue.put()
            break                   # 3. consumer exits cleanly

        Same order in the exception handler.
        """
        logger.info("CONSUMER: started")

        while not self.interrupted:
            try:
                batch = await queue.get()

                # Sentinel — producer finished (success or error)
                if batch is None:
                    queue.task_done()
                    logger.info("CONSUMER: received sentinel — exiting")
                    break

                batch_number = batch["batch_number"]
                records = batch["records"]
                next_offset = batch["next_offset"]
                records_count = len(records)

                logger.info(f"CONSUMER: upserting batch {batch_number} ({records_count} records)")
                t0 = time.time()
                success = await self.async_upsert_records(records)
                elapsed = time.time() - t0

                if success:
                    self.checkpoint.update(batch_number, records_count, next_offset)
                    self.stats["upserted"] += records_count
                    self.stats["batches_processed"] += 1
                    pbar.update(records_count)
                    throughput = records_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"CONSUMER: batch {batch_number} ✓ — "
                        f"{records_count} records in {elapsed:.2f}s "
                        f"({throughput:.0f} rec/s)"
                    )
                    queue.task_done()
                else:
                    self.stats["failed"] += records_count
                    logger.error(f"CONSUMER: batch {batch_number} ✗ — failed after retries")
                    self._stop_event.set()   # 1. signal producer   ← BUG 3 fix
                    queue.task_done()        # 2. unblock producer
                    break                    # 3. exit

            except Exception as e:
                logger.error(f"CONSUMER: exception — {e}")
                logger.error(traceback.format_exc())
                self._stop_event.set()       # 1. signal producer   ← BUG 3 fix
                queue.task_done()            # 2. unblock producer
                break                        # 3. exit

        logger.info("CONSUMER: finished")

    # ══════════════════════════════════════════════════════════════
    # ASYNC ORCHESTRATOR
    # ══════════════════════════════════════════════════════════════
    async def async_migrate(self):
        """
        Create queue + stop event, run producer and consumer concurrently.
        _stop_event created HERE — must be inside a running event loop.
        """
        logger.info("=" * 80)
        logger.info("ASYNC MILVUS → ENDEE MIGRATION  (producer-consumer)")
        logger.info("=" * 80)
        logger.info(f"Source     : {self.milvus_collection} @ {self.milvus_url}")
        logger.info(f"Target     : {self.endee_index_name}")
        logger.info(f"Fetch size : {self.fetch_batch_size}")
        logger.info(f"Upsert size: {self.upsert_batch_size}")
        logger.info(f"Queue size : {self.max_queue_size}")
        logger.info("=" * 80)

        if self.checkpoint.get_processed_count() > 0:
            logger.info("RESUMING from checkpoint:")
            logger.info(f"  Already processed : {self.checkpoint.get_processed_count()} records")
            logger.info(f"  Starting at batch : {self.checkpoint.get_batch_number() + 1}")
            logger.info(f"  Offset            : {self.checkpoint.get_last_offset()}")
            logger.info("=" * 80)

        # Bounded queue — producer suspends when full (backpressure)
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)

        # Must be created inside running event loop
        self._stop_event = asyncio.Event()

        with tqdm(
            desc="Migrating records",
            unit="records",
            initial=self.checkpoint.get_processed_count(),
        ) as pbar:
            await asyncio.gather(
                self.async_producer(queue),
                self.async_consumer(queue, pbar),
            )

        logger.info("=" * 80)
        logger.info("ASYNC MIGRATION COMPLETED")
        logger.info("=" * 80)

    # ══════════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════
    def migrate(self):
        """
        BUG 1 fix: original migrate() ran the OLD SYNC LOOP.
        Now: sync setup first, then asyncio.run(async_migrate()).
        All blocking SDK calls happen here (before the event loop starts)
        so they never freeze the event loop.
        """
        if self.is_multivector:
            raise ValueError(
                "Multivector mode is not supported for Milvus → Endee hybrid migration"
            )

        self.stats["start_time"] = time.time()

        # ── Sync setup BEFORE event loop ───────────────────────────
        self.connect_milvus()
        self.connect_endee()
        self.detect_vector_fields()
        self.get_or_create_endee_index()

        if self.endee_index is None:
            raise RuntimeError("Endee index not initialized!")
        logger.info("✓ Endee index ready")

        # ── Async pipeline ─────────────────────────────────────────
        asyncio.run(self.async_migrate())

        self._print_final_report()

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    def _print_final_report(self):
        """Print migration summary"""
        duration = time.time() - self.stats[START_TIME_KEY]

        logger.info("\n" + "="*80)
        if self.interrupted:
            logger.warning("MIGRATION INTERRUPTED")
        elif self.stats[FAILED_KEY] > 0:
            logger.warning("MIGRATION COMPLETED WITH ERRORS")
        else:
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total records processed: {self.checkpoint.get_processed_count()}")
        logger.info(f"Records fetched this run: {self.stats[FETCHED_KEY]}")
        logger.info(f"Records upserted this run: {self.stats[UPSERTED_KEY]}")

        if self.vector_field_info and self.vector_field_info.get('is_hybrid'):
            logger.info(f"Records with sparse vectors: {self.stats[RECORDS_WITH_SPARSE_KEY]}")
            logger.info(f"Records without sparse vectors: {self.stats[RECORDS_WITHOUT_SPARSE_KEY]}")

        logger.info(f"Records failed: {self.stats[FAILED_KEY]}")
        logger.info(f"Batches processed: {self.stats[BATCHES_PROCESSED_KEY]}")

        if self.stats[UPSERTED_KEY] > 0:
            rate = self.stats[UPSERTED_KEY] / duration
            logger.info(f"Throughput: {rate:.2f} records/second")
        
        logger.info("="*80)
        
        # Show field mapping used
        if self.vector_field_info:
            logger.info("Field mapping (Milvus → Endee):")
            logger.info(f"  {self.id_field_name} → id")
            logger.info(f"  {self.dense_vector_field_name} → vector")
            if self.sparse_vector_field_name:
                logger.info(f"  {self.sparse_vector_field_name} → sparse_indices + sparse_values")
            logger.info("  Other fields → filter / meta")
        logger.info("=" * 80)
        if self.interrupted:
            logger.info("Progress saved. Run again to resume from checkpoint.")
        elif self.stats[FAILED_KEY] > 0:
            logger.warning("Migration had errors. Check logs and retry.")
        else:
            logger.info("Migration successful!")
        logger.info("=" * 80)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Async Milvus → Endee migration (hybrid dense+sparse, producer-consumer)"
    )

    # Source
    parser.add_argument("--source_url",        default=os.getenv("SOURCE_URL"),         help="Milvus URI")
    parser.add_argument("--source_api_key",    default=os.getenv("SOURCE_API_KEY", ""), help="Milvus token")
    parser.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"),  help="Milvus collection name")
    parser.add_argument("--source_port",       type=int, default=int(os.getenv("SOURCE_PORT", 19530)), help="Milvus port")
    parser.add_argument("--filter_fields",     default=os.getenv("FILTER_FIELDS", ""),  help="Comma-separated filter fields")
    parser.add_argument("--is_multivector",    action="store_true",
                        default=os.getenv("IS_MULTIVECTOR", "false").lower() == "true")  # BUG 7 fix

    # Target
    parser.add_argument("--target_url",        default=os.getenv("TARGET_URL"),         help="Endee URL")
    parser.add_argument("--target_api_key",    default=os.getenv("TARGET_API_KEY", ""), help="Endee API key")
    parser.add_argument("--target_collection", default=os.getenv("TARGET_COLLECTION"),  help="Endee index name")

    # Performance
    parser.add_argument("--batch_size",     type=int, default=int(os.getenv("BATCH_SIZE",   1000)), help="Fetch batch size")
    parser.add_argument("--upsert_size",    type=int, default=int(os.getenv("UPSERT_SIZE",   100)), help="Upsert chunk size")
    parser.add_argument("--max_queue_size", type=int, default=int(os.getenv("MAX_QUEUE_SIZE",  5)), help="Queue max size")

    # Index config
    parser.add_argument("--space_type",   default=os.getenv("SPACE_TYPE", "cosine"))
    parser.add_argument("--M",            type=int, default=int(os.getenv("M",   16)))
    parser.add_argument("--ef_construct", type=int, default=int(os.getenv("EF_CONSTRUCT", 128)))

    # Resume
    parser.add_argument("--checkpoint_file", default=os.getenv("CHECKPOINT_FILE", "./migration_checkpoint.json"))
    parser.add_argument("--clear_checkpoint", action="store_true",
                        default=os.getenv("CLEAR_CHECKPOINT", "false").lower() == "true")  # BUG 7 fix

    # Misc
    parser.add_argument("--debug", action="store_true",
                        default=os.getenv("DEBUG", "false").lower() == "true")

    args = parser.parse_args()

    # if args.debug:
    #     logging.getLogger().setLevel(logging.DEBUG)

    migrator = AsyncHybridMilvusToEndeeMigrator(
        milvus_url=args.source_url,
        milvus_token=args.source_api_key,
        milvus_collection=args.source_collection,
        milvus_port=args.source_port,
        filter_fields=args.filter_fields,
        endee_url=args.target_url,
        endee_api_key=args.target_api_key,
        endee_index=args.target_collection,
        fetch_batch_size=args.batch_size,
        upsert_batch_size=args.upsert_size,
        space_type=args.space_type,
        M=args.M,
        ef_construct=args.ef_construct,
        checkpoint_file=args.checkpoint_file,
        is_multivector=args.is_multivector,
        max_queue_size=args.max_queue_size,
    )

    if args.clear_checkpoint:
        logger.info("Clearing checkpoint for fresh start...")
        migrator.checkpoint.clear()

    try:
        migrator.migrate()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()