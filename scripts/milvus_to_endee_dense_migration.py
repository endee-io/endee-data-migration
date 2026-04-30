from typing import Dict, Any, Optional
from pymilvus import MilvusClient, DataType
from endee import Endee, Precision
from tqdm import tqdm
import logging
from endee.exceptions import NotFoundException
import argparse
import json
import time
import signal
import sys
import urllib
import os
import dotenv
import numpy as np
import asyncio
import orjson
from constants import *
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MILVUS_DTYPE_TO_ENDEE_PRECISION = {
    DataType.FLOAT_VECTOR:    Precision.FLOAT32,   # 32-bit float
    DataType.FLOAT16_VECTOR:  Precision.FLOAT16,   # 16-bit half precision
    # DataType.BFLOAT16_VECTOR: Precision.FLOAT16,   # bfloat16 → closest Endee match
    DataType.BINARY_VECTOR:   Precision.BINARY2,    # binary
}
# Also handle string versions just in case
MILVUS_STR_TO_ENDEE_PRECISION = {
    'FLOAT_VECTOR':    Precision.FLOAT32,
    'FLOAT16_VECTOR':  Precision.FLOAT16,
    'BFLOAT16_VECTOR': Precision.FLOAT16,
    'BINARY_VECTOR':   Precision.BINARY2,
}


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
            with open(self.checkpoint_file, 'r') as f:
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
        """Save checkpoint to file"""
        try:
            dirpath = os.path.dirname(self.checkpoint_file)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(self.checkpoint_file, 'wb') as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
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
        return self.data.get(LAST_OFFSET_KEY, 0)

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
            LAST_OFFSET_KEY: DEFAULT_PROCESSED_COUNT,
            BATCH_NUMBER_KEY: DEFAULT_BATCH_NUMBER
        }
        self.save()


class SimpleMilvusToEndeeMigrator:
    """Simple sequential migration from Milvus (Dense) to Endee
        Async producer-consumer migration from Milvus (dense) to Endee.

        Architecture:
            asyncio.run(async_migrate())
                └── asyncio.gather(
                        async_producer(queue),   # fetches from Milvus
                        async_consumer(queue)    # upserts to Endee
                    )

        Both SDKs are synchronous, so every blocking call is wrapped in
        loop.run_in_executor() to avoid freezing the event loop.
    """
    
    def __init__(
        self,
        milvus_url: str,
        milvus_token: str,
        milvus_collection: str,
        endee_url: str,
        endee_api_key: str,
        endee_index: str,
        precision: str,
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
        self.filter_fields = set(f.strip() for f in filter_fields.split(",") if f.strip()) if filter_fields else set()
        # Collection config
        self.space_type = space_type
        self.M = M
        self.ef_construct = ef_construct
        self.precision = precision
        self.is_multivector = is_multivector
        self.checkpoint = MigrationCheckpoint(checkpoint_file)
        self.interrupted = False
        self.max_queue_size = max_queue_size
        self._stop_event = None
        # Field detection info (will be populated after connection)
        self.vector_field_info = None
        self.id_field_name = None
        self.vector_field_name = None
        self.vectors_dimension = None
        
        # Clients
        self.milvus_client = None
        self.endee_client = None
        self.endee_index = None

        self.vector_field_type = None
        
        # Statistics
        self.stats = {
            FETCHED_KEY: 0,
            UPSERTED_KEY: 0,
            FAILED_KEY: 0,
            BATCHES_PROCESSED_KEY: 0,
            START_TIME_KEY: None
        }
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"\n{'='*80}")
        logger.warning("Received shutdown signal. Saving progress and stopping...")
        logger.warning(f"{'='*80}")
        self.interrupted = True
    
    def connect_milvus(self):
        """Connect to Milvus"""
        logger.info("Connecting to Milvus...")
        try:
            # Fix URI if needed - add protocol if missing
            uri = self.milvus_url
            if not uri.startswith(('http://', 'https://', 'tcp://', 'unix://')):
                # If it's localhost or an IP without protocol, add http://
                if uri.startswith('localhost') or uri.replace('.', '').replace(':', '').isdigit():
                    uri = f"http://{uri}:{self.milvus_port}"
                    logger.info(f"Added protocol to URI: {uri}")
            
            self.milvus_client = MilvusClient(uri=uri, token=self.milvus_token)
            logger.info("✓ Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
    
    def connect_endee(self):
        """Connect to Endee"""
        logger.info("Connecting to Endee...")
        
        # Initialize Endee client with API key
        self.endee_client = Endee(token=self.endee_api_key)

        
        # # Set custom base URL if provided
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, ENDEE_V1_API)
            self.endee_client.set_base_url(url)
            logger.info(f"Set Endee base URL: {url}")

        logger.info(f"{self.endee_client.list_indexes()}")
        logger.info("✓ Connected to Endee")
    
    def decode_vector(self, raw_vector, field_type):
        """Decode Milvus vector bytes to float list for Endee"""
        
        logger.debug(f"decode_vector called: type={field_type}, raw type={type(raw_vector)}, value preview={str(raw_vector)[:80]}")
        
        # Unwrap list wrapper if present e.g. [b'\x99...'] → b'\x99...'
        if isinstance(raw_vector, list):
            if len(raw_vector) == 1 and isinstance(raw_vector[0], bytes):
                raw_bytes = raw_vector[0]
            elif len(raw_vector) > 0 and isinstance(raw_vector[0], (int, float)):
                return raw_vector  # already float list, no conversion needed
            else:
                raw_bytes = raw_vector[0] if raw_vector else b''
        elif isinstance(raw_vector, bytes):
            raw_bytes = raw_vector
        else:
            return raw_vector  # already usable
        
        # Now decode bytes based on field type
        if field_type == DataType.FLOAT16_VECTOR:
            arr = np.frombuffer(raw_bytes, dtype=np.float16)
            return arr.astype(np.float32).tolist()
        
        elif field_type == DataType.BFLOAT16_VECTOR:
            raise ValueError(
                "BFLOAT16_VECTOR is not supported. "
                "Convert to FLOAT32 or FLOAT16 before migrating."
            )
        
        elif field_type == DataType.FLOAT_VECTOR:
            # FLOAT_VECTOR shouldn't be bytes but handle just in case
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            return arr.tolist()
        
        # Fallback - try float16 decode
        logger.warning(f"Unknown field type {field_type}, attempting float16 decode")
        arr = np.frombuffer(raw_bytes, dtype=np.float16)
        return arr.astype(np.float32).tolist()

    def detect_vector_field(self):
        """
        Auto-detect vector field name, ID field name, and dimension
        Works regardless of what the fields are named
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Detecting fields in collection: {self.milvus_collection}")
        logger.info(f"{'='*80}\n")
        
        # Get collection schema
        desc = self.milvus_client.describe_collection(self.milvus_collection)

        # Storage for detected fields
        vector_fields = []
        id_field = None
        other_fields = []
        
        logger.info("FIELDS DETECTED:")
        logger.info("-" * 80)
        
        for field in desc.get('fields', []):
            field_name = field.get('name')
            field_type = field.get('type')
            is_primary = field.get('is_primary', False)
            
            # Detect ID field
            if is_primary:
                id_field = {
                    'name': field_name,
                    'type': field_type
                }
                self.id_field_name = field_name
                logger.info(f"✓ ID Field (Primary Key): '{field_name}' [{field_type}]")

            elif field_type in [DataType.BFLOAT16_VECTOR,'BFLOAT16_VECTOR']:
                raise ValueError(
                    f"Unsupported vector type: BFLOAT16_VECTOR in field '{field_name}'. "
                    f"Endee does not support BFLOAT16 precision. "
                    f"Please convert your vectors to FLOAT32 or FLOAT16 before migrating."
                )

            # Detect vector fields
            elif field_type in ['FLOAT_VECTOR', 'FLOAT16_VECTOR', 'BINARY_VECTOR',
                    DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, 
                     DataType.BINARY_VECTOR]:
                index_info = self.milvus_client.describe_index(self.milvus_collection, field_name) or {}
                print(index_info)
                self.ef_construct = index_info.get('params', {}).get('efConstruction', self.ef_construct)
                self.M = index_info.get('params', {}).get('M', self.M)
                params = field.get('params', {})
                dim = params.get('dim') or field.get('dim')
                
                # Detect precision from field type
                precision = (
                    MILVUS_DTYPE_TO_ENDEE_PRECISION.get(field_type) or
                    MILVUS_STR_TO_ENDEE_PRECISION.get(field_type) or
                    Precision.FLOAT32  # fallback
                )
                
                vector_fields.append({
                    'name': field_name,
                    'type': field_type,
                    'dimension': dim,
                    'precision': precision   # ← store per vector field
                })
                
                # Use the first vector field found
                if self.vector_field_name is None:
                    self.vector_field_name = field_name
                    self.vectors_dimension = dim
                    self.precision = precision  # ← set on self
                
                logger.info(f"✓ Vector Field: '{field_name}' [{field_type}, dim={dim}, precision={precision}]")
            
            # Other fields
            else:
                other_fields.append({
                    'name': field_name,
                    'type': field_type
                })
                logger.info(f"  • Metadata Field: '{field_name}' [{field_type}]")
        
        logger.info("-" * 80)
        logger.info(f"\nSUMMARY:")
        logger.info(f"  Primary Key: {id_field['name'] if id_field else 'NOT FOUND'}")
        logger.info(f"  Vector Fields: {len(vector_fields)}")
        
        if vector_fields:
            for vf in vector_fields:
                logger.info(f"    - {vf['name']} (dim={vf['dimension']})")
        
        logger.info(f"  Metadata Fields: {len(other_fields)}")
        if other_fields:
            for of in other_fields:
                logger.info(f"    - {of['name']}")
        
        logger.info(f"\n{'='*80}\n")
        
        self.vector_field_info = {
            'id_field_meta': id_field,
            'vector_field_meta': vector_fields,
            'other_fields_meta': other_fields
        }

        if not self.id_field_name:
            raise ValueError("No primary key field found in collection")
        if not self.vector_field_name:
            raise ValueError("No vector field found in collection")
        
        return self.vector_field_info
    
    def get_or_create_endee_index(self):
        """Get or create Endee dense vector index"""
        if not self.vectors_dimension:
            raise ValueError("Vector dimension not detected. Run detect_vector_field() first.")
        
        try:
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Index already exists: {self.endee_index_name}")
        except NotFoundException:
            logger.info(f"Creating dense vector index: {self.endee_index_name}")
            logger.info(f"  - Dimension: {self.vectors_dimension}")
            logger.info(f"  - Space type: {self.space_type}")
            logger.info(f"  - M: {self.M}")
            logger.info(f"  - ef_construct: {self.ef_construct}")
            
            self.endee_client.create_index(
                name=self.endee_index_name,
                dimension=self.vectors_dimension,
                space_type=self.space_type,
                M=self.M,
                ef_con=self.ef_construct,
                precision=self.precision
            )
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Created dense vector index: {self.endee_index_name}")
    
    # async def async_fetch_batch(self, offset: int) -> list:
    #     """Fetch a single batch from Milvus"""
    #     loop = asyncio.get_running_loop()
    #     results = await loop.run_in_executor(None, self.milvus_client.query(
    #         collection_name=self.milvus_collection,
    #         filter="",
    #         output_fields=["*"],
    #         limit=self.fetch_batch_size,
    #         offset=offset
    #     ))
    #     return results
    # async def async_fetch_batch(self, offset: int) -> list[Dict]:
    #     """
    #     Fetch one batch from Milvus.

    #     milvus_client.query() is a blocking synchronous call.
    #     Wrapping it in run_in_executor frees the event loop during
    #     the HTTP round-trip so the consumer can upsert concurrently.

    #     FIX: Without run_in_executor this call would freeze the event
    #     loop for the entire HTTP duration — zero concurrency.
    #     """
    #     loop = asyncio.get_running_loop()
    #     try:
    #         results = await loop.run_in_executor(
    #             None,
    #             lambda: self.milvus_client.query(
    #                 collection_name=self.milvus_collection,
    #                 filter="",
    #                 output_fields=["*"],
    #                 limit=self.fetch_batch_size,
    #                 offset=offset,
    #             ),
    #         )
    #         return results or []
    #     except Exception as e:
    #         logger.error(f"Error fetching batch at offset {offset}: {e}")
    #         raise  # re-raise so producer can handle it

    
    def convert_records(self, milvus_records) -> list:
        records = []

        # CHECK IF FILTER FIELDS ARE PRESENT IN THE PAYLOAD
        if self.vector_field_info:
            payload_field_names = set(i.get('name') for i in self.vector_field_info.get('other_fields_meta',{}))
            for field in self.filter_fields:
                if field not in payload_field_names:
                    raise ValueError(f"Field {field} not found in payload")
        for record in milvus_records:
            record_id = str(record.get(self.vector_field_info.get('id_field_meta').get('name')))
            raw_vector = record.get(self.vector_field_info.get('vector_field_meta')[0].get('name'))
            # ← this must be called
            # logger.debug(f"vector_field_type={self.vector_field_type}, raw_vector type={type(raw_vector)}")
            vector = self.decode_vector(raw_vector, self.vector_field_type)

            if self.filter_fields:
                filter_data = {key: value for key, value in record.items() if key in self.filter_fields}
                meta_data = {key: value for key, value in record.items() if key not in self.filter_fields and key in payload_field_names}
            else:
                filter_data = {}
                meta_data = {key:value for key, value in record.items() if key in payload_field_names}
            endee_record = {
                ENDEE_ID_KEY: record_id,
                ENDEE_VECTOR_KEY: vector,
                ENDEE_FILTER_KEY: filter_data,
                ENDEE_META_KEY: meta_data
            }

            records.append(endee_record)

        return records
    
    async def async_upsert_chunk(self, chunk: list[Dict]):
        """
        Upsert a single chunk to Endee.

        FIX 1 — Must RAISE, not return False:
            asyncio.gather(..., return_exceptions=True) captures raised
            exceptions as Exception objects in the results list.
            isinstance(False, Exception) is always False, so if this
            method returned False the retry logic would never trigger.

        FIX 2 — run_in_executor:
            endee_index.upsert() is a blocking sync call.
            Without the executor it would freeze the event loop.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.endee_index.upsert(chunk),
        )
        # Endee SDK returns falsy on failure — convert to exception
        # so gather() captures it correctly.
        if not result:
            raise RuntimeError(f"Endee upsert returned falsy for chunk of {len(chunk)} records")
        logger.debug(f"  Upserted chunk: {len(chunk)} records")


    async def async_upsert_records(self, records: list) -> bool:
        """
        Split records into chunks and upsert all chunks in parallel.
        Retry each failed chunk up to 3 times with exponential backoff.
        Returns True on full success, False if any chunk exhausts retries.
        """

        chunks = [records[i:i + self.upsert_batch_size] for i in range(0, len(records), self.upsert_batch_size)]
        
        # PAHSE 1 : UPSERT ALL CHUNKS SIMULTANOUSLY
        tasks = [self.async_upsert_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # PHASE 2 : RETRY FAILED CHUNKS
        failed_chunks = [chunks[i] for i, result in enumerate(results) if isinstance(result, Exception)]

        if failed_chunks:
            logger.warning(f"Failed to upsert {len(failed_chunks)} chunks. Retrying...")
        
        # PAHSE 3: RETRY FAILED CHUNKS WITH EXPONENTIAL BACKOFF
        while failed_chunks:
            chunk = failed_chunks.pop(0)
            succeeded = False

            for attempt in range(3):
                try:
                    await self.async_upsert_chunk(chunk)
                    succeeded = True
                    break
                except Exception as e:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"  Retry attempt {attempt + 1}/3 failed: {e}. "
                        f"Waiting {wait}s..."
                    )
                    # FIX: asyncio.sleep() — yields to event loop.
                    # time.sleep() would freeze the entire event loop.
                    await asyncio.sleep(wait)

            if not succeeded:
                logger.error(f"  Chunk of {len(chunk)} records failed after 3 retries")
                return False

        return True
    
    async def async_producer(self, queue: asyncio.Queue):
        """
        Fetch batches from Milvus using QueryIterator (no offset limit).

        QueryIterator handles pagination internally — no 16384 offset cap.
        iterator.next() is synchronous, wrapped in run_in_executor to
        avoid freezing the event loop during the network call.
        """
        loop = asyncio.get_running_loop()
        batch_number = self.checkpoint.get_batch_number()
        processed_so_far = self.checkpoint.get_processed_count()

        logger.info("PRODUCER: Creating Milvus query iterator")

        # CREATE ITERATOR ONCE — it pages through all records internally
        try:
            iterator = await loop.run_in_executor(
                None,
                lambda: self.milvus_client.query_iterator(
                    collection_name=self.milvus_collection,
                    filter="",
                    output_fields=["*"],
                    batch_size=self.fetch_batch_size,
                ),
            )
        except Exception as e:
            logger.error(f"PRODUCER: Failed to create iterator: {e}")
            await queue.put(None)
            return

        # SKIP ALREADY-PROCESSED RECORDS FROM CHECKPOINT
        if processed_so_far > 0:
            skipped = 0
            logger.info(f"PRODUCER: Skipping {processed_so_far} already-processed records (checkpoint)")
            while skipped < processed_so_far:
                batch = await loop.run_in_executor(None, iterator.next)
                if not batch:
                    logger.warning("PRODUCER: Ran out of records while skipping checkpoint — already fully migrated?")
                    await loop.run_in_executor(None, iterator.close)
                    await queue.put(None)
                    return
                skipped += len(batch)
            logger.info(f"PRODUCER: Skipped {skipped} records, resuming from batch {batch_number}")

        logger.info("PRODUCER STARTED FETCHING FROM MILVUS")

        while not self.interrupted and not self._stop_event.is_set():
            try:
                logger.info(f"FETCHING BATCH FROM MILVUS {batch_number}")
                milvus_result = await loop.run_in_executor(None, iterator.next)

                # EMPTY RESULT = END OF COLLECTION
                if not milvus_result:
                    logger.info("PRODUCER: No more data to fetch")
                    await queue.put(None)
                    break

                # CONVERT TO ENDEE FORMAT
                records = self.convert_records(milvus_result)
                records_count = len(records)
                self.stats[FETCHED_KEY] += records_count
                current_offset = processed_so_far + self.stats[FETCHED_KEY]

                logger.info(f"[Batch {batch_number}] Fetched {records_count} records")

                # CHECK IF INTERRUPTED
                if self.interrupted:
                    logger.info("PRODUCER: Interrupted by user")
                    await queue.put(None)
                    break

                # PUT RECORDS INTO QUEUE (blocks if queue is full — backpressure)
                await queue.put({
                    "batch_number": batch_number,
                    "records": records,
                    "next_offset": current_offset,
                })

                # CHECK STOP EVENT AFTER QUEUE PUT (closes race window)
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: Stopped due to Stop Event")
                    break

                # TRACK QUEUE USAGE
                current_size = queue.qsize()
                if current_size >= self.max_queue_size:
                    logger.warning(f"QUEUE IS FULL. CURRENT SIZE: {current_size}")

                # Show sample record structure (first batch only)
                if batch_number == 0 and records:
                    logger.info(f"\nSample Endee record structure:")
                    sample = records[0].copy()
                    if ENDEE_VECTOR_KEY in sample and len(sample[ENDEE_VECTOR_KEY]) > 5:
                        sample[ENDEE_VECTOR_KEY] = f"[... ({len(sample[ENDEE_VECTOR_KEY])} dims)]"
                    logger.info(json.dumps(sample, indent=2, default=str))

                batch_number += 1

            except Exception as e:
                logger.error(f"[Producer] Exception: {e}")
                self._stop_event.set()
                await queue.put(None)
                break

        await loop.run_in_executor(None, iterator.close)
        logger.info("PRODUCER: Iterator closed. Finished.")

    async def async_consumer(self, queue: asyncio.Queue, pbar: tqdm):
        """
        Get batches from the queue and upsert to Endee.

        The consumer does NOT check _stop_event in its while condition.
        It is the one that SETS the flag on failure — by the time the
        flag is set the consumer is already on the break line.

        task_done() ORDER — critical to prevent deadlock:

          WRONG:
            _stop_event.set()
            break                  ← exits without calling task_done()
            # producer suspended in queue.put() → nobody calls task_done()
            # queue slot never freed → queue.put() never unblocks → DEADLOCK

          CORRECT (what we do):
            _stop_event.set()      # 1. signal producer
            queue.task_done()      # 2. unblock producer from queue.put()
            break                  # 3. consumer exits
        """
        while not self.interrupted:
            # GET BATCH FROM QUEUE
            batch = await queue.get()

            if batch is None:
                queue.task_done()
                logger.info("CONSUMER: RECEIVED END SIGNAL")
                break
            records = batch.get("records")
            records_count = len(records)
            batch_number = batch.get("batch_number")
            next_offset = batch.get("next_offset")

            logger.info(f"CONSUMER: RECEIVED BATCH {batch_number} WITH {records_count} RECORDS")
            logger.info(f"CONSUMER: UPSERTING {records_count} RECORDS TO ENDEE")

            success = await self.async_upsert_records(records)

            if success:
                # ── Update checkpoint ONLY on full success ─────────
                # If we update before success, a partial failure would
                # advance the offset and those records would never be
                # retried — silent data loss.
                self.checkpoint.update(batch_number, records_count, next_offset)
                self.stats[UPSERTED_KEY] += records_count
                self.stats[BATCHES_PROCESSED_KEY] += 1
                pbar.update(records_count)
                queue.task_done()  # unblock producer
                logger.info(
                    f"[Batch {batch_number}] ✓ Upserted {records_count} records"
                )
            else:
                # ── Failure path — ORDER MATTERS ───────────────────
                self.stats[FAILED_KEY] += records_count
                logger.error(
                    f"[Batch {batch_number}] ✗ Failed after retries — stopping migration"
                )
                self._stop_event.set()   # 1. signal producer to stop
                queue.task_done()        # 2. unblock producer from queue.put()
                break                    # 3. consumer exits

        logger.info("[Consumer] Finished")


    async def async_migrate(self):
        """
            Main async migration with Producer-Consumer pattern
            
            How it works:
            1. Producer fetches batches → puts in queue (max size = 5)
            2. Consumer takes from queue → upserts to Endee
            3. If queue is full, producer WAITS (no memory overflow!)
            4. If queue is empty, consumer WAITS (no busy waiting!)
        """
        self.stats[START_TIME_KEY] = time.time()
        if self.is_multivector:
            raise ValueError("Multivector mode is not supported for Milvus to Endee dense migration")
        
        logger.info("="*80)
        logger.info("SIMPLE SEQUENTIAL MILVUS → ENDEE MIGRATION")
        logger.info("="*80)
        logger.info(f"Source: {self.milvus_collection} @ {self.milvus_url}")
        logger.info(f"Target: {self.endee_index_name}")
        logger.info(f"Format: Dense vectors with metadata (payload)")
        logger.info(f"Fetch batch size: {self.fetch_batch_size}")
        logger.info(f"Upsert batch size: {self.upsert_batch_size}")
        logger.info("="*80)
        
        # Show checkpoint status
        if self.checkpoint.get_processed_count() > 0:
            logger.info(f"RESUMING from checkpoint:")
            logger.info(f"  - Already processed: {self.checkpoint.get_processed_count()} records")
            logger.info(f"  - Starting from batch: {self.checkpoint.get_batch_number() + 1}")
            logger.info(f"  - Offset: {self.checkpoint.get_last_offset()}")
            logger.info("="*80)
        
        # Setup connections
        self.connect_milvus()
        self.connect_endee()
        
        # Detect field names and dimensions from Milvus schema
        self.detect_vector_field()
        
        # Create/verify Endee index with detected dimension
        self.get_or_create_endee_index()
        
        # Verify index is ready
        if self.endee_index is None:
            raise RuntimeError("Endee index not initialized!")
        logger.info(f"✓ Endee dense vector index ready")
        

        
        logger.info("\nStarting migration loop...")
        logger.info("="*80)

        # CREATE BOUNDED QUEUE WITH MAX SIZE
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        logger.info(f"BOUNDED QUEUE CREATED WITH MAX SIZE OF {self.max_queue_size}")
        
        with tqdm(desc="Migrating records", unit="records", 
                 initial=self.checkpoint.get_processed_count()) as pbar:

            # CREATE STOP EVENT FOR GRACEFUL SHUTDOWN
            # THIS IS USED TO STOP THE PRODUCER AND CONSUMER
            self._stop_event = asyncio.Event()

            # START PRODUCER AND CONSUMER
            await asyncio.gather(self.async_producer(queue), self.async_consumer(queue, pbar),)

        logger.info("ASYNC MIGRATION COMPLETED")
        logger.info("="*80)

        self._print_final_report()

           
    
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
        logger.info(f"Records failed: {self.stats[FAILED_KEY]}")
        logger.info(f"Batches processed: {self.stats[BATCHES_PROCESSED_KEY]}")

        if self.stats[UPSERTED_KEY] > 0:
            rate = self.stats[UPSERTED_KEY] / duration
            logger.info(f"Throughput: {rate:.2f} records/second")
        
        logger.info("="*80)
        
        # Show field mapping used
        if self.vector_field_info:
            logger.info("\nField Mapping (Milvus → Endee):")
            logger.info(f"  {self.id_field_name} → id")
            logger.info(f"  {self.vector_field_name} → vector")
            if self.vector_field_info.get('other_fields'):
                logger.info(f"  Other fields → metadata.{{field_name}}")
            logger.info("="*80)
        
        if self.interrupted:
            logger.info("Progress saved. Run again to resume from checkpoint.")
        elif self.stats[FAILED_KEY] > 0:
            logger.warning("Migration had errors. Check logs and retry.")
        else:
            logger.info("Migration successful!")
        logger.info("="*80)

    def migrate(self):
        """Wrapper Function to run the migration"""
        if self.is_multivector:
            raise ValueError("Multivector mode is not supported for Milvus to Endee dense migration")
        asyncio.run(self.async_migrate())

def main():
    parser = argparse.ArgumentParser(
        description="Simple sequential migration from Milvus to Endee (Dense vectors)"
    )
    
    # Source arguments
    parser.add_argument("--source_url", default=os.getenv("SOURCE_URL"), help="Milvus URI")
    parser.add_argument("--source_api_key", default=os.getenv("SOURCE_API_KEY"), help="Milvus token")
    parser.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"), help="Milvus collection name")
    parser.add_argument("--source_port", type=int, default=os.getenv("SOURCE_PORT"), help="Milvus port")
    parser.add_argument("--filter_fields", default=os.getenv("FILTER_FIELDS",""), help="Filter fields")
    parser.add_argument("--is_multivector", action="store_true",
                       default=os.getenv("IS_MULTIVECTOR","false").lower() == "true",
                       help="Is multivector")

    # Target arguments
    parser.add_argument("--target_url", default=os.getenv("TARGET_URL"), help="Endee URI")
    parser.add_argument("--target_api_key", default=os.getenv("TARGET_API_KEY"), help="Endee API key")
    parser.add_argument("--target_collection", default=os.getenv("TARGET_COLLECTION"), help="Endee index name")
    
    # Performance arguments
    parser.add_argument("--batch_size", type=int, default=os.getenv("BATCH_SIZE",1000), 
                       help="Fetch batch size (default: 1000)")
    parser.add_argument("--upsert_size", type=int, default=os.getenv("UPSERT_SIZE",1000), 
                       help="Upsert batch size (default: 1000)")

    # Collection configuration
    parser.add_argument("--space_type", default=DEFAULT_SPACE_TYPE,
                       help="Distance metric (default: cosine)")
    parser.add_argument("--M", type=int, default=DEFAULT_M,
                       help="HNSW M parameter (default: 16)")
    parser.add_argument("--ef_construct", type=int, default=DEFAULT_EF_CONSTRUCT,
                       help="HNSW ef_construct parameter (default: 128)")

    # Resume arguments
    parser.add_argument("--checkpoint_file", default=CHECKPOINT_FILE,
                       help="Checkpoint file path (default: ./migration_checkpoint.json)")
    parser.add_argument("--clear_checkpoint", action="store_true", 
                       default=os.getenv("CLEAR_CHECKPOINT","false").lower() == "true",
                       help="Clear existing checkpoint and start fresh")
    
    # Debug
    parser.add_argument("--debug", action="store_true", 
                       default=os.getenv("DEBUG",False),
                       help="Enable debug logging")

    parser.add_argument("--precision", default=Precision.INT16)
    
    args = parser.parse_args()
    
    # # Set debug level if requested
    # if args.debug:
    #     logging.getLogger().setLevel(logging.DEBUG)
    
    # Create migrator
    migrator = SimpleMilvusToEndeeMigrator(
        milvus_url=args.source_url,
        milvus_token=args.source_api_key,
        milvus_collection=args.source_collection,
        milvus_port=args.source_port,
        endee_url=args.target_url,
        endee_api_key=args.target_api_key,
        endee_index=args.target_collection,
        fetch_batch_size=args.batch_size,
        upsert_batch_size=args.upsert_size,
        space_type=args.space_type,
        M=args.M,
        ef_construct=args.ef_construct,
        checkpoint_file=args.checkpoint_file,
        filter_fields=args.filter_fields,
        is_multivector=args.is_multivector,
        precision=args.precision
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        logger.info("Clearing checkpoint for fresh start...")
        migrator.checkpoint.clear()
    
    # Run migration
    try:
        migrator.migrate()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Migration failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()