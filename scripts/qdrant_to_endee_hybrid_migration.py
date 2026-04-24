from typing import Dict, Any, Optional
from qdrant_client import QdrantClient
from endee import Endee, Precision
from tqdm import tqdm
import logging
from endee.exceptions import NotFoundException
import argparse
import orjson
import time
import signal
import sys
import urllib
import os
import dotenv
import asyncio
from constants import *

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                data = orjson.loads(f.read())
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
            with open(self.checkpoint_file, 'wb') as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def update(self, batch_number: int, records_count: int, offset: Optional[Any] = None):
        """Update checkpoint after successful batch"""
        self.data[PROCESSED_COUNT_KEY] += records_count
        self.data[BATCH_NUMBER_KEY] = batch_number
        if offset is not None:
            self.data[LAST_OFFSET_KEY] = offset
        self.save()
    
    def get_last_offset(self):
        """Get the last processed offset"""
        return self.data.get(LAST_OFFSET_KEY)
    
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


class QdrantHybridToEndeeMigrator:
    """Simple sequential migration from Qdrant (Hybrid) to Endee"""
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_port: int,
        qdrant_api_key: str,
        qdrant_collection: str,
        endee_url: str,
        endee_api_key: str,
        endee_index: str,
        fetch_batch_size: int = DEFAULT_FETCH_BATCH_SIZE,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        use_https: bool = False,
        checkpoint_file: str = CHECKPOINT_FILE,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        filter_fields: str = "",
        is_multivector: bool = False
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection = qdrant_collection
        self.endee_url = endee_url
        self.endee_api_key = endee_api_key
        self.endee_index_name = endee_index
        self.fetch_batch_size = fetch_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.use_https = False
        self.filter_fields = set(f.strip() for f in filter_fields.split(",") if f.strip()) if filter_fields else set()
        self.checkpoint = MigrationCheckpoint(checkpoint_file)
        self.interrupted = False
        self.max_queue_size = max_queue_size
        self.is_multivector = is_multivector
        # Clients
        self.qdrant_client = None
        self.endee_client = None
        self.endee_index = None
        self._stop_event = None
        # Statistics
        self.stats = {
            FETCHED_KEY: 0,
            UPSERTED_KEY: 0,
            FAILED_KEY: 0,
            BATCHES_PROCESSED_KEY: 0,
            START_TIME_KEY: None
        }
        
        # Setup signal handler for graceful shutdown
        # SIGINT -> INTERRUPTED FROM KEYBOARD THAT IS CTRL+C
        # SIGTERM -> INTERRUPTED FROM TERMINAL THAT IS KILL <PID> OR DOCKER STOP
        # WHENEVER THERE IS INTERRUPTION IN MIGRATION PART ONLY IT WILL SEND SIGNAL AND SIGNAL_HANDLER WILL BE CALLED
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"\n{'='*80}")
        logger.warning("Received shutdown signal. Saving progress and stopping...")
        logger.warning(f"{'='*80}")
        self.interrupted = True
    
    def connect_qdrant(self):
        """Connect to Qdrant"""
        logger.info("Connecting to Qdrant...")
        try:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key,
                https=bool(self.use_https)
            )
            logger.info(self.qdrant_client.get_collections())
            logger.info("✓ Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
    
    def connect_endee(self):
        """Connect to Endee"""
        logger.info("Connecting to Endee...")
        
        # Initialize Endee client with API key
        self.endee_client = Endee(token=self.endee_api_key)
        
        # # Set custom base URL if provided
        logger.info(f"Endee URL: {self.endee_url}")
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, ENDEE_V1_API)
            self.endee_client.set_base_url(url)
            logger.info(f"Set Endee base URL: {url}")

        logger.info("✓ Connected to Endee")
    
    async def async_producer(self, queue: asyncio.Queue):
        """
            PRODUCER FUNCTION TO FETCH DATA FROM QDRANT

        """
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()

        logger.info("PRODUCER STARTED FETCHING FROM QDRANT")

        while not self.interrupted and not self._stop_event.is_set():
            try:
                # FETCH BATCH FROM QDRANT
                logger.debug(f"FETCHING BATCH FROM QDRANT {batch_number} WITH OFFSET {offset}")
                points_batch, next_offset = await self.async_fetch_batch(offset)
                
                # CHECK IF POINTS BATCH IS EMPTY
                if not points_batch:
                    logger.info("PRODUCER: No more data to fetch")
                    await queue.put(None)
                    break
                
                # CONVERT TO ENDEE FORMAT
                records = self.convert_records(points_batch)

                # UPDATE STATS
                self.stats[FETCHED_KEY] += len(records)

                # CHECK IF INTERRUPTED THAT IS CTRL+C OR TERMINAL KILL
                if self.interrupted:
                    logger.info("PRODUCER: Interrupted by user")
                    await queue.put(None)
                    break

                # PUT RECORDS INTO QUEUE
                await queue.put({
                    "batch_number": batch_number,
                    "records": records,
                    "next_offset": next_offset
                })
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: Stopped due to Stop Event")
                    break

                # TRACK QUEUE USAGE
                current_size = queue.qsize()
                if current_size > self.max_queue_size:
                    logger.warning(f"QUEUE IS FULL. CURRENT SIZE: {current_size}")
                
                # MOVE TO NEXT BATCH
                offset = next_offset
                batch_number  += 1

                # PUT NONE TO QUEUE IF OFFSET IS NONE
                if next_offset is None:
                    await queue.put(None)
                    break
            
            except Exception as e:
                logger.error(f"[Producer] Exception: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await queue.put(None)  # Signal error
                break
        logger.info("PRODUCER FINISHED")

    async def async_consumer(self, queue: asyncio.Queue, pbar:tqdm):
        """
        CONSUMER FUNCTION TO UPSERT DATA INTO ENDEE
        """
        logger.info("CONSUMER STARTED UPSERTING INTO ENDEE")

        while not self.interrupted:
            try:
                # GET THE DATA FROM QUEUE
                batch = await queue.get()

                # CHECK IF DATA IS NONE THAT IS CTRL+C OR TERMINAL KILL or NO DATA PRESENT
                if batch is None:
                    logger.info("CONSUMER: RECEIVED END SIGNAL")
                    queue.task_done()
                    break

                batch_number = batch.get("batch_number")
                records = batch.get("records")
                next_offset = batch.get("next_offset")
                records_count = len(records)

                # UPSERT TO ENDEE
                logger.info(f"UPSERTING BATCH {batch_number} TO ENDEE")
                start_time = time.time()

                success = await self.async_upsert_records(records)

                end_time = time.time()

                # UPSERT SUCCESSFULLY
                if success:
                    # UPDATE CHECKPOINT FIELDS
                    self.checkpoint.update(batch_number, records_count, next_offset)

                    # UPDATE STATS
                    self.stats[UPSERTED_KEY] += records_count
                    self.stats[BATCHES_PROCESSED_KEY] += 1
                    pbar.update(records_count)
                    upsert_time = end_time - start_time
                    throughput = records_count / upsert_time
                    logger.info(f"CONSUMER: SUCCESSFULLY UPSERTED {records_count} RECORDS IN {upsert_time:.2f} seconds WITH THROUGHPUT {throughput:.2f} records/second")
                    
                else:
                    self.stats[FAILED_KEY] += records_count
                    logger.error(f"Batch {batch_number}: Failed To Upsert")
                    self._stop_event.set()
                    # UNFINISHED TASKS REDUCE BY 1
                    queue.task_done()
                    break
                
                # UNFINISHED TASKS REDUCE BY 1
                queue.task_done()
            except Exception as e:
                logger.error(f"[Consumer] Exception: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self._stop_event.set()
                # UNFINISHED TASKS REDUCE BY 1
                queue.task_done()
                break
        
        logger.info("CONSUMER FINISHED")
    


    def get_qdrant_collection_info(self) -> Dict[str, Any]:
        """Get collection configuration from Qdrant (Hybrid format)"""
        logger.info(f"Getting collection info for: {self.qdrant_collection}")
        collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
        
        params =  collection_info.config.params
        vectors = params.vectors
        # =========================================================================================
        if isinstance(vectors, dict):
            vectors_map = vectors
        elif vectors is not None:
            vectors_map = {"default": vectors}
        else:
            vectors_map = {}

        # =========================================================================================
        # SET DIMENSION, AND SPACE TYPE
        logger.info(f"collection_info.config.params.vectors: {collection_info.config.params}")
        vectors_dimension = vectors['dense'].size
        qdrant_space_type = vectors['dense'].distance
        if qdrant_space_type == "Cosine":
            endee_space_type = "cosine"
        elif qdrant_space_type == "Euclid":
            endee_space_type = "l2"
        elif qdrant_space_type == "Dot":
            endee_space_type = "ip"
        else:
            raise ValueError(f"Invalid space type: {qdrant_space_type}")
        sparse_dimension = DEFAULT_SPARSE_DIMENSION
        # =========================================================================================

        # AUTO DETECT DENSE FIELD 
        self.dense_field_name = None

        if isinstance(vectors, dict):
            # Named vectors → find the dense one (has .size attribute)
            for name, config in vectors.items():
                if hasattr(config, 'size') and config.size is not None:
                    self.dense_field_name = name

                    logger.info(f"✓ Detected dense field: '{name}'")
                    break  # use first dense field found
        elif vectors is not None:
            # Single unnamed dense vector
            self.dense_field_name = "default"
            logger.info(f"✓ Single dense vector")
        
        if self.dense_field_name is None:
            raise ValueError("No dense vector field found in collection")
        # =========================================================================================
        

        # ─── AUTO DETECT SPARSE FIELD ──────────────────────────────────────────
        # Sparse vectors are in params.sparse_vectors, NOT in params.vectors
        sparse_vectors = params.sparse_vectors  # e.g. {'sparse': SparseVectorParams(...)}
        self.sparse_field_name = None

        if sparse_vectors and isinstance(sparse_vectors, dict):
            for name, config in sparse_vectors.items():
                self.sparse_field_name = name  # use first sparse field found
                logger.info(f"✓ Detected sparse field: '{name}'")
                break
        
        if self.sparse_field_name:
            logger.info(f"Collection type: HYBRID (dense='{self.dense_field_name}', sparse='{self.sparse_field_name}')")
        else:
            logger.info(f"Collection type: DENSE ONLY (dense='{self.dense_field_name}')")
        # =========================================================================================
        
        
        # Get HNSW config
        M = collection_info.config.hnsw_config.m
        ef_construct = collection_info.config.hnsw_config.ef_construct

        if collection_info.config.quantization_config:
            quantization_config = dict(collection_info.config.quantization_config)
            
            key = quantization_config.keys()
            if "scalar" in key:
                # QDRANT SUPPORT ONLY INT8 IN SCALAR
                endee_precision = Precision.INT8
            elif "product" in key:
                raise ValueError(f"Product quantization is not supported: {quantization_config}")
                
            elif "binary" in key:
                # DON'T SUPPORT ASYMMETRIC QUANTIZATION
                binary_config = quantization_config.get("binary", {})
                if "query_encoding" in binary_config:
                    raise ValueError(
                        f"Asymmetric quantization is not supported: {quantization_config}"
                    )
                encoding = quantization_config.get("encoding",None)
                if not encoding:
                    endee_precision = Precision.BINARY2
                else:
                    raise ValueError(f"Invalid binary quantization encoding: {encoding}")
            else:
                endee_precision = Precision.FLOAT32
        else:
            endee_precision = Precision.FLOAT32
        
        
        config = {
            DIMENSION_KEY: vectors_dimension,
            SPACE_TYPE_KEY: endee_space_type,
            SPARSE_DIMENSION_KEY: sparse_dimension,
            M_KEY: M,
            EF_CONSTRUCT_KEY: ef_construct,
            PRECISION_KEY: endee_precision
        }
        
        logger.info(f"✓ Collection config: dim={config[DIMENSION_KEY]}, "
                   f"space={config[SPACE_TYPE_KEY]}, sparse_dim={config[SPARSE_DIMENSION_KEY]}, "
                   f"M={config[M_KEY]}, ef={config[EF_CONSTRUCT_KEY]}")
        return config
    
    def get_or_create_endee_index(self, config: Dict[str, Any]):
        """Get or create Endee hybrid index"""
        try:
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Index already exists: {self.endee_index_name}")
        except NotFoundException:
            logger.info(f"Creating hybrid index: {self.endee_index_name}")
            self.endee_client.create_index(
                name=self.endee_index_name,
                dimension=config[DIMENSION_KEY],
                sparse_model=DEFAULT_SPARSE_MODEL,
                M=config[M_KEY],
                ef_con=config[EF_CONSTRUCT_KEY],
                precision=config[PRECISION_KEY]
            )
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Created hybrid index: {self.endee_index_name}")
    
    async def async_fetch_batch(self, offset: Optional[Any]) -> tuple:
        """Fetch a single batch from Qdrant"""
        loop = asyncio.get_running_loop()
        logger.info(f"FETCH: submitting scroll to executor, offset={offset}")  # add this
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: self._scroll_with_logging(offset)  # wrap with logging
            ),
            timeout=30,
        )
        points_batch, next_offset = result
        return points_batch, next_offset

    def _scroll_with_logging(self, offset):
        logger.info(f"SCROLL: thread started, offset={offset}")   # runs in executor thread
        result = self.qdrant_client.scroll(
            collection_name=self.qdrant_collection,
            limit=self.fetch_batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        logger.info(f"SCROLL: thread finished, got {len(result[0])} records")
        return result

    
    def convert_records(self, points) -> list:
        """Convert Qdrant hybrid points to Endee format
        points: list of Qdrant points
        points type: list of qdrant_client.models.PointStruct
        Endee format: {id, vector, sparse_indices, sparse_values, meta}
        """
        records = []
        # CHECK IF FILTER FIELDS ARE PRESENT IN THE PAYLOAD
        if points:
            payload = points[0].payload
            logger.info(f"SAMPLE PAYLOAD: {payload.keys()}")
            for field in self.filter_fields:
                if field not in payload.keys():
                    raise ValueError(f"Field {field} not found in payload")
            if self.filter_fields:
                sample_filter = {k: v for k, v in payload.items() if k in self.filter_fields}
            else:
                sample_filter = payload
            logger.info(f"SAMPLE FILTER DATA: {sample_filter}")
            logger.info(f"SAMPLE FILTER TYPES: { {k: type(v).__name__ for k, v in sample_filter.items()} }")
        for point in points:
            try:
                # Extract dense and sparse vectors
                vector_data = point.vector
                
                # =============== HANDLE BOTH DENSE AND SPARSE VECTORS =========================
                if isinstance(vector_data, dict):
                    dense_vector = vector_data.get(self.dense_field_name)
                    sparse_data = vector_data.get(self.sparse_field_name)
                else:
                    # If vector is not a dict, assume it's dense only
                    dense_vector = vector_data
                    sparse_data = None
                payload = point.payload or {}

                # =============== HANDLE FILTERS AND META DATA =========================
                

                if self.filter_fields:
                    filter_data = {key: value for key, value in payload.items() if key in self.filter_fields}
                    meta_data = {key: value for key, value in payload.items() if key not in self.filter_fields}
                else:
                    filter_data = {}
                    meta_data = payload

                
                record = {
                    ENDEE_ID_KEY: str(point.id),
                    ENDEE_VECTOR_KEY: dense_vector,
                    ENDEE_FILTER_KEY: filter_data,
                    ENDEE_META_KEY: point.payload
                }

                # Add sparse vector if present
                if sparse_data:
                    record[ENDEE_SPARSE_INDICES_KEY] = sparse_data.indices
                    record[ENDEE_SPARSE_VALUES_KEY] = sparse_data.values
                
                records.append(record)
                
            except Exception as e:
                logger.error(f"Error converting point {point.id}: {e}")
                # Skip this point and continue
                continue
        
        return records
    

    async def async_upsert_chunk(self, chunk: list) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: self.endee_index.upsert(chunk))
            return True
        except Exception as e:
            logger.error(f"Upsert chunk exception: {e}")
            raise


    async def async_upsert_records(self, records: list) -> bool:
        """Upsert records to Endee in chunks"""
        if self.interrupted:
            return False
        
        # SPLIT INTO CHUNKS
        chunks = [records[i:i + self.upsert_batch_size] for i in range(0, len(records), self.upsert_batch_size)]


        # UPSERT CHUNKS
        tasks = [self.async_upsert_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # FAILED RESULTS WILL HOLD CHUNKS
        failed_chunks = [chunks[i] for i, result in enumerate(results) if isinstance(result, Exception)]
        # RETRY FAILED CHUNKS
        while failed_chunks:
            chunk = failed_chunks.pop(0)
            retried = False
            for attempt in range(3):
                try:
                    # ASYNC UPSERT CHUNK RAISE EXCEPTION IF FAILED
                    await self.async_upsert_chunk(chunk)
                    retried = True
                    break
                except Exception as e:
                    logger.warning(f"Retry attempt {attempt + 1}/3 failed: {e}")
                    await asyncio.sleep(2 ** attempt)

            if not retried:
                logger.error("Chunk failed after 3 retries. Stopping migration.")
                return False
        
        return True



    
    async def async_migrate(self):
        """
            Main async migration with Producer-Consumer pattern
            
            How it works:
            1. Producer fetches batches → puts in queue (max size = 3)
            2. Consumer takes from queue → upserts to Endee
            3. If queue is full, producer WAITS (no memory overflow!)
            4. If queue is empty, consumer WAITS (no busy waiting!)
        """
        self.stats[START_TIME_KEY] = time.time()

        logger.info("="*80)
        logger.info("ASYNC HYBRID MIGRATION STARTED")
        logger.info("="*80)
        logger.info(f"Source: {self.qdrant_collection} @ {self.qdrant_url}:{self.qdrant_port}")
        logger.info(f"Target: {self.endee_index_name}")
        logger.info(f"Format: Hybrid (Dense + Sparse)")
        logger.info(f"Fetch batch size: {self.fetch_batch_size}")
        logger.info(f"Upsert batch size: {self.upsert_batch_size}")
        logger.info("="*80)
        
        # Show checkpoint status
        if self.checkpoint.get_processed_count() > 0:
            logger.info(f"RESUMING from checkpoint:")
            logger.info(f"  - Already processed: {self.checkpoint.get_processed_count()} records")
            logger.info(f"  - Starting from batch: {self.checkpoint.get_batch_number() + 1}")
            logger.info("="*80)
        
        # Setup connections
        self.connect_qdrant()
        self.connect_endee()
        
        # Get collection config and create/verify index
        config = self.get_qdrant_collection_info()

        self.get_or_create_endee_index(config)
        
        # Verify index is ready
        if self.endee_index is None:
            raise RuntimeError("Endee index not initialized!")
        logger.info(f"✓ Endee hybrid index ready")
        
        # Start migration
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        
        logger.info("\nStarting migration loop...")
        logger.info("="*80)

        # CREATE BOUNDED QUEUE WITH MAX SIZE OF 5 
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        logger.info(f"BOUNDED QUEUE CREATED WITH MAX SIZE OF {self.max_queue_size}")

        
        with tqdm(desc="Migrating records", unit="records", 
                 initial=self.checkpoint.get_processed_count()) as pbar:
            
            self._stop_event = asyncio.Event()
            await asyncio.gather(self.async_producer(queue), self.async_consumer(queue, pbar),)
        

        logger.info("ASYNC MIGRATION EXIT")
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

        if self.interrupted:
            logger.info("Progress saved. Run again to resume from checkpoint.")
        elif self.stats[FAILED_KEY] > 0:
            logger.warning("Migration had errors. Check logs and retry.")
        else:
            logger.info("Migration successful!")
        logger.info("="*80)

    def migrate(self):
        """SYNCHRONOUS WRAPPER"""
        if self.is_multivector:
            raise ValueError("Multivector mode is not supported for Qdrant to Endee migration")
        asyncio.run(self.async_migrate())

def main():
    parser = argparse.ArgumentParser(
        description="Simple sequential hybrid migration from Qdrant to Endee"
    )
    
    # Source arguments
    parser.add_argument("--source_url", default=os.getenv("SOURCE_URL"), help="Qdrant cluster endpoint")
    parser.add_argument("--source_api_key", default=os.getenv("SOURCE_API_KEY",""), help="Qdrant API key")
    parser.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"), help="Qdrant collection name")
    parser.add_argument("--source_port", type=int, default=os.getenv("SOURCE_PORT"), help="Qdrant port")
    parser.add_argument("--filter_fields", default=os.getenv("FILTER_FIELDS",""), help="Comma-separated payload fields to use as Endee filter (e.g. category,price,year). "
         "All other fields go to meta. If not set, everything goes to meta.")
    # Target arguments
    parser.add_argument("--target_url", default=os.getenv("TARGET_URL"), help="Endee URL")
    parser.add_argument("--target_api_key", default=os.getenv("TARGET_API_KEY",""), help="Endee API key")
    parser.add_argument("--target_collection", default=os.getenv("TARGET_COLLECTION"), help="Endee index name")
    
    # Performance arguments
    parser.add_argument("--batch_size", type=int, default=DEFAULT_FETCH_BATCH_SIZE, 
                       help="Fetch batch size (default: 1000)")
    parser.add_argument("--upsert_size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE, 
                       help="Upsert batch size (default: 1000)")
    
    # Connection arguments

    
    # Resume arguments
    parser.add_argument("--checkpoint_file", default=CHECKPOINT_FILE, 
                       help="Checkpoint file path (default: ./migration_checkpoint.json)")
    parser.add_argument("--clear_checkpoint", action="store_true", 
                       default=os.getenv("CLEAR_CHECKPOINT",'false').lower()=="true",
                       help="Clear existing checkpoint and start fresh")

    parser.add_argument(
        "--is_multivector",
        action="store_true",
        default=os.getenv("IS_MULTIVECTOR")==True,
        help="Enable multivector mode. When set, each record can contain multiple vectors instead of a single vector."
    )
    
    
    # Debug
    parser.add_argument("--debug", action="store_true", 
                       default=os.getenv("DEBUG",False),
                       help="Enable debug logging")

    parser.add_argument("--use_https", action="store_true",
                       default=os.getenv("USE_HTTPS",False),
                       help="Use HTTPS for Qdrant connection")

    parser.add_argument("--max_queue_size", type=int, default=os.getenv("MAX_QUEUE_SIZE",5),
                       help="Max queue size (default: 5)")

    
    args = parser.parse_args()
    
    # # Set debug level if requested
    # if args.debug:
    #     logging.getLogger().setLevel(logging.DEBUG)
    
    # Create migrator
    migrator = QdrantHybridToEndeeMigrator(
        qdrant_url=args.source_url,
        qdrant_port=args.source_port,
        qdrant_api_key=args.source_api_key,
        qdrant_collection=args.source_collection,
        endee_url=args.target_url,
        endee_api_key=args.target_api_key,
        endee_index=args.target_collection,
        max_queue_size=args.max_queue_size,
        fetch_batch_size=args.batch_size,
        upsert_batch_size=args.upsert_size,
        use_https=args.use_https,
        checkpoint_file=args.checkpoint_file,
        filter_fields=args.filter_fields,
        is_multivector=args.is_multivector
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
        # import traceback
        # logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()