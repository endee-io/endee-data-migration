from typing import Dict, Any, Optional
import urllib
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
import os
import dotenv
import numpy as np
import asyncio
import orjson

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
    
    def __init__(self, checkpoint_file: str = "./migration_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load checkpoint from file"""
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                logger.info(f"✓ Loaded checkpoint: {data.get('processed_count', 0)} records processed")
                return data
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh migration")
            return {
                "processed_count": 0,
                "last_offset": 0,
                "batch_number": 0
            }
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}, starting fresh")
            return {
                "processed_count": 0,
                "last_offset": 0,
                "batch_number": 0
            }
    
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
        self.data["processed_count"] += records_count
        self.data["batch_number"] = batch_number
        self.data["last_offset"] = offset
        self.save()
    
    def get_last_offset(self) -> int:
        """Get the last processed offset"""
        return self.data.get("last_offset", 0)
    
    def get_batch_number(self) -> int:
        """Get the last processed batch number"""
        return self.data.get("batch_number", 0)
    
    def get_processed_count(self) -> int:
        """Get total processed records"""
        return self.data.get("processed_count", 0)
    
    def clear(self):
        """Clear checkpoint for fresh start"""
        self.data = {
            "processed_count": 0,
            "last_offset": 0,
            "batch_number": 0
        }
        self.save()


class HybridMilvusToEndeeMigrator:
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
        M: int,
        ef_construct: int,
        milvus_port: int = 19530,
        fetch_batch_size: int = 1000,
        upsert_batch_size: int = 1000,
        space_type: str = "cosine",
        checkpoint_file: str = "./migration_checkpoint.json",
        filter_fields: str = "",
        is_multivector: bool = False,
        max_queue_size: int = 5
    ):
        self.milvus_url = milvus_url
        self.milvus_token = milvus_token
        self.milvus_collection = milvus_collection
        self.endee_url = endee_url
        self.endee_api_key = endee_api_key
        self.endee_index_name = endee_index
        self.fetch_batch_size = fetch_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.filter_fields = set(f.strip() for f in filter_fields.split(",") if f.strip()) if filter_fields else set()
        # Collection config (will be auto-detected)
        self.space_type = space_type
        self.M = M
        self.ef_construct = ef_construct
        self.precision = Precision.FLOAT16
        self.is_multivector = is_multivector
        self.checkpoint = MigrationCheckpoint(checkpoint_file)
        self.interrupted = False
        self.max_queue_size = max_queue_size
        self._stop_event = None
        # Field detection info (populated after connection)
        self.vector_field_info = None
        self.id_field_name = None
        self.dense_vector_field_name = None
        self.sparse_vector_field_name = None
        self.vectors_dimension = None
        self.sparse_dimension = None
        
        # Clients
        self.milvus_client = None
        self.endee_client = None
        self.endee_index = None

        self.dense_vector_field_type=None

        # Statistics
        self.stats = {
            "fetched": 0,
            "upserted": 0,
            "failed": 0,
            "batches_processed": 0,
            "records_with_sparse": 0,
            "records_without_sparse": 0,
            "start_time": None
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
        
        # Fix URI if needed - add protocol if missing
        uri = self.milvus_url
        if not uri.startswith(('http://', 'https://', 'tcp://', 'unix://')):
            if uri.startswith('localhost') or uri.replace('.', '').replace(':', '').isdigit():
                uri = f"http://{uri}:{self.milvus_port}"
                logger.info(f"Added protocol to URI: {uri}")
        
        self.milvus_client = MilvusClient(uri=uri, token=self.milvus_token)
        logger.info("✓ Connected to Milvus")
    
    def connect_endee(self):
        """Connect to Endee"""
        logger.info("Connecting to Endee...")
        
        # Initialize Endee client with API key
        self.endee_client = Endee(token=self.endee_api_key)

        
        # # Set custom base URL if provided
        if self.endee_url:
            url = urllib.parse.urljoin(self.endee_url, "/api/v1")
            self.endee_client.set_base_url(url)
            logger.info(f"Set Endee base URL: {url}")

        logger.info(f"{self.endee_client.list_indexes()}")
        logger.info("✓ Connected to Endee")
    
    def detect_sparse_dimension(self) -> int:
        """
        Detect actual sparse dimension by sampling records
        Returns the maximum sparse index + 1 with some buffer
        """
        if not self.sparse_vector_field_name:
            return 30000  # Default if no sparse vectors
        
        logger.info("\nDetecting sparse dimension from data...")
        
        max_index = 0
        sample_size = min(1000, self.fetch_batch_size)
        
        try:
            # Fetch a sample of records
            sample_records = self.milvus_client.query(
                collection_name=self.milvus_collection,
                filter="",
                output_fields=["*"],
                limit=sample_size,
                offset=0
            )
            
            # Find maximum sparse index
            for record in sample_records:
                sparse_data = record.get(self.sparse_vector_field_name, {})
                if sparse_data and isinstance(sparse_data, dict):
                    indices = sparse_data.keys()
                    if indices:
                        record_max = max(int(idx) for idx in indices)
                        max_index = max(max_index, record_max)
            
            # Add 10% buffer and round up
            sparse_dim = int(max_index * 1.1) + 100
            
            logger.info(f"✓ Sampled {len(sample_records)} records")
            logger.info(f"  - Maximum sparse index found: {max_index}")
            logger.info(f"  - Setting sparse dimension to: {sparse_dim}")
            
            return sparse_dim
            
        except Exception as e:
            logger.warning(f"Could not detect sparse dimension: {e}")
            logger.warning(f"Using default sparse dimension: 30000")
            return 30000
    
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

    def detect_vector_fields(self):
        """
        Auto-detect vector field names, ID field name, and dimensions
        Handles both dense-only and hybrid (dense + sparse) collections
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Detecting fields in collection: {self.milvus_collection}")
        logger.info(f"{'='*80}\n")
        
        # Get collection schema
        desc = self.milvus_client.describe_collection(self.milvus_collection)
        logger.info(f"desc: {desc}")
        # Storage for detected fields
        dense_vector_fields = []
        sparse_vector_fields = []
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
            
            # Detect dense vector fields
            elif field_type in ['FLOAT_VECTOR', 'FLOAT16_VECTOR', 'BINARY_VECTOR',
                    DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, 
                     DataType.BINARY_VECTOR]:
                index_info = self.milvus_client.describe_index(self.milvus_collection, field_name)
                params = field.get('params', {})
                dim = params.get('dim') or field.get('dim')

                
                # Detect precision from field type
                precision = (
                    MILVUS_DTYPE_TO_ENDEE_PRECISION.get(field_type) or
                    MILVUS_STR_TO_ENDEE_PRECISION.get(field_type) or
                    Precision.FLOAT32  # fallback
                )
                
                dense_vector_fields.append({
                    'name': field_name,
                    'type': field_type,
                    'dimension': dim,
                    'precision': precision   # ← store per vector field
                })
                
                self.ef_construct = index_info.get('params', {}).get('efConstruction', self.ef_construct)
                self.M = index_info.get('params', {}).get('M', self.M)

                # Use the first dense vector field found
                if self.dense_vector_field_name is None:
                    self.dense_vector_field_name = field_name
                    self.vectors_dimension = dim
                    self.precision = precision  # ← set on self
                logger.info(f"✓ Dense Vector Field: '{field_name}' [{field_type}, dim={dim}]")
            
            # Detect sparse vector fields
            elif field_type in ['SPARSE_FLOAT_VECTOR', DataType.SPARSE_FLOAT_VECTOR]:
                sparse_vector_fields.append({
                    'name': field_name,
                    'type': field_type
                })
                
                # Use the first sparse vector field found
                if self.sparse_vector_field_name is None:
                    self.sparse_vector_field_name = field_name
                
                logger.info(f"✓ Sparse Vector Field: '{field_name}' [{field_type}]")
            
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
        logger.info(f"  Dense Vector Fields: {len(dense_vector_fields)}")
        
        if dense_vector_fields:
            for vf in dense_vector_fields:
                logger.info(f"    - {vf['name']} (dim={vf['dimension']})")
        
        logger.info(f"  Sparse Vector Fields: {len(sparse_vector_fields)}")
        if sparse_vector_fields:
            for sf in sparse_vector_fields:
                logger.info(f"    - {sf['name']}")
        
        logger.info(f"  Metadata Fields: {len(other_fields)}")
        if other_fields:
            for of in other_fields:
                logger.info(f"    - {of['name']}")
        
        logger.info(f"\n{'='*80}\n")
        
        # Determine if hybrid or dense-only
        is_hybrid = len(sparse_vector_fields) > 0
        logger.info(f"Collection Type: {'HYBRID (Dense + Sparse)' if is_hybrid else 'DENSE ONLY'}")
        
        self.vector_field_info = {
            'id_field': id_field,
            'dense_vector_fields': dense_vector_fields,
            'sparse_vector_fields': sparse_vector_fields,
            'other_fields_meta': other_fields,
            'is_hybrid': is_hybrid
        }
        
        if not self.id_field_name:
            raise ValueError("No primary key field found in collection")
        if not self.dense_vector_field_name:
            raise ValueError("No dense vector field found in collection")
        
        return self.vector_field_info
    
    def get_or_create_endee_index(self):
        """Get or create Endee index (hybrid or dense-only based on detection)"""
        if not self.vectors_dimension:
            raise ValueError("Vector dimension not detected. Run detect_vector_fields() first.")
        
        is_hybrid = self.vector_field_info.get('is_hybrid', False)
        
        try:
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Index already exists: {self.endee_index_name}")
        except NotFoundException:
            if is_hybrid:
                # Detect actual sparse dimension from data
                self.sparse_dimension = self.detect_sparse_dimension()
                
                logger.info(f"Creating HYBRID index: {self.endee_index_name} {type(self.endee_index_name)}")
                logger.info(f"  - Dense dimension: {self.vectors_dimension} {type(self.vectors_dimension)}")
                logger.info(f"  - Sparse dimension: {self.sparse_dimension} {type(self.sparse_dimension)}")
                logger.info(f"  - Space type: {self.space_type} {type(self.space_type)}")
                logger.info(f"  - M: {self.M} {type(self.M)}")
                logger.info(f"  - ef_construct: {self.ef_construct} {type(self.ef_construct)}")
                logger.info(f"  - Precision: {self.precision} {type(self.precision)}")
                logger.info(f"  - Creating Index")
                self.endee_client.create_index(
                    name=self.endee_index_name,
                    dimension=self.vectors_dimension,
                    space_type=self.space_type,
                    sparse_dim=self.sparse_dimension,
                    M=self.M,
                    ef_con=self.ef_construct,
                    precision=self.precision
                )
                logger.info(f"✓ Created HYBRID index: {self.endee_index_name}")
            else:
                logger.info(f"Creating DENSE index: {self.endee_index_name}")
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
                logger.info(f"✓ Created DENSE index: {self.endee_index_name}")
            
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Created Endee index: {self.endee_index_name}")
    

    async def async_producer(self, queue: asyncio.Queue):
        """
            Fetch batches from Milvus and put them into the queue.

        Two stop checks:
          (A) while condition — top of every iteration
          (B) post-put check — closes race window where consumer fails
              WHILE producer is suspended inside await queue.put()

        """
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()

        logger.info("PRODUCER STARTED FETCHING FROM MILVUS")

        while not self.interrupted and not self._stop_event.is_set():
            try:
                # FETCH BATCH FROM MILVUS
                logger.info(f"FETCHING BATCH FROM QDRANT {batch_number} WITH OFFSET {offset}")
                points_batch, next_offset = await self.async_fetch_batch(offset)

                # CHECK IF POINTS BATCH IS EMPTY
                if not points_batch:
                    logger.info("PRODUCER: NO MORE DATA TO FETCH")
                    await queue.put(None)
                    break

                # CONVERT TO ENDEE FORMAT
                records = self.convert_records(points_batch)

                # UPDATE STATS
                self.stats["fetched"] += len(records)

                # CHECK IF INTERRUPTED THAT IS CTRL+C OR TERMINAL KILL
                if self.interrupted:
                    logger.info("PRODUCER: INTERRUPTED BY USER")
                    await queue.put(None)
                    break

                # PUT RECORDS INTO QUEUE
                await queue.put({
                    "batch_number": batch_number,
                    "records": records,
                    "next_offset": next_offset
                })
                
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: STOPPED DUE TO STOP EVENT")
                    break

                # TRACK QUEUE USAGE
                current_size = queue.qsize()
                if current_size > self.max_queue_size:
                    logger.warning(f"QUEUE IS FULL. CURRENT SIZE: {current_size}")
                    
                # MOVE TO NEXT BATCH
                offset = next_offset
                batch_number += 1

                # PUT NONE TO QUEUE IF OFFSET IS NONE
                if next_offset is None:
                    await queue.put(None)
                    break

            except Exception as e:
                logger.error(f"[Producer] Exception: {e}")
                self._stop_event.set()
                await queue.put(None)  # Signal error
                break
        logger.info("PRODUCER FINISHED")

    async def async_consumer(self, queue: asyncio.Queue, pbar:tqdm):
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
        logger.info("CONSUMER STARTED UPSERTING INTO ENDEE")
        while not self.interrupted:
            try:
                # GET THE DATA FROM QUEUE
                batch = await queue.get()

                # CHECK IF DATA IS NONE THAT IS CTRL+C OR TERMINAL KILL OR NO DATA PRESENT
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
                elapsed = end_time - start_time
                # UPSERT SUCCESSFULLY
                if success:
                    # UPDATE CHECKPOINT FIELDS
                    self.checkpoint.update(batch_number, records_count, next_offset)
                    # UPDATE STATS
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
                self._stop_event.set()       # 1. signal producer   ← BUG 3 fix
                queue.task_done()            # 2. unblock producer
                break   
        
        logger.info("CONSUMER FINISHED")

    async def async_fetch_batch(self, offset: int) -> list:
        """Fetch a single batch from Milvus"""
        loop = asyncio.get_running_loop()
        try:
            # RUN IN SEPARATE THREAD USING EVENT LOOP EXECUTOR
            result = await loop.run_in_executor(None, lambda: self.milvus_client.query(
                collection_name=self.milvus_collection,
                filter="",
                output_fields=["*"],
                limit=self.fetch_batch_size,
                offset=offset
            ))
            return result or []
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            raise # RE-RAISE SO PRODUCER CAN HANDLE IT  
    
    def convert_records(self, milvus_records) -> list:
        """
        Convert Milvus records to Endee format
        Endee format: {id, vector, sparse_indices, sparse_values, meta}
        """
        records = []
        if self.vector_field_info:
            payload_field_names = set(i.get('name') for i in self.vector_field_info.get('other_fields_meta',{}))
            print(f"payload_field_names:{payload_field_names}")
            for field in self.filter_fields:
                if field not in payload_field_names:
                    raise ValueError(f"Field {field} not found in payload")
        for record in milvus_records:
            try:
                # Extract ID using detected field name
                record_id = str(record.get(self.id_field_name, ''))
                raw_dense_vector = record.get(self.dense_vector_field_name, [])

                # Extract dense vector using detected field name
                dense_vector = self.decode_vector(raw_dense_vector, self.dense_vector_field_type)
                # sparse_vector = record.get(self.sparse_vector_field_name,{})
                if self.filter_fields:
                    filter_data = {key: value for key, value in record.items() if key in self.filter_fields}
                    meta_data = {key: value for key, value in record.items() if key not in self.filter_fields and key in payload_field_names}
                else:
                    filter_data = {key:value for key, value in record.items() if key in payload_field_names}
                    meta_data = {}
          
                # Build Endee record
                endee_record = {
                    "id": record_id,
                    "vector": dense_vector,
                    "filter": filter_data,
                    "meta": meta_data  # All other fields go here
                }
                
                # Extract sparse vector if present
                if self.sparse_vector_field_name:
                    sparse_data = record.get(self.sparse_vector_field_name, {})
                    
                    if sparse_data and isinstance(sparse_data, dict):
                        # Convert dict {index: value} to separate lists
                        indices = []
                        values = []
                        
                        for idx, val in sorted(sparse_data.items()):
                            indices.append(int(idx))
                            values.append(float(val))
                        
                        if indices:
                            endee_record["sparse_indices"] = indices
                            endee_record["sparse_values"] = values
                            self.stats["records_with_sparse"] += 1
                        else:
                            self.stats["records_without_sparse"] += 1
                    else:
                        self.stats["records_without_sparse"] += 1
                
                # Add all other fields to meta
                # Exclude: ID, dense vector, sparse vector
                excluded_fields = [self.id_field_name, self.dense_vector_field_name]
                if self.sparse_vector_field_name:
                    excluded_fields.append(self.sparse_vector_field_name)
                
                # for k, v in record.items():
                #     if k not in excluded_fields:
                #         # Handle complex types
                #         if isinstance(v, (dict, list)):
                #             endee_record["meta"][k] = json.dumps(v)
                #         else:
                #             endee_record["meta"][k] = v
                
                records.append(endee_record)
                
            except Exception as e:
                logger.error(f"Error converting record {record.get(self.id_field_name, 'unknown')}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Skip this record and continue
                continue
        
        return records
    
    async def async_upsert_chunk(self, chunk: list) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: self.endee_index.upsert(chunk))
            return True
        except Exception as e:
            logger.error(f"Upsert chunk exception: {e}")
            raise

    async def async_upsert_records(self, records: list) -> bool:
        """Upsert records to Endee in chunks
        - splits into chunks
        - upserts all chunks in parallel via gather()
        - retries each failed chunk with exponential backoff
        """
        if self.interrupted:
            return False

        chunks = [records[i:i + self.upsert_batch_size] for i in range(0, len(records), self.upsert_batch_size)]

        # PHASE 1: ALL CHUNKS IN PARALLEL
        results = await asyncio.gather(
            *[self.async_upsert_chunk(c) for c in chunks],
            return_exceptions=True
        )

        # PHASE 2: COLLECT FAILURES
        failed_chunks = [
            chunks[i] for i, result in enumerate(results) if isinstance(result, Exception)
        ]
        if failed_chunks:
            logger.warning(f"Failed to upsert {len(failed_chunks)} chunks. Retrying...")

        # PHASE 3: RETRY WITH EXPONENTIAL BACKOFF
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
    # ══════════════════════════════════════════════════════════════
    # ASYNC ORCHESTRATOR
    # ══════════════════════════════════════════════════════════════
    async def async_migrate(self):
        """Main migration function"""
        self.stats["start_time"] = time.time()
        if self.is_multivector:
            raise ValueError("Multivector mode is not supported for Milvus to Endee hybrid migration")
        logger.info("="*80)
        logger.info("MILVUS → ENDEE MIGRATION (AUTO-DETECT HYBRID/DENSE)")
        logger.info("="*80)
        logger.info(f"Source: {self.milvus_collection} @ {self.milvus_url}")
        logger.info(f"Target: {self.endee_index_name}")
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
        logger.info(f"milvus connected")
        self.connect_endee()
        logger.info(f"endee connected")
        # Detect field names and types from Milvus schema
        self.detect_vector_fields()
        logger.info(f"vector fields detected")
        # Create/verify Endee index with detected configuration
        self.get_or_create_endee_index()
        logger.info(f"endee_index: {self.endee_index}")
        # Verify index is ready
        if self.endee_index is None:
            raise RuntimeError("Endee index not initialized!")
        logger.info(f"✓ Endee index ready")
        
        # Start migration
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        
        logger.info("\nStarting migration loop...")
        logger.info("="*80)

        queue = asyncio.Queue(maxsize=self.max_queue_size)
        logger.info(f"BOUNDED QUEUE CREATED WITH MAX SIZE OF {self.max_queue_size}")
        
        with tqdm(desc="Migrating records", unit="records", 
                 initial=self.checkpoint.get_processed_count()) as pbar:
            
            # CREATE STOP EVENT FOR GRACEFUL SHUTDOWN
            # THIS IS USED TO STOP THE PRODUCER AND CONSUMER
            self._stop_event = asyncio.Event()
            # START PRODUCER AND CONSUMER
            await asyncio.gather(self.async_producer(queue), self.async_consumer(queue, pbar))

        logger.info("ASYNC MIGRATION COMPLETED")
        logger.info("="*80)

        self._print_final_report()

            # while not self.interrupted:
            #     try:
            #         # Fetch batch from Milvus
            #         logger.info(f"\n[Batch {batch_number}] Fetching from Milvus (offset: {offset})...")
            #         milvus_results = self.fetch_batch(offset)
                    
            #         # Check if done
            #         if not milvus_results or len(milvus_results) == 0:
            #             logger.info("✓ No more data to fetch")
            #             break
                    
            #         # Convert to Endee format
            #         records = self.convert_records(milvus_results)
            #         records_count = len(records)
            #         self.stats["fetched"] += records_count
                    
            #         logger.info(f"[Batch {batch_number}] Fetched {records_count} records")
                    
            #         # Show sample record structure (first batch only)
            #         if batch_number == 0 and records:
            #             logger.info(f"\nSample Endee record structure:")
            #             sample = records[0].copy()
            #             # Truncate vectors for display
            #             if 'vector' in sample and len(sample['vector']) > 5:
            #                 sample['vector'] = f"[{sample['vector'][:3]}... ({len(sample['vector'])} dims)]"
            #             if 'sparse_indices' in sample and len(sample['sparse_indices']) > 5:
            #                 sample['sparse_indices'] = f"{sample['sparse_indices'][:5]}... ({len(sample['sparse_indices'])} items)"
            #             if 'sparse_values' in sample and len(sample['sparse_values']) > 5:
            #                 sample['sparse_values'] = f"{sample['sparse_values'][:5]}... ({len(sample['sparse_values'])} items)"
            #             logger.info(json.dumps(sample, indent=2))
                    
            #         # Upsert to Endee
            #         logger.info(f"[Batch {batch_number}] Upserting to Endee...")
            #         success = self.upsert_records(records)
                    
            #         if success:
            #             # Update offset for next batch
            #             new_offset = offset + records_count
                        
            #             # Update checkpoint
            #             self.checkpoint.update(batch_number, records_count, new_offset)
            #             self.stats["upserted"] += records_count
            #             self.stats["batches_processed"] += 1
                        
            #             # Update progress bar
            #             pbar.update(records_count)
                        
            #             logger.info(f"[Batch {batch_number}] ✓ Successfully upserted {records_count} records")
                        
            #             # If we got fewer results than requested, we've reached the end
            #             if len(milvus_results) < self.fetch_batch_size:
            #                 logger.info(f"✓ Reached end of collection (got {len(milvus_results)} < {self.fetch_batch_size})")
            #                 break
                        
            #             # Move to next batch
            #             offset = new_offset
            #             batch_number += 1
            #         else:
            #             self.stats["failed"] += records_count
            #             logger.error(f"[Batch {batch_number}] ✗ Failed to upsert")
            #             break
                    
            #     except Exception as e:
            #         logger.error(f"[Batch {batch_number}] Exception: {e}")
            #         import traceback
            #         logger.error(f"Traceback: {traceback.format_exc()}")
            #         self.stats["failed"] += records_count if 'records_count' in locals() else 0
            #         break
        
        ## Print final report
        # self._print_final_report()
    
    def _print_final_report(self):
        """Print migration summary"""
        duration = time.time() - self.stats["start_time"]
        
        logger.info("\n" + "="*80)
        if self.interrupted:
            logger.warning("MIGRATION INTERRUPTED")
        elif self.stats["failed"] > 0:
            logger.warning("MIGRATION COMPLETED WITH ERRORS")
        else:
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total records processed: {self.checkpoint.get_processed_count()}")
        logger.info(f"Records fetched this run: {self.stats['fetched']}")
        logger.info(f"Records upserted this run: {self.stats['upserted']}")
        
        if self.vector_field_info and self.vector_field_info.get('is_hybrid'):
            logger.info(f"Records with sparse vectors: {self.stats['records_with_sparse']}")
            logger.info(f"Records without sparse vectors: {self.stats['records_without_sparse']}")
        
        logger.info(f"Records failed: {self.stats['failed']}")
        logger.info(f"Batches processed: {self.stats['batches_processed']}")
        
        if self.stats['upserted'] > 0:
            rate = self.stats['upserted'] / duration
            logger.info(f"Throughput: {rate:.2f} records/second")
        
        logger.info("="*80)
        
        # Show field mapping used
        if self.vector_field_info:
            logger.info("\nField Mapping (Milvus → Endee):")
            logger.info(f"  {self.id_field_name} → id")
            logger.info(f"  {self.dense_vector_field_name} → vector")
            if self.sparse_vector_field_name:
                logger.info(f"  {self.sparse_vector_field_name} → sparse_indices + sparse_values")
            if self.vector_field_info.get('other_fields'):
                logger.info(f"  Other fields → meta.{{field_name}}")
            logger.info("="*80)
        
        if self.interrupted:
            logger.info("Progress saved. Run again to resume from checkpoint.")
        elif self.stats['failed'] > 0:
            logger.warning("Migration had errors. Check logs and retry.")
        else:
            logger.info("Migration successful!")
        logger.info("="*80)

    def migrate(self):
        asyncio.run(self.async_migrate())
def main():
    parser = argparse.ArgumentParser(
        description="Migration from Milvus to Endee (Auto-detects Hybrid or Dense)"
    )
    
    # Source arguments
    parser.add_argument("--source_url", default=os.getenv("SOURCE_URL"), help="Milvus URI")
    parser.add_argument("--source_api_key", default=os.getenv("SOURCE_API_KEY"), help="Milvus token")
    parser.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"), help="Milvus collection name")
    parser.add_argument("--source_port", type=int, default=os.getenv("SOURCE_PORT"), help="Milvus port")
    parser.add_argument("--filter_fields", default=os.getenv("FILTER_FIELDS",""), help="Filter fields")
    parser.add_argument("--is_multivector", action="store_true",
                       default=os.getenv("IS_MULTIVECTOR",False),
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
    parser.add_argument("--space_type", default=os.getenv("SPACE_TYPE","cosine"),
                       help="Distance metric (default: cosine)")
    parser.add_argument("--M", type=int, default=os.getenv("M",16),
                       help="HNSW M parameter (default: 16)")
    parser.add_argument("--ef_construct", type=int, default=os.getenv("EF_CONSTRUCT",128),
                       help="HNSW ef_construct parameter (default: 128)")
    
    # Resume arguments
    parser.add_argument("--checkpoint_file", default=os.getenv("CHECKPOINT_FILE","./migration_checkpoint.json"), 
                       help="Checkpoint file path (default: ./migration_checkpoint.json)")
    parser.add_argument("--clear_checkpoint", action="store_true", 
                       default=os.getenv("CLEAR_CHECKPOINT","false").lower() == "true",
                       help="Clear existing checkpoint and start fresh")
    
    # Debug
    parser.add_argument("--debug", action="store_true", 
                       default=os.getenv("DEBUG",False),
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"creating migrator")
    # Create migrator
    migrator = HybridMilvusToEndeeMigrator(
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
        is_multivector=args.is_multivector
    )
    logger.info(f"migrator created")
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()