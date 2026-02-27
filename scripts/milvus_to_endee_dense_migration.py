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
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.data, f, indent=2)
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


class SimpleMilvusToEndeeMigrator:
    """Simple sequential migration from Milvus (Dense) to Endee"""
    
    def __init__(
        self,
        milvus_url: str,
        milvus_token: str,
        milvus_collection: str,
        endee_url: str,
        endee_api_key: str,
        endee_index: str,
        milvus_port: int = 19530,
        fetch_batch_size: int = 1000,
        upsert_batch_size: int = 1000,
        space_type: str = "cosine",
        M: int = 16,
        ef_construct: int = 128,
        checkpoint_file: str = "./migration_checkpoint.json"
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
        
        # Collection config
        self.space_type = space_type
        self.M = M
        self.ef_construct = ef_construct
        self.precision = Precision.FLOAT32
        
        self.checkpoint = MigrationCheckpoint(checkpoint_file)
        self.interrupted = False
        
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
            "fetched": 0,
            "upserted": 0,
            "failed": 0,
            "batches_processed": 0,
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
            # If it's localhost or an IP without protocol, add http://
            if uri.startswith('localhost') or uri.replace('.', '').replace(':', '').isdigit():
                uri = f"http://{uri}:{self.milvus_port}"
                logger.info(f"Added protocol to URI: {uri}")
        
        self.milvus_client = MilvusClient(uri=uri, token=self.milvus_token)
        print("milvus_collections: ", self.milvus_client.list_collections())
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
        print("desc: ",desc)
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
                index_info = self.milvus_client.describe_index(self.milvus_collection, field_name)
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
            'id_field': id_field,
            'vector_fields': vector_fields,
            'other_fields': other_fields
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
    
    def fetch_batch(self, offset: int) -> list:
        """Fetch a single batch from Milvus"""
        try:
            results = self.milvus_client.query(
                collection_name=self.milvus_collection,
                filter="",
                output_fields=["*"],
                limit=self.fetch_batch_size,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            raise
    
    def convert_records(self, milvus_records) -> list:
        records = []
        for record in milvus_records:
            try:
                record_id = str(record.get(self.id_field_name, ''))
                raw_vector = record.get(self.vector_field_name, [])
                
                # ← this must be called
                logger.debug(f"vector_field_type={self.vector_field_type}, raw_vector type={type(raw_vector)}")
                vector = self.decode_vector(raw_vector, self.vector_field_type)

                endee_record = {
                    "id": record_id,
                    "vector": vector,
                    "meta": {}
                }

                for k, v in record.items():
                    if k not in [self.id_field_name, self.vector_field_name]:
                        if isinstance(v, (dict, list)):
                            endee_record["meta"][k] = json.dumps(v)
                        else:
                            endee_record["meta"][k] = v

                records.append(endee_record)

            except Exception as e:
                logger.error(f"Error converting record {record.get(self.id_field_name, 'unknown')}: {e}")
                continue

        return records
    
    def upsert_records(self, records: list) -> bool:
        """Upsert records to Endee in chunks"""
        # Split into smaller chunks if needed
        for i in range(0, len(records), self.upsert_batch_size):
            if self.interrupted:
                return False
            
            chunk = records[i:i + self.upsert_batch_size]
            try:
                result = self.endee_index.upsert(chunk)
                if not result:
                    logger.error(f"Upsert chunk failed (returned False)")
                    return False
                logger.debug(f"  Upserted chunk: {len(chunk)} records")
            except Exception as e:
                logger.error(f"Upsert chunk exception: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        
        return True
    
    def migrate(self):
        """Main migration function - simple sequential processing"""
        self.stats["start_time"] = time.time()
        
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
        
        # Start migration
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        
        logger.info("\nStarting migration loop...")
        logger.info("="*80)
        
        with tqdm(desc="Migrating records", unit="records", 
                 initial=self.checkpoint.get_processed_count()) as pbar:
            
            while not self.interrupted:
                try:
                    # Fetch batch from Milvus
                    logger.info(f"\n[Batch {batch_number}] Fetching from Milvus (offset: {offset})...")
                    milvus_results = self.fetch_batch(offset)
                    
                    # Check if done
                    if not milvus_results or len(milvus_results) == 0:
                        logger.info("✓ No more data to fetch")
                        break
                    
                    # Convert to Endee format
                    records = self.convert_records(milvus_results)
                    records_count = len(records)
                    self.stats["fetched"] += records_count
                    
                    logger.info(f"[Batch {batch_number}] Fetched {records_count} records")
                    
                    # # Show sample record structure (first batch only)
                    # if batch_number == 0 and records:
                    #     logger.info(f"\nSample Endee record structure:")
                    #     sample = records[0].copy()
                    #     # Truncate vector for display
                    #     if 'vector' in sample and len(sample['vector']) > 5:
                    #         sample['vector'] = f"[{sample['vector'][:3]}... ({len(sample['vector'])} dims)]"
                    #     logger.info(json.dumps(sample, indent=2))
                    
                    # Upsert to Endee
                    logger.info(f"[Batch {batch_number}] Upserting to Endee...")
                    print("records: ",records)
                    success = self.upsert_records(records)
                    
                    if success:
                        # Update offset for next batch
                        new_offset = offset + records_count
                        
                        # Update checkpoint
                        self.checkpoint.update(batch_number, records_count, new_offset)
                        self.stats["upserted"] += records_count
                        self.stats["batches_processed"] += 1
                        
                        # Update progress bar
                        pbar.update(records_count)
                        
                        logger.info(f"[Batch {batch_number}] ✓ Successfully upserted {records_count} records")
                        
                        # If we got fewer results than requested, we've reached the end
                        if len(milvus_results) < self.fetch_batch_size:
                            logger.info(f"✓ Reached end of collection (got {len(milvus_results)} < {self.fetch_batch_size})")
                            break
                        
                        # Move to next batch
                        offset = new_offset
                        batch_number += 1
                    else:
                        self.stats["failed"] += records_count
                        logger.error(f"[Batch {batch_number}] ✗ Failed to upsert")
                        break
                    
                except Exception as e:
                    logger.error(f"[Batch {batch_number}] Exception: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self.stats["failed"] += records_count if 'records_count' in locals() else 0
                    break
        
        # Print final report
        self._print_final_report()
    
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
            logger.info(f"  {self.vector_field_name} → vector")
            if self.vector_field_info.get('other_fields'):
                logger.info(f"  Other fields → metadata.{{field_name}}")
            logger.info("="*80)
        
        if self.interrupted:
            logger.info("Progress saved. Run again to resume from checkpoint.")
        elif self.stats['failed'] > 0:
            logger.warning("Migration had errors. Check logs and retry.")
        else:
            logger.info("Migration successful!")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Simple sequential migration from Milvus to Endee (Dense vectors)"
    )
    
    # Source arguments
    parser.add_argument("--source_url", default=os.getenv("SOURCE_URL"), help="Milvus URI")
    parser.add_argument("--source_api_key", default=os.getenv("SOURCE_API_KEY"), help="Milvus token")
    parser.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"), help="Milvus collection name")
    parser.add_argument("--source_port", type=int, default=os.getenv("SOURCE_PORT"), help="Milvus port")

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
                       default=os.getenv("CLEAR_CHECKPOINT",False),
                       help="Clear existing checkpoint and start fresh")
    
    # Debug
    parser.add_argument("--debug", action="store_true", 
                       default=os.getenv("DEBUG",False),
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
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
        checkpoint_file=args.checkpoint_file
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()