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
from constants import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationCheckpoint:
    """Simple checkpoint for resume capability"""
    
    def __init__(self, checkpoint_file: str = CHECKPOINT_FILE, use_https: bool = False):
        self.checkpoint_file = checkpoint_file
        self.use_https = use_https
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
                data = orjson.load(f)
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
            with open(self.checkpoint_file, 'w') as f:
                orjson.dump(self.data, f, indent=2)
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


class SimpleQdrantToEndeeMigrator:
    """Simple sequential migration from Qdrant to Endee"""
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_port: int,
        qdrant_api_key: str,
        qdrant_collection: str,
        endee_api_key: str,
        endee_index: str,
        fetch_batch_size: int = DEFAULT_FETCH_BATCH_SIZE,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        checkpoint_file: str = CHECKPOINT_FILE,
        use_https: bool = False
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection = qdrant_collection
        self.endee_api_key = endee_api_key
        self.endee_index_name = endee_index
        self.fetch_batch_size = fetch_batch_size
        self.upsert_batch_size = upsert_batch_size
        print(f"use_https: {use_https}")
        self.use_https = use_https
        self.checkpoint = MigrationCheckpoint(checkpoint_file, use_https)
        self.interrupted = False

        # Clients
        self.qdrant_client = None
        self.endee_client = None
        self.endee_index = None

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
    
    def connect_qdrant(self):
        """Connect to Qdrant"""
        logger.info("Connecting to Qdrant...")
        self.qdrant_client = QdrantClient(
            host=self.qdrant_url,
            port=self.qdrant_port,
            api_key=self.qdrant_api_key,
            https=bool(self.use_https)
        )
        logger.info("✓ Connected to Qdrant")
    
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
    
    def get_qdrant_collection_info(self) -> Dict[str, Any]:
        """Get collection configuration from Qdrant"""
        logger.info(f"Getting collection info for: {self.qdrant_collection}")
        collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
        
        vectors = collection_info.config.params.vectors
        
        if isinstance(vectors, dict):
            vectors_map = vectors
        elif vectors is not None:
            vectors_map = {"default": vectors}
        else:
            vectors_map = {}
        
        vectors_dimension = DEFAULT_VECTOR_DIMENSION
        space_type = DEFAULT_SPACE_TYPE
        
        for _, config in vectors_map.items():
            vectors_dimension = config.size
            space_type = config.distance
            break
        
        M = collection_info.config.hnsw_config.m
        ef_construct = collection_info.config.hnsw_config.ef_construct
        
        config = {
            DIMENSION_KEY: vectors_dimension,
            SPACE_TYPE_KEY: space_type,
            M_KEY: M,
            EF_CONSTRUCT_KEY: ef_construct,
            PRECISION_KEY: Precision.FLOAT16
        }
        
        logger.info(f"✓ Collection config: dim={config[DIMENSION_KEY]}, "
                   f"space={config[SPACE_TYPE_KEY]}, M={config[M_KEY]}, ef={config[EF_CONSTRUCT_KEY]}")
        return config
    
    def get_or_create_endee_index(self, config: Dict[str, Any]):
        """Get or create Endee index"""
        try:
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Index already exists: {self.endee_index_name}")
        except NotFoundException:
            logger.info(f"Creating index: {self.endee_index_name}")
            self.endee_client.create_index(
                name=self.endee_index_name,
                dimension=config[DIMENSION_KEY],
                space_type=config[SPACE_TYPE_KEY],
                M=config[M_KEY],
                ef_con=config[EF_CONSTRUCT_KEY],
                precision=config[PRECISION_KEY]
            )
            self.endee_index = self.endee_client.get_index(self.endee_index_name)
            logger.info(f"✓ Created index: {self.endee_index_name}")
    
    def fetch_batch(self, offset: Optional[Any]) -> tuple:
        """Fetch a single batch from Qdrant"""
        points_batch, next_offset = self.qdrant_client.scroll(
            collection_name=self.qdrant_collection,
            limit=self.fetch_batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        return points_batch, next_offset
    
    def convert_records(self, points) -> list:
        """Convert Qdrant points to Endee format"""
        return [
            {
                ENDEE_ID_KEY: str(point.id),
                ENDEE_VECTOR_KEY: point.vector,
                ENDEE_META_KEY: point.payload
            }
            for point in points
        ]
    
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
        self.stats[START_TIME_KEY] = time.time()
        
        logger.info("="*80)
        logger.info("SIMPLE SEQUENTIAL MIGRATION")
        logger.info("="*80)
        logger.info(f"Source: {self.qdrant_collection} @ {self.qdrant_url}:{self.qdrant_port}")
        logger.info(f"Target: {self.endee_index_name}")
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
        logger.info(f"Endee index ready")
        
        # Start migration
        offset = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        
        logger.info("\nStarting migration loop...")
        logger.info("="*80)
        
        with tqdm(desc="Migrating records", unit="records", 
                 initial=self.checkpoint.get_processed_count()) as pbar:
            
            while not self.interrupted:
                try:
                    # Fetch batch from Qdrant
                    logger.info(f"\n[Batch {batch_number}] Fetching from Qdrant (offset: {offset})...")
                    points_batch, next_offset = self.fetch_batch(offset)
                    
                    # Check if done
                    if not points_batch:
                        logger.info("✓ No more data to fetch")
                        break
                    
                    # Convert to Endee format
                    records = self.convert_records(points_batch)
                    records_count = len(records)
                    self.stats[FETCHED_KEY] += records_count
                    
                    logger.info(f"[Batch {batch_number}] Fetched {records_count} records")
                    
                    # Upsert to Endee
                    logger.info(f"[Batch {batch_number}] Upserting to Endee...")
                    success = self.upsert_records(records)
                    
                    if success:
                        # Update checkpoint
                        self.checkpoint.update(batch_number, records_count, next_offset)
                        self.stats[UPSERTED_KEY] += records_count
                        self.stats[BATCHES_PROCESSED_KEY] += 1
                        
                        # Update progress bar
                        pbar.update(records_count)
                        
                        logger.info(f"[Batch {batch_number}] ✓ Successfully upserted {records_count} records")
                    else:
                        self.stats[FAILED_KEY] += records_count
                        logger.error(f"[Batch {batch_number}] ✗ Failed to upsert")
                        break
                    
                    # Check if done
                    if next_offset is None:
                        logger.info("✓ Reached end of collection")
                        break
                    
                    # Move to next batch
                    offset = next_offset
                    batch_number += 1
                    
                except Exception as e:
                    logger.error(f"[Batch {batch_number}] Exception: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    self.stats[FAILED_KEY] += records_count if 'records_count' in locals() else 0
                    break
        
        # Print final report
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


def main():
    parser = argparse.ArgumentParser(
        description="Simple sequential migration from Qdrant to Endee"
    )
    
    # Source arguments
    parser.add_argument("--source_url", required=True, help="Qdrant cluster endpoint")
    parser.add_argument("--source_api_key", required=True, help="Qdrant API key")
    parser.add_argument("--source_collection", required=True, help="Qdrant collection name")
    parser.add_argument("--source_port", type=int, required=True, help="Qdrant port")
    
    # Target arguments
    parser.add_argument("--target_api_key", required=True, help="Endee API key")
    parser.add_argument("--target_collection", required=True, help="Endee index name")
    
    # Performance arguments
    parser.add_argument("--batch_size", type=int, default=DEFAULT_FETCH_BATCH_SIZE, 
                       help="Fetch batch size (default: 1000)")
    parser.add_argument("--upsert_size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE, 
                       help="Upsert batch size (default: 1000)")
    
    # Resume arguments
    parser.add_argument("--checkpoint_file", default=CHECKPOINT_FILE, 
                       help="Checkpoint file path (default: ./migration_checkpoint.json)")
    parser.add_argument("--clear_checkpoint", action="store_true", 
                       help="Clear existing checkpoint and start fresh")
    
    # Debug
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    parser.add_argument("--use_https", action="store_true", default=False,
                       help="Use HTTPS for Qdrant connection")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    
    # Create migrator
    migrator = SimpleQdrantToEndeeMigrator(
        qdrant_url=args.source_url,
        qdrant_port=args.source_port,
        qdrant_api_key=args.source_api_key,
        qdrant_collection=args.source_collection,
        endee_api_key=args.target_api_key,
        endee_index=args.target_collection,
        fetch_batch_size=args.batch_size,
        upsert_batch_size=args.upsert_size,
        checkpoint_file=args.checkpoint_file,
        use_https=args.use_https
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