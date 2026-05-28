"""
core/pipeline.py  –  Source-agnostic async producer-consumer migration pipeline.
"""
from __future__ import annotations
import asyncio, logging, signal, time
from typing import Optional
from tqdm import tqdm
from .base_source import BaseSource
from .base_target import BaseTarget
from .checkpoint  import MigrationCheckpoint
from .schema import RowSchema
logger = logging.getLogger(__name__)


class MigrationPipeline:
    def __init__(
        self,
        source: BaseSource,
        target: BaseTarget,
        checkpoint: MigrationCheckpoint,
        fetch_batch_size: int = 1000,
        max_queue_size: int = 5,
    ):
        self.source = source
        self.target = target
        self.checkpoint = checkpoint
        self.fetch_batch_size = fetch_batch_size
        self.max_queue_size = max_queue_size
        self.interrupted = False
        self._stop_event: Optional[asyncio.Event] = None
        self._schema: Optional[RowSchema] = None
        self.producer_failed = False
        self.stats = {"fetched": 0, "upserted": 0, "failed": 0, "batches_processed": 0, "start_time": None}
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.warning("\n" + "=" * 80)
        logger.warning("Received shutdown signal. Saving progress and stopping...")
        logger.warning("=" * 80)
        self.interrupted = True

    async def _producer(self, queue: asyncio.Queue):
        batch_number   = self.checkpoint.get_batch_number()
        initial_cursor = self.checkpoint.get_last_cursor()
        logger.info("PRODUCER: started")
        try:
            async for records, next_cursor in self.source.iterate_batches(
                self.fetch_batch_size, initial_cursor, self._schema
            ):
                if self.interrupted or self._stop_event.is_set():
                    logger.info("PRODUCER: stop requested — exiting loop")
                    break
                self.stats["fetched"] += len(records)
                logger.info(f"[Batch {batch_number}] Fetched {len(records)} records")
                await queue.put({"batch_number": batch_number, "records": records,
                                 "next_cursor": next_cursor, "enqueue_time": time.time()})
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: stop event set after queue.put()")
                    break
                batch_number += 1
        except Exception as e:
            import traceback
            logger.error(f"PRODUCER: exception — {e}")
            logger.error(traceback.format_exc())
            self.producer_failed = True
            self._stop_event.set()
        finally:
            await queue.put(None)
            logger.info("PRODUCER: finished")

    async def _consumer(self, queue: asyncio.Queue, pbar: tqdm):
        logger.info("CONSUMER: started")
        while not self.interrupted:
            batch = await queue.get()
            if batch is None:
                queue.task_done()
                logger.info("CONSUMER: received sentinel — exiting")
                if not self.interrupted and not self._stop_event.is_set():
                    self.checkpoint.mark_completed()
                    logger.info("CONSUMER: migration marked as completed in checkpoint")
                break
            batch_number  = batch["batch_number"]
            records       = batch["records"]
            next_cursor   = batch["next_cursor"]
            queue_wait    = time.time() - batch["enqueue_time"]
            logger.info(f"CONSUMER: upserting batch {batch_number} ({len(records)} records, wait={queue_wait:.2f}s)")
            t0 = time.time()
            success = await self.target.upsert_batch(records, self._schema)
            upsert_time = time.time() - t0
            if success:
                self.checkpoint.update(batch_number, len(records), next_cursor)
                self.stats["upserted"] += len(records)
                self.stats["batches_processed"] += 1
                pbar.update(len(records))
                queue.task_done()
                throughput = len(records) / upsert_time if upsert_time > 0 else 0
                logger.info(f"[Batch {batch_number}] ✓ {len(records)} records | upsert={upsert_time:.2f}s | {throughput:.1f} rec/s")
            else:
                self.stats["failed"] += len(records)
                logger.error(f"[Batch {batch_number}] ✗ failed after retries — stopping")
                self._stop_event.set()
                queue.task_done()
                break
        logger.info("CONSUMER: finished")

    async def _run_async(self):
        if self.checkpoint.is_completed():
            logger.warning("Previous migration is already COMPLETE.")
            logger.warning(f"Already migrated: {self.checkpoint.get_processed_count()} records.")
            logger.warning("Pass --resume (RESUME=false) to clear and re-run.")
            return
        self._stop_event = asyncio.Event()
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        logger.info(f"Bounded queue created (max_size={self.max_queue_size})")
        with tqdm(desc="Migrating records", unit="records",
                  initial=self.checkpoint.get_processed_count()) as pbar:
            await asyncio.gather(self._producer(queue), self._consumer(queue, pbar))

    def run(self):
        self.stats["start_time"] = time.time()
        logger.info("=" * 70)
        logger.info("MIGRATION PIPELINE — setup")
        logger.info("=" * 70)

        # ── step 1: connect ───────────────────────────────────────────────────
        self.source.connect()
        self.target.connect()

        # ── step 2: source builds schema (the contract for the whole pipeline) ─
        self._schema = self.source.detect_schema()

        print("schema: ", self._schema, type(self._schema))
        logger.info(f"Schema built: {[f.name for f in self._schema.fields]}")
        logger.info(f"  dimension={self._schema.dimension}, "
                    f"space={self._schema.space_type}, "
                    f"hybrid={self._schema.is_hybrid}")

        # ── step 3: target uses schema to create index + resolve slots ──────────
        self.target.setup_index(self._schema)

        # ── step 4: data flows ─────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("MIGRATION PIPELINE — data flow starting")
        logger.info(f"  Fetch batch size : {self.fetch_batch_size}")
        logger.info(f"  Queue size       : {self.max_queue_size}")
        if self.checkpoint.get_processed_count() > 0:
            logger.info(f"  Resuming from    : {self.checkpoint.get_processed_count()} records")
        logger.info("=" * 70)

        try:
            asyncio.run(self._run_async())
        finally:
            self.source.close()
            self.target.close()

        self._print_report()

    def _print_report(self):
        import sys
        duration = time.time() - self.stats["start_time"]
        logger.info("\n" + "=" * 80)
        if self.interrupted:
            logger.warning("MIGRATION INTERRUPTED — progress saved, run again to resume")
        elif self.producer_failed:
            logger.error("MIGRATION FAILED — producer could not iterate source records")
        elif self.stats["failed"] > 0:
            logger.warning("MIGRATION COMPLETED WITH ERRORS")
        else:
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Duration           : {duration:.2f}s  ({duration / 60:.2f} min)")
        logger.info(f"Total processed    : {self.checkpoint.get_processed_count()}")
        logger.info(f"Fetched  (this run): {self.stats['fetched']}")
        logger.info(f"Upserted (this run): {self.stats['upserted']}")
        logger.info(f"Failed             : {self.stats['failed']}")
        logger.info(f"Batches            : {self.stats['batches_processed']}")
        if self.stats["upserted"] > 0:
            logger.info(f"Throughput         : {self.stats['upserted']/duration:.2f} rec/s")
        logger.info("=" * 80)
        if self.producer_failed:
            sys.exit(1)