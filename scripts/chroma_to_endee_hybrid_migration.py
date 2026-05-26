"""
chroma_to_endee_hybrid_migration.py  (v2 — uses endee-model SparseModel)
=========================================================================
Migrates a ChromaDB dense collection → Endee HYBRID index.

Sparse vectors are generated using Endee's own BM25 model via the
`endee-model` package (SparseModel("endee/bm25")):

  • Documents → SparseModel.embed()        (TF × IDF + length normalisation)
  • Queries   → SparseModel.query_embed()  (IDF only — not used here,
                                            but important for search time)

Index is created with  sparse_model="endee_bm25"  so Endee applies
server-side IDF weights during search — the client only sends the
per-document TF weights.  This is the CORRECT integration with Endee's
BM25 — using any other library (rank_bm25, etc.) produces incompatible
token indices and IDF tables.

Architecture
------------
Identical producer-consumer asyncio queue pattern used in the Milvus /
Qdrant scripts — no core structural changes.

  Producer → fetches ChromaDB batches (embeddings + documents + metadata)
           → encodes sparse vectors with SparseModel.embed()  [batch]
           → puts converted Endee records on bounded asyncio.Queue

  Consumer → pulls from queue
           → upserts to Endee in parallel chunks
           → retries failures with exponential back-off (max 3 attempts)

Required parameters (script exits if any is missing)
-----------------------------------------------------
  --source_collection   SOURCE_COLLECTION   ChromaDB collection name
  --target_collection   TARGET_COLLECTION   Endee index name
  --M                   M                   HNSW M parameter
  --ef_construct        EF_CONSTRUCT        HNSW ef_construction
  --space_type          SPACE_TYPE          cosine | l2 | ip

Dependencies
------------
  pip install chromadb endee endee-model orjson tqdm python-dotenv

Usage
-----
  python chroma_to_endee_hybrid_migration.py \\
    --source_host        localhost            \\
    --source_port        8000                 \\
    --source_collection  my_chroma_collection \\
    --target_url         https://endee-cluster \\
    --target_api_key     YOUR_KEY             \\
    --target_collection  my_endee_index       \\
    --precision          int16                \\
    --space_type         cosine               \\
    --M                  16                   \\
    --ef_construct       128                  \\
    --filter_fields      category,source

  All flags can also be supplied via environment variables (or a .env file).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set

import dotenv
import orjson
from tqdm import tqdm
from constants import *

dotenv.load_dotenv()

# ── Dependency guards — give clear install hints before crashing ──────────────

try:
    import chromadb
except ImportError:
    sys.exit("ERROR: chromadb is not installed.  Run: pip install chromadb")

try:
    from endee_model import SparseModel
except ImportError:
    sys.exit("ERROR: endee-model is not installed.  Run: pip install endee-model")

try:
    from endee import Endee, Precision
    from endee.exceptions import NotFoundException
except ImportError:
    sys.exit("ERROR: endee is not installed.  Run: pip install endee")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ENDEE_V1_API        = "/api/v1"

# Checkpoint keys
PROCESSED_COUNT_KEY = "processed_count"
LAST_OFFSET_KEY     = "last_offset"       # ChromaDB uses integer offset, not cursor
BATCH_NUMBER_KEY    = "batch_number"
COMPLETED_KEY       = "completed"

# Defaults
DEFAULT_PROCESSED_COUNT = 0
DEFAULT_LAST_OFFSET     = 0
DEFAULT_BATCH_NUMBER    = 0
DEFAULT_FETCH_BATCH     = 500
DEFAULT_UPSERT_BATCH    = 100
DEFAULT_MAX_QUEUE_SIZE  = 5

# Stats keys
FETCHED_KEY            = "fetched"
UPSERTED_KEY           = "upserted"
FAILED_KEY             = "failed"
BATCHES_PROCESSED_KEY  = "batches_processed"
START_TIME_KEY         = "start_time"

# ChromaDB hnsw:space → Endee space_type
CHROMA_TO_ENDEE_SPACE: Dict[str, str] = {
    "l2":     "l2",
    "cosine": "cosine",
    "ip":     "ip",
}

# Endee Precision — probes the SDK to support both old (INT8) and new (INT8D) naming
def _build_precision_map() -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for key, attrs in {
        "float32": ["FLOAT32"],
        "float16": ["FLOAT16"],
        "int16d":  ["INT16D", "INT16"],
        "int16":   ["INT16D", "INT16"],
        "int8d":   ["INT8D",  "INT8"],
        "int8":    ["INT8D",  "INT8"],
        "binary":  ["BINARY2"],
    }.items():
        for attr in attrs:
            if hasattr(Precision, attr):
                m[key] = getattr(Precision, attr)
                break
    return m
 
PRECISION_STR_TO_ENDEE: Dict[str, Any] = _build_precision_map()
 
# Endee Precision — probes the SDK to support both old (INT8) and new (INT8D) naming
def _build_precision_map() -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for key, attrs in {
        "float32": ["FLOAT32"],
        "float16": ["FLOAT16"],
        "int16d":  ["INT16D", "INT16"],
        "int16":   ["INT16D", "INT16"],
        "int8d":   ["INT8D",  "INT8"],
        "int8":    ["INT8D",  "INT8"],
        "binary":  ["BINARY2"],
    }.items():
        for attr in attrs:
            if hasattr(Precision, attr):
                m[key] = getattr(Precision, attr)
                break
    return m
 
PRECISION_STR_TO_ENDEE: Dict[str, Any] = _build_precision_map()
 
def _default_precision() -> Any:
    for attr in ("INT8D", "INT8", "INT16D", "INT16", "FLOAT32"):
        if hasattr(Precision, attr):
            return getattr(Precision, attr)
    raise RuntimeError("No usable Precision constant found in installed endee SDK.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
 
class MigrationCheckpoint:
    """Persist migration progress for resumable runs."""
 
    def __init__(self, checkpoint_file: str = CHECKPOINT_FILE):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()
 
    def _load(self) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            PROCESSED_COUNT_KEY: DEFAULT_PROCESSED_COUNT,
            LAST_OFFSET_KEY:     DEFAULT_LAST_OFFSET,
            BATCH_NUMBER_KEY:    DEFAULT_BATCH_NUMBER,
        }
        try:
            with open(self.checkpoint_file, "rb") as f:
                data = orjson.loads(f.read())
                logger.info(
                    f"Loaded checkpoint: "
                    f"{data.get(PROCESSED_COUNT_KEY, 0):,} records already migrated"
                )
                # Qdrant/Milvus checkpoints use null for last_offset (cursor-based
                # pagination). ChromaDB uses integer offsets — reset null to 0 so
                # a stale checkpoint from another migration type doesn't crash.
                if data.get(LAST_OFFSET_KEY) is None:
                    logger.warning(
                        "Checkpoint last_offset is null (likely from a Qdrant/Milvus run). "
                        "Resetting to 0 for ChromaDB integer-offset pagination."
                    )
                    data[LAST_OFFSET_KEY] = DEFAULT_LAST_OFFSET
                return data
        except FileNotFoundError:
            logger.info("No checkpoint found — starting fresh")
            return default
        except Exception as exc:
            logger.warning(f"Could not load checkpoint ({exc}) — starting fresh")
            return default
 
    def save(self) -> None:
        try:
            dirpath = os.path.dirname(self.checkpoint_file)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(self.checkpoint_file, "wb") as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
        except Exception as exc:
            logger.error(f"Failed to save checkpoint: {exc}")
 
    def update(
        self,
        batch_number:  int,
        records_count: int,
        next_offset:   Optional[int],
    ) -> None:
        self.data[PROCESSED_COUNT_KEY] += records_count
        self.data[BATCH_NUMBER_KEY]     = batch_number
        if next_offset is not None:
            self.data[LAST_OFFSET_KEY] = next_offset
        else:
            self.data[COMPLETED_KEY] = True
        self.save()
 
    def mark_completed(self) -> None:
        """Mark migration as fully complete — called by consumer on clean end-of-data."""
        self.data[COMPLETED_KEY] = True
        self.save()
 
    def is_completed(self) -> bool:
        return bool(self.data.get(COMPLETED_KEY, False))
 
    def get_last_offset(self) -> int:
        # Qdrant/Milvus checkpoints store last_offset as null (None) — handle gracefully
        val = self.data.get(LAST_OFFSET_KEY, DEFAULT_LAST_OFFSET)
        return int(val) if val is not None else DEFAULT_LAST_OFFSET
 
    def get_batch_number(self) -> int:
        val = self.data.get(BATCH_NUMBER_KEY, DEFAULT_BATCH_NUMBER)
        return int(val) if val is not None else DEFAULT_BATCH_NUMBER
 
    def get_processed_count(self) -> int:
        val = self.data.get(PROCESSED_COUNT_KEY, DEFAULT_PROCESSED_COUNT)
        return int(val) if val is not None else DEFAULT_PROCESSED_COUNT
 
    def clear(self) -> None:
        self.data = {
            PROCESSED_COUNT_KEY: DEFAULT_PROCESSED_COUNT,
            LAST_OFFSET_KEY:     DEFAULT_LAST_OFFSET,
            BATCH_NUMBER_KEY:    DEFAULT_BATCH_NUMBER,
            COMPLETED_KEY:       False,
        }
        self.save()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Migrator
# ─────────────────────────────────────────────────────────────────────────────
 
class ChromaToEndeeHybridMigrator:
    """
    Migrate ChromaDB dense collection → Endee hybrid index.
 
    Sparse vectors use endee-model SparseModel("endee/bm25"):
      • .embed() for documents  — TF × IDF + length normalisation
    Index uses sparse_model="endee_bm25" so server applies IDF at search time.
 
    Parameters (all required except those with defaults)
    ----------------------------------------------------
    source_collection      ChromaDB collection name           REQUIRED
    target_collection      Endee index name                   REQUIRED
    space_type             cosine | l2 | ip                   REQUIRED
    M                      HNSW M parameter                   REQUIRED
    ef_construct           HNSW ef_construction               REQUIRED
    source_host            ChromaDB host          default localhost
    source_port            ChromaDB port          default 8000
    source_api_key         Chroma API key (cloud) default ""
    source_path            PersistentClient path  default None
    target_url             Endee cluster URL      default ""
    target_api_key         Endee API key          default ""
    precision              Endee Precision enum   default INT8D / INT8
    filter_fields          Comma-sep meta keys → Endee filter payload
    fetch_batch_size       Records per .get() call
    upsert_batch_size      Records per .upsert() chunk
    max_queue_size         asyncio.Queue depth
    store_document_in_meta Store ChromaDB text in meta["document"]
    """
 
    def __init__(
        self,
        source_collection:      str,
        target_collection:      str,
        space_type:             str,
        M:                      int,
        ef_construct:           int,
        source_host:            str           = "localhost",
        source_port:            int           = 8000,
        source_api_key:         str           = "",
        source_path:            Optional[str] = None,
        target_url:             str           = "",
        target_api_key:         str           = "",
        precision:              Any           = None,
        filter_fields:          str           = "",
        fetch_batch_size:       int           = DEFAULT_FETCH_BATCH,
        upsert_batch_size:      int           = DEFAULT_UPSERT_BATCH,
        max_queue_size:         int           = DEFAULT_MAX_QUEUE_SIZE,
        checkpoint_file:        str           = CHECKPOINT_FILE,
        store_document_in_meta: bool          = True,
    ):
        self.source_collection      = source_collection
        self.target_collection      = target_collection
        self.space_type             = space_type
        self.M                      = M
        self.ef_construct           = ef_construct
        self.source_host            = source_host
        self.source_port            = source_port
        self.source_api_key         = source_api_key
        self.source_path            = source_path
        self.target_url             = target_url
        self.target_api_key         = target_api_key
        self.precision              = precision or _default_precision()
        self.filter_fields: Set[str] = (
            {f.strip() for f in filter_fields.split(",") if f.strip()}
            if filter_fields else set()
        )
        self.fetch_batch_size       = fetch_batch_size
        self.upsert_batch_size      = upsert_batch_size
        self.max_queue_size         = max_queue_size
        self.store_document_in_meta = store_document_in_meta
 
        # Runtime state
        self.chroma_client:     Any                     = None
        self.chroma_collection: Any                     = None
        self.endee_client:      Any                     = None
        self.endee_index:       Any                     = None
        self.sparse_model:      Optional[SparseModel]   = None
        self.checkpoint         = MigrationCheckpoint(checkpoint_file)
        self.interrupted        = False
        self._stop_event:       Optional[asyncio.Event] = None
 
        self.stats: Dict[str, Any] = {
            FETCHED_KEY:           0,
            UPSERTED_KEY:          0,
            FAILED_KEY:            0,
            BATCHES_PROCESSED_KEY: 0,
            START_TIME_KEY:        None,
            "producer_failed":     False,   # set True when producer exits via exception
        }
 
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
 
    # ── Signal ────────────────────────────────────────────────────────────────
 
    def _signal_handler(self, signum, frame):
        logger.warning("\n" + "=" * 80)
        logger.warning("Shutdown signal received — saving progress and stopping.")
        logger.warning("=" * 80)
        self.interrupted = True
 
    # ── Connections ───────────────────────────────────────────────────────────
 
    def connect_chroma(self) -> None:
        logger.info("Connecting to ChromaDB...")
        if self.source_path:
            logger.info(f"  Mode: PersistentClient  path={self.source_path}")
            self.chroma_client = chromadb.PersistentClient(path=self.source_path)
        else:
            kwargs: Dict[str, Any] = {"host": self.source_host, "port": self.source_port}
            if self.source_api_key:
                kwargs["headers"] = {"X-Chroma-Token": self.source_api_key}
            logger.info(
                f"  Mode: HttpClient  host={self.source_host}  port={self.source_port}"
            )
            self.chroma_client = chromadb.HttpClient(**kwargs)
 
        self.chroma_collection = self.chroma_client.get_collection(
            name=self.source_collection
        )
        total = self.chroma_collection.count()
        logger.info(
            f"✓ Connected to ChromaDB | collection='{self.source_collection}' | "
            f"total={total:,}"
        )
 
    def connect_endee(self) -> None:
        logger.info("Connecting to Endee...")
        self.endee_client = Endee(token=self.target_api_key)
        if self.target_url:
            url = urllib.parse.urljoin(self.target_url, ENDEE_V1_API)
            self.endee_client.set_base_url(url)
            logger.info(f"  Base URL: {url}")
        logger.info("✓ Connected to Endee")
 
    def load_sparse_model(self) -> None:
        """
        Load Endee's BM25 model (endee/bm25) from the endee-model package.
        This is the ONLY correct sparse encoder for indexes created with
        sparse_model="endee_bm25" — Endee's server uses the same vocabulary
        and IDF table.
        """
        logger.info("Loading Endee BM25 sparse model  (endee/bm25)...")
        self.sparse_model = SparseModel(model_name="endee/bm25")
        logger.info("✓ Sparse model ready")
 
    # ── Collection inspection ─────────────────────────────────────────────────
 
    def get_collection_dimension(self) -> int:
        """
        Detect embedding dimension by peeking at the first record.
        ChromaDB does not expose dimension in collection metadata.
        """
        sample = self.chroma_collection.get(limit=1, include=["embeddings"])
        if not sample["ids"]:
            logger.error("ChromaDB collection is empty — nothing to migrate.")
            sys.exit(1)
        embedding = sample["embeddings"][0]
        if embedding is None:
            logger.error(
                "First record has no stored embedding. "
                "Ensure embeddings are pre-computed before running migration."
            )
            sys.exit(1)
        dim = len(embedding)
        logger.info(f"  Dimension: auto-detected → {dim}")
        return dim
 
    def validate_filter_fields(self) -> None:
        if not self.filter_fields:
            return
        sample = self.chroma_collection.get(limit=1, include=["metadatas"])
        if sample["metadatas"] and sample["metadatas"][0]:
            available = set(sample["metadatas"][0].keys())
            invalid   = self.filter_fields - available
            if invalid:
                logger.warning(
                    f"filter_fields {invalid} not found in first record metadata. "
                    f"Available: {available}. May appear in other records — continuing."
                )
 
    # ── Endee index ───────────────────────────────────────────────────────────
 
    def get_or_create_endee_index(self, dimension: int) -> None:
        """Get existing index or create a new hybrid index."""
        try:
            self.endee_index = self.endee_client.get_index(self.target_collection)
            logger.info(f"✓ Index already exists: '{self.target_collection}'")
            return
        except NotFoundException:
            pass
 
        logger.info(
            f"Creating HYBRID index '{self.target_collection}' | "
            f"dim={dimension}  sparse_model=endee_bm25  "
            f"space={self.space_type}  M={self.M}  ef_con={self.ef_construct}  "
            f"precision={self.precision}"
        )
        # sparse_model="endee_bm25" tells Endee to apply its server-side IDF
        # weights during search — the upsert carries TF values only.
        self.endee_client.create_index(
            name         = self.target_collection,
            dimension    = dimension,
            sparse_model = "endee_bm25",
            space_type   = self.space_type,
            M            = self.M,
            ef_con       = self.ef_construct,
            precision    = self.precision,
        )
        self.endee_index = self.endee_client.get_index(self.target_collection)
        logger.info(f"✓ Created hybrid index: '{self.target_collection}'")
 
    # ── Record conversion ─────────────────────────────────────────────────────
 
    def _convert_batch(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert one ChromaDB .get() result (columnar) into a list of
        Endee hybrid record dicts.
 
        Sparse encoding
        ---------------
        Calls SparseModel.embed(batch_of_texts) — this is the document-side
        BM25 function (TF × IDF + length normalisation). It processes the
        whole batch at once for efficiency.
 
        From Endee docs: "Always use .embed() for documents and .query_embed()
        for queries. Mixing them produces incorrect BM25 scores."
 
        Endee record schema
        -------------------
        {
          "id":             str,
          "vector":         list[float],   # dense embedding from ChromaDB
          "sparse_indices": list[int],     # BM25 token IDs  (endee/bm25 vocab)
          "sparse_values":  list[float],   # BM25 TF weights
          "filter":         dict,          # fields in filter_fields
          "meta":           dict,          # remaining metadata + optional document text
        }
        """
        ids        = result.get("ids",       [])
        embeddings = result.get("embeddings", None)
        documents  = result.get("documents",  None)
        metadatas  = result.get("metadatas",  None)
 
        # Build doc_texts for BM25 encoding.
        # Use `is not None` — ChromaDB returns embeddings as a numpy array, and
        # `if numpy_array` raises ValueError when it has >1 element.
        doc_texts = []
        for i in range(len(ids)):
            if documents is not None and i < len(documents) and documents[i]:
                doc_texts.append(documents[i])
            else:
                doc_texts.append("")
        if all(t == "" for t in doc_texts):
            logger.error("=" * 70)
            logger.error(
                "FATAL: No document text found in this batch — "
                "cannot generate BM25 sparse vectors."
            )
            logger.error(
                "ChromaDB collection was likely created with embeddings only "
                "(no 'documents' stored). Hybrid migration requires document "
                "text to encode sparse vectors."
            )
            logger.error(
                "Fix: re-ingest your data into ChromaDB with the 'documents' "
                "field populated, or switch to a dense-only migration."
            )
            logger.error("=" * 70)
            raise RuntimeError(
                "No document text available — BM25 sparse encoding aborted. "
                "See FATAL error above."
            )
        sparse_embs = list(self.sparse_model.embed(doc_texts))
 
        records: List[Dict[str, Any]] = []
 
        for i, doc_id in enumerate(ids):
            # Always use `is not None` checks — never bare `if array`
            embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else None
            document  = documents[i]  if documents  is not None and i < len(documents)  else ""
            metadata  = metadatas[i]  if metadatas  is not None and i < len(metadatas)  else {}
            sparse_emb = sparse_embs[i]
 
            if embedding is None:
                logger.warning(f"  Record '{doc_id}' has no embedding — skipping.")
                continue
 
            # Route metadata fields into filter vs meta
            filter_data: Dict[str, Any] = {}
            meta_data:   Dict[str, Any] = {}
            for key, val in (metadata or {}).items():
                if key in self.filter_fields:
                    filter_data[key] = val
                else:
                    meta_data[key] = val
 
            if self.store_document_in_meta and document:
                meta_data["document"] = document
 
            rec: Dict[str, Any] = {
                "id":     str(doc_id),
                "vector": list(embedding),
            }
            # Attach sparse fields only when the model produced non-zero tokens
            if len(sparse_emb.indices) > 0:
                rec["sparse_indices"] = sparse_emb.indices.tolist()
                rec["sparse_values"]  = sparse_emb.values.tolist()
 
            rec["filter"] = filter_data
            rec["meta"]   = meta_data
 
            records.append(rec)
 
        return records
 
    # ── Async producer ────────────────────────────────────────────────────────
 
    async def _async_producer(self, queue: asyncio.Queue) -> None:
        """
        Fetches ChromaDB pages starting from checkpoint offset, converts
        each page to Endee records (including BM25 sparse encoding), then
        puts them on the bounded queue for the consumer to upsert.
        """
        offset       = self.checkpoint.get_last_offset()
        batch_number = self.checkpoint.get_batch_number()
        loop         = asyncio.get_running_loop()
 
        logger.info(f"PRODUCER STARTED  offset={offset}  batch={batch_number}")
 
        while not self.interrupted and not self._stop_event.is_set():
            try:
                logger.info(
                    f"FETCHING batch {batch_number} from ChromaDB  offset={offset}"
                )
                fetch_start = time.time()
 
                # ChromaDB .get() is synchronous — keep event loop free
                result = await loop.run_in_executor(
                    None,
                    lambda o=offset: self.chroma_collection.get(
                        limit   = self.fetch_batch_size,
                        offset  = o,
                        include = ["embeddings", "documents", "metadatas"],
                    ),
                )
                fetch_time = time.time() - fetch_start
 
                ids = result.get("ids", [])
                if not ids:
                    logger.info("PRODUCER: No more records — signalling end.")
                    await queue.put(None)
                    break
 
                # BM25 encoding is CPU-bound but fast per batch — run inline
                transform_start = time.time()
                records         = self._convert_batch(result)
                transform_time  = time.time() - transform_start
 
                self.stats[FETCHED_KEY] += len(ids)
 
                next_offset = offset + len(ids)
                is_last     = len(ids) < self.fetch_batch_size   # partial page = end
 
                logger.info(
                    f"[Batch {batch_number}] Fetched {len(ids)} records | "
                    f"fetch={fetch_time:.2f}s | transform+BM25={transform_time:.2f}s"
                )
 
                if self.interrupted:
                    await queue.put(None)
                    break
 
                await queue.put({
                    "batch_number":   batch_number,
                    "records":        records,
                    "next_offset":    None if is_last else next_offset,
                    "fetch_time":     fetch_time,
                    "transform_time": transform_time,
                    "enqueue_time":   time.time(),
                })
 
                if self._stop_event.is_set():
                    logger.warning("PRODUCER: Stop event set — exiting.")
                    break
 
                if queue.qsize() >= self.max_queue_size:
                    logger.warning(
                        f"Queue at capacity ({queue.qsize()}) — "
                        "producer will block until consumer catches up."
                    )
 
                offset        = next_offset
                batch_number += 1
 
                if is_last:
                    await queue.put(None)
                    break
 
            except Exception as exc:
                import traceback
                logger.error(f"[Producer] Exception: {exc}")
                logger.error(traceback.format_exc())
                self.stats["producer_failed"] = True
                self._stop_event.set()
                await queue.put(None)
                break
 
        logger.info("PRODUCER FINISHED")
 
    # ── Async consumer ────────────────────────────────────────────────────────
 
    async def _upsert_chunk(self, chunk: List[Dict[str, Any]]) -> None:
        """Upsert one chunk to Endee (sync SDK wrapped in thread executor)."""
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.endee_index.upsert(chunk),
        )
        if result is False:
            raise RuntimeError(
                f"Endee upsert returned falsy for chunk of {len(chunk)} records"
            )
        logger.debug(f"  Upserted chunk of {len(chunk)} records")
 
    async def _upsert_records(self, records: List[Dict[str, Any]]) -> bool:
        """
        Split records into upsert_batch_size chunks, fire all in parallel,
        then retry failures with exponential back-off (1s → 2s → 4s, max 3 attempts).
        Returns True on full success, False if any chunk exhausts retries.
        """
        chunks = [
            records[i: i + self.upsert_batch_size]
            for i in range(0, len(records), self.upsert_batch_size)
        ]
 
        results = await asyncio.gather(
            *[self._upsert_chunk(c) for c in chunks],
            return_exceptions=True,
        )
 
        failed = [chunks[i] for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed:
            logger.warning(f"  {len(failed)}/{len(chunks)} chunks failed — retrying...")
 
        while failed:
            chunk   = failed.pop(0)
            retried = False
            for attempt in range(3):
                try:
                    await self._upsert_chunk(chunk)
                    retried = True
                    logger.info(f"  Retry {attempt + 1} succeeded ({len(chunk)} records)")
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        f"  Retry {attempt + 1}/3 failed: {exc}. Waiting {wait}s..."
                    )
                    await asyncio.sleep(wait)
            if not retried:
                logger.error(f"  Chunk of {len(chunk)} records failed after 3 retries.")
                return False
 
        return True
 
    async def _async_consumer(self, queue: asyncio.Queue, pbar: tqdm) -> None:
        logger.info("CONSUMER STARTED")
 
        while not self.interrupted:
            try:
                batch = await queue.get()
 
                if batch is None:
                    logger.info("CONSUMER: Received end signal.")
                    queue.task_done()
                    # Only mark completed if producer finished naturally (not via error/interrupt)
                    if not self.interrupted and not self._stop_event.is_set():
                        self.checkpoint.mark_completed()
                        logger.info("CONSUMER: Migration marked as completed in checkpoint.")
                    break
 
                batch_number   = batch["batch_number"]
                records        = batch["records"]
                next_offset    = batch["next_offset"]
                fetch_time     = batch.get("fetch_time",     0)
                transform_time = batch.get("transform_time", 0)
                enqueue_time   = batch.get("enqueue_time",   time.time())
                records_count  = len(records)
                queue_wait     = time.time() - enqueue_time
 
                logger.info(
                    f"UPSERTING batch {batch_number} → Endee ({records_count} records)"
                )
                upsert_start = time.time()
                success      = await self._upsert_records(records)
                upsert_time  = time.time() - upsert_start
 
                if success:
                    self.checkpoint.update(batch_number, records_count, next_offset)
                    self.stats[UPSERTED_KEY]          += records_count
                    self.stats[BATCHES_PROCESSED_KEY] += 1
                    pbar.update(records_count)
                    queue.task_done()
 
                    throughput = records_count / upsert_time if upsert_time > 0 else 0
                    total_time = fetch_time + transform_time + queue_wait + upsert_time
                    logger.info(
                        f"[Batch {batch_number}]  {records_count} records | "
                        f"fetch={fetch_time:.2f}s | "
                        f"transform+BM25={transform_time:.2f}s | "
                        f"queue_wait={queue_wait:.2f}s | "
                        f"upsert={upsert_time:.2f}s | "
                        f"total={total_time:.2f}s | "
                        f"throughput={throughput:.1f} rec/s"
                    )
                else:
                    self.stats[FAILED_KEY] += records_count
                    logger.error(
                        f"CONSUMER: Failed to upsert batch {batch_number} — stopping."
                    )
                    self._stop_event.set()
                    queue.task_done()
                    break
 
            except Exception as exc:
                import traceback
                logger.error(f"[Consumer] Exception: {exc}")
                logger.error(traceback.format_exc())
                break
 
        logger.info("CONSUMER FINISHED")
 
    # ── Orchestration ─────────────────────────────────────────────────────────
 
    async def async_migrate(self) -> None:
        self.stats[START_TIME_KEY] = time.time()
 
        if self.checkpoint.is_completed():
            logger.info(
                f"Migration already completed — "
                f"{self.checkpoint.get_processed_count():,} records migrated previously. "
                "Set RESUME=false in your .env (or pass --resume) to start fresh."
            )
            return
 
        logger.info("=" * 80)
        logger.info("CHROMA → ENDEE HYBRID MIGRATION")
        logger.info("=" * 80)
        logger.info(f"Source       : collection='{self.source_collection}'")
        logger.info(f"Target       : index='{self.target_collection}'")
        logger.info(f"Space type   : {self.space_type}")
        logger.info(f"M            : {self.M}")
        logger.info(f"ef_construct : {self.ef_construct}")
        logger.info(f"Sparse model : endee/bm25  (endee-model)")
        logger.info(f"Fetch batch  : {self.fetch_batch_size}")
        logger.info(f"Upsert chunk : {self.upsert_batch_size}")
        logger.info(f"Queue depth  : {self.max_queue_size}")
        logger.info("=" * 80)
 
        if self.checkpoint.get_processed_count() > 0:
            logger.info(
                f"RESUMING: {self.checkpoint.get_processed_count():,} records already migrated, "
                f"offset={self.checkpoint.get_last_offset()}"
            )
 
        # Connections
        self.connect_chroma()
        self.connect_endee()
 
        # Inspect collection
        dimension = self.get_collection_dimension()
        self.validate_filter_fields()
 
        # Load Endee BM25 model
        self.load_sparse_model()
 
        # Create / get Endee hybrid index
        self.get_or_create_endee_index(dimension)
        if self.endee_index is None:
            raise RuntimeError("Endee index not initialised — aborting.")
 
        # Async producer-consumer pipeline
        total = self.chroma_collection.count()
        queue = asyncio.Queue(maxsize=self.max_queue_size)
        logger.info(f"Bounded queue created  max_size={self.max_queue_size}")
        logger.info("\nStarting migration pipeline...")
        logger.info("=" * 80)
 
        with tqdm(
            desc    = "Migrating records",
            unit    = "records",
            total   = total,
            initial = self.checkpoint.get_processed_count(),
        ) as pbar:
            self._stop_event = asyncio.Event()
            await asyncio.gather(
                self._async_producer(queue),
                self._async_consumer(queue, pbar),
            )
 
        logger.info("ASYNC MIGRATION COMPLETED")
        self._print_final_report()
 
    def _print_final_report(self) -> None:
        duration        = time.time() - self.stats[START_TIME_KEY]
        producer_failed = self.stats.get("producer_failed", False)
 
        logger.info("\n" + "=" * 80)
        if self.interrupted:
            logger.warning("MIGRATION INTERRUPTED — progress saved; re-run to resume.")
        elif producer_failed:
            logger.error(
                "MIGRATION FAILED — producer exited with an error. "
                "0 records were migrated. Check the traceback above."
            )
        elif self.stats[FAILED_KEY] > 0:
            logger.warning("MIGRATION COMPLETED WITH ERRORS — check logs.")
        elif self.stats[UPSERTED_KEY] == 0 and self.checkpoint.get_processed_count() > 0:
            # Previous run already migrated everything — this run found nothing new.
            # This is normal when re-running without RESUME=false.
            logger.info(
                f"NOTHING TO MIGRATE — checkpoint shows "
                f"{self.checkpoint.get_processed_count():,} records already migrated. "
                "Set RESUME=false in your .env to start fresh."
            )
        elif self.stats[UPSERTED_KEY] == 0 and not self.checkpoint.is_completed():
            logger.error(
                "MIGRATION COMPLETED WITH 0 RECORDS — nothing was upserted. "
                "Check producer logs above for errors."
            )
        else:
            logger.info("MIGRATION COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 80)
        logger.info(f"Duration              : {duration:.1f}s ({duration / 60:.1f} min)")
        logger.info(f"Total already migrated: {self.checkpoint.get_processed_count():,}")
        logger.info(f"Fetched this run      : {self.stats[FETCHED_KEY]:,}")
        logger.info(f"Upserted this run     : {self.stats[UPSERTED_KEY]:,}")
        logger.info(f"Failed                : {self.stats[FAILED_KEY]:,}")
        logger.info(f"Batches processed     : {self.stats[BATCHES_PROCESSED_KEY]:,}")
        if self.stats[UPSERTED_KEY] > 0 and duration > 0:
            logger.info(
                f"Throughput            : "
                f"{self.stats[UPSERTED_KEY] / duration:.1f} rec/s"
            )
        logger.info("=" * 80)
 
        # Exit with non-zero code on any failure so Docker / CI detects it
        if producer_failed or (self.stats[FAILED_KEY] > 0 and self.stats[UPSERTED_KEY] == 0):
            sys.exit(1)
 
    def migrate(self) -> None:
        asyncio.run(self.async_migrate())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate ChromaDB dense collection → Endee HYBRID index.\n"
            "Sparse vectors: endee-model SparseModel('endee/bm25').\n\n"
            "M, ef_construct, and space_type are REQUIRED — no defaults."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
 
    # ChromaDB source
    src = parser.add_argument_group("ChromaDB source")
    src.add_argument("--source_url",
                     default=os.getenv("SOURCE_URL", ""),
                     help="Full ChromaDB URL e.g. http://3.110.158.42  "
                          "(host and port parsed automatically; overrides --source_host)")
    src.add_argument("--source_host",
                     default=os.getenv("SOURCE_HOST", ""),
                     help="ChromaDB hostname — used only when --source_url is not set")
    src.add_argument("--source_port",
                     type=int, default=int(os.getenv("SOURCE_PORT", 8000)))
    src.add_argument("--source_api_key",
                     default=os.getenv("SOURCE_API_KEY", ""))
    src.add_argument("--source_collection",
                     default=os.getenv("SOURCE_COLLECTION", ""),
                     help="ChromaDB collection name  [REQUIRED]")
    src.add_argument("--source_path",
                     default=os.getenv("SOURCE_PATH", None),
                     help="Local PersistentClient path (overrides source_url / host+port)")
 
    # Endee target
    tgt = parser.add_argument_group("Endee target")
    tgt.add_argument("--target_url",
                     default=os.getenv("TARGET_URL", ""))
    tgt.add_argument("--target_api_key",
                     default=os.getenv("TARGET_API_KEY", ""))
    tgt.add_argument("--target_collection",
                     default=os.getenv("TARGET_COLLECTION", ""),
                     help="Endee index name  [REQUIRED]")
 
    # Index configuration — all REQUIRED
    idx = parser.add_argument_group("Index configuration  [all REQUIRED]")
    idx.add_argument("--space_type",
                     default=os.getenv("SPACE_TYPE", None),
                     help="cosine | l2 | ip  [REQUIRED]")
    idx.add_argument("--M", type=int,
                     default=int(os.getenv("M")) if os.getenv("M") else None,
                     help="HNSW M parameter  [REQUIRED]")
    idx.add_argument("--ef_construct", type=int,
                     default=int(os.getenv("EF_CONSTRUCT")) if os.getenv("EF_CONSTRUCT") else None,
                     help="HNSW ef_construction  [REQUIRED]")
    idx.add_argument("--precision",
                     default=os.getenv("PRECISION", "int8d"),
                     help=f"Vector precision. Valid: {sorted(PRECISION_STR_TO_ENDEE.keys())}")
    idx.add_argument("--filter_fields",
                     default=os.getenv("FILTER_FIELDS", ""),
                     help="Comma-sep metadata keys routed to Endee filter payload")
 
    # Performance
    perf = parser.add_argument_group("Performance")
    perf.add_argument("--batch_size",     type=int,
                      default=int(os.getenv("BATCH_SIZE",     DEFAULT_FETCH_BATCH)))
    perf.add_argument("--upsert_size",    type=int,
                      default=int(os.getenv("UPSERT_SIZE",    DEFAULT_UPSERT_BATCH)))
    perf.add_argument("--max_queue_size", type=int,
                      default=int(os.getenv("MAX_QUEUE_SIZE", DEFAULT_MAX_QUEUE_SIZE)))
 
    # Resume / checkpoint
    res = parser.add_argument_group("Resume & checkpoint")
    res.add_argument("--checkpoint_file",
                     default=os.getenv("CHECKPOINT_FILE", CHECKPOINT_FILE))
    res.add_argument("--resume", action="store_true",
                     default=os.getenv("RESUME", "true").lower() == "false",
                     help="Set RESUME=false (env) or pass --resume to clear the checkpoint "
                          "and start fresh.  Matches Milvus/Qdrant script behaviour.")
    res.add_argument("--clear_checkpoint", action="store_true",
                     default=os.getenv("CLEAR_CHECKPOINT", "false").lower() == "true",
                     help="Alias for --resume; also clears checkpoint and starts fresh")
    res.add_argument("--no_store_document", action="store_true",
                     default=os.getenv("NO_STORE_DOCUMENT", "false").lower() == "true",
                     help="Do NOT store document text in Endee meta.document")
 
    parser.add_argument("--debug", action="store_true",
                        default=os.getenv("DEBUG", "false").lower() == "true")
 
    args = parser.parse_args()
 
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
 
    # ── Validate ALL required parameters — collect every error, then exit ─────
    errors: List[str] = []
 
    source_col = (args.source_collection or "").strip()
    target_col = (args.target_collection or "").strip()
 
    if not source_col:
        errors.append(
            "--source_collection  /  SOURCE_COLLECTION  is required."
        )
    if not target_col:
        errors.append(
            "--target_collection  /  TARGET_COLLECTION  is required."
        )
 
    space_type = (args.space_type or "").strip().lower()
    if not space_type:
        errors.append(
            "--space_type  /  SPACE_TYPE  is required.  "
            "Valid values: cosine | l2 | ip"
        )
    elif space_type not in {"cosine", "l2", "ip"}:
        errors.append(
            f"--space_type '{args.space_type}' is invalid.  "
            "Valid values: cosine | l2 | ip"
        )
 
    if args.M is None:
        errors.append(
            "--M  /  M  is required.  Example: --M 16"
        )
    elif args.M <= 0:
        errors.append(f"--M must be a positive integer, got: {args.M}")
 
    if args.ef_construct is None:
        errors.append(
            "--ef_construct  /  EF_CONSTRUCT  is required.  "
            "Example: --ef_construct 128"
        )
    elif args.ef_construct <= 0:
        errors.append(
            f"--ef_construct must be a positive integer, got: {args.ef_construct}"
        )
 
    if errors:
        logger.error("=" * 70)
        logger.error(
            "MISSING OR INVALID REQUIRED PARAMETERS — migration cannot start:"
        )
        for msg in errors:
            logger.error(f"  • {msg}")
        logger.error("=" * 70)
        sys.exit(1)
 
    # ── Resolve precision ─────────────────────────────────────────────────────
    precision_key = (args.precision or "int8d").lower().strip()
    if precision_key not in PRECISION_STR_TO_ENDEE:
        logger.error(
            f"Invalid --precision '{args.precision}'. "
            f"Valid values: {sorted(PRECISION_STR_TO_ENDEE.keys())}"
        )
        sys.exit(1)
 
    # ── Resolve source host/port from SOURCE_URL if provided ─────────────────
    # Priority: source_path > source_url > source_host
    # Matches how the Qdrant/Milvus scripts handle SOURCE_URL.
    resolved_host = "localhost"
    resolved_port = args.source_port  # default 8000 or from SOURCE_PORT env
 
    if args.source_url and args.source_url.strip():
        # Parse full URL like "http://3.110.158.42" or "http://3.110.158.42:9000"
        parsed = urllib.parse.urlparse(args.source_url.strip())
        if parsed.hostname:
            resolved_host = parsed.hostname
        if parsed.port:
            resolved_port = parsed.port
        logger.info(
            f"Resolved ChromaDB host from SOURCE_URL: "
            f"{resolved_host}:{resolved_port}"
        )
    elif args.source_host and args.source_host.strip():
        resolved_host = args.source_host.strip()
        logger.info(
            f"Resolved ChromaDB host from SOURCE_HOST: "
            f"{resolved_host}:{resolved_port}"
        )
    else:
        logger.warning(
            "Neither SOURCE_URL nor SOURCE_HOST is set — "
            "defaulting to localhost. This is likely wrong for Docker deployments."
        )
 
    # ── Build migrator ────────────────────────────────────────────────────────
    migrator = ChromaToEndeeHybridMigrator(
        source_collection      = source_col,
        target_collection      = target_col,
        space_type             = space_type,
        M                      = args.M,
        ef_construct           = args.ef_construct,
        source_host            = resolved_host,
        source_port            = resolved_port,
        source_api_key         = args.source_api_key,
        source_path            = args.source_path,
        target_url             = args.target_url,
        target_api_key         = args.target_api_key,
        precision              = PRECISION_STR_TO_ENDEE[precision_key],
        filter_fields          = args.filter_fields,
        fetch_batch_size       = args.batch_size,
        upsert_batch_size      = args.upsert_size,
        max_queue_size         = args.max_queue_size,
        checkpoint_file        = args.checkpoint_file,
        store_document_in_meta = not args.no_store_document,
    )
 
    if args.resume or args.clear_checkpoint:
        logger.info("Clearing checkpoint for fresh start...")
        migrator.checkpoint.clear()
 
    try:
        migrator.migrate()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Progress has been saved.")
    except Exception as exc:
        import traceback
        logger.error(f"Migration failed: {exc}")
        logger.error(traceback.format_exc())
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()