"""
sources/chroma_source.py
──────────────────────────────────────────────────────────────────────────────
ChromaDB source connectors (dense-only and hybrid).

ChromaDenseSource  — migrates dense vectors only.
ChromaHybridSource — generates sparse vectors on the source side from
                     document text using endee-model SparseModel, then
                     migrates dense + generated sparse to a hybrid Endee index.

Both share a common ChromaBaseSource that handles:
  - Connection (HttpClient or PersistentClient)
  - Dimension auto-detection (peek at first record)
  - Schema building -> RowSchema
  - Offset-based async batch iteration with retry

Sparse generation (ChromaHybridSource only)
───────────────────────────────────────────
  SparseModel("endee/bm25").embed(doc_texts)  — TF x IDF + length normalisation
  Called per batch in _convert_records().

  Requires document text to be stored in the ChromaDB collection.
  The sink creates the Endee index with  sparse_model=args.sparse_model
  (e.g. "endee_bm25") so the server applies IDF weights at search time.

Cursor format
─────────────
ChromaDB uses integer offset pagination.
Cursor = integer count of records already processed.
initial_cursor=0 means start from the beginning.

RowSchema slot layout
──────────────────────
  Dense:
    SLOT 0  -> ID         (STRING)
    SLOT 1  -> DENSE_VECTOR
    SLOT 2  -> JSON payload (metadata)

  Hybrid:
    SLOT 0  -> ID         (STRING)
    SLOT 1  -> DENSE_VECTOR
    SLOT 2  -> SPARSE_VECTOR  (generated from document text)
    SLOT 3  -> JSON payload   (metadata + optionally document text)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import chromadb
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from core.base_source import BaseSource
from core.schema import FieldRole, FieldSchema, FieldType, MigrationRow, RowSchema

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared ChromaDB base
# ══════════════════════════════════════════════════════════════════════════════

class ChromaBaseSource(BaseSource):
    """
    Internal base class — all ChromaDB plumbing lives here.
      - detect_schema() builds and returns RowSchema
      - _convert_records() fills MigrationRow slots positionally
      - Subclasses override _build_sparse_field() to opt into sparse generation

    ChromaDB collections carry no typed field metadata, so all payload
    fields are bundled into a single JSON slot (same as Qdrant).
    """

    def __init__(
        self,
        url:                    str,
        collection:             str,
        api_key:                str  = "",
        source_path:            Optional[str] = None,
        store_document_in_meta: bool = True,
    ):
        self.url                    = url
        self.collection             = collection
        self.api_key                = api_key
        self.source_path            = source_path
        self.store_document_in_meta = store_document_in_meta

        self.chroma_client:     Any                  = None
        self.chroma_collection: Any                  = None
        self._schema:           Optional[RowSchema]  = None

        # slot positions — resolved once in detect_schema()
        self._dense_slot:   int = -1
        self._sparse_slot:  int = -1
        self._payload_slot: int = -1

    # ── connect ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to ChromaDB...")

        try:
            if self.source_path:
                logger.info(f"  Mode: PersistentClient  path={self.source_path}")
                self.chroma_client = chromadb.PersistentClient(path=self.source_path)
            else:
                parsed = urllib.parse.urlparse(self.url)
                host   = parsed.hostname or self.url
                port   = parsed.port or 8000
                kwargs: Dict[str, Any] = {"host": host, "port": port}
                if self.api_key:
                    kwargs["headers"] = {"X-Chroma-Token": self.api_key}
                logger.info(f"  Mode: HttpClient  host={host}  port={port}")
                self.chroma_client = chromadb.HttpClient(**kwargs)

            self.chroma_collection = self.chroma_client.get_collection(
                name=self.collection
            )
            total = self.chroma_collection.count()
            logger.info(
                f"Connected to ChromaDB | "
                f"collection='{self.collection}' | total={total:,}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            sys.exit(1)

    # ── Schema detection ──────────────────────────────────────────────────────

    def detect_schema(self) -> RowSchema:
        """
        Peeks at the first record to detect embedding dimension.
        Builds RowSchema — source knows nothing about Endee.
        Subclasses override _build_sparse_field() to add a sparse slot.
        """
        logger.info(f"\n{'='*80}\nDetecting collection config: {self.collection}\n{'='*80}")

        # Peek at first record to get dimension
        sample = self.chroma_collection.get(limit=1, include=["embeddings"])
        if not sample["ids"]:
            raise ValueError(
                f"ChromaDB collection '{self.collection}' is empty — nothing to migrate."
            )
        embedding = sample["embeddings"][0]
        if embedding is None:
            raise ValueError(
                "First record has no stored embedding. "
                "Ensure embeddings are pre-computed before running migration."
            )
        dimension = len(embedding)
        logger.info(f"  [DENSE]  dim={dimension} (auto-detected)")

        # Build field list
        schema_fields: List[FieldSchema] = []

        # SLOT 0: ID
        schema_fields.append(FieldSchema(
            name       = "id",
            field_type = FieldType.STRING,
            role       = FieldRole.ID,
        ))

        # SLOT 1: DENSE VECTOR
        schema_fields.append(FieldSchema(
            name       = "embedding",
            field_type = FieldType.DENSE_VECTOR,
            role       = FieldRole.DENSE_VECTOR,
            dimension  = dimension,
        ))
        self._dense_slot = 1

        # SLOT 2 (hybrid only): SPARSE VECTOR — subclass decides
        sparse_field = self._build_sparse_field()
        if sparse_field:
            schema_fields.append(sparse_field)
            self._sparse_slot = len(schema_fields) - 1
            logger.info(f"  [SPARSE] generated from document text (slot={self._sparse_slot})")

        # LAST SLOT: JSON payload (metadata + optionally document text)
        schema_fields.append(FieldSchema(
            name       = "payload",
            field_type = FieldType.JSON,
            role       = FieldRole.METADATA,
        ))
        self._payload_slot = len(schema_fields) - 1

        logger.info(
            f"  Slot map: id=0, dense={self._dense_slot}, "
            f"sparse={self._sparse_slot}, payload={self._payload_slot}"
        )

        self._schema = RowSchema(
            fields     = schema_fields,
            dimension  = dimension,
            space_type = "cosine",          # ChromaDB doesn't expose this; user provides via args
            is_hybrid  = sparse_field is not None,
        )

        self._validate_schema()
        return self._schema

    def _build_sparse_field(self) -> Optional[FieldSchema]:
        """Override in ChromaHybridSource to add a SPARSE_VECTOR slot."""
        return None

    def _validate_schema(self):
        """Override in subclass for extra checks."""

    # ── Record conversion ─────────────────────────────────────────────────────

    def _convert_records(
        self, result: Dict[str, Any]
    ) -> Tuple[List[MigrationRow], float]:
        """
        Converts one ChromaDB .get() result (columnar dict) to MigrationRow list.
        Fills slots POSITIONALLY — matches RowSchema.fields order.
        Returns (rows, transform_time_seconds).
        """
        t0         = time.time()
        ids        = result.get("ids",        [])
        embeddings = result.get("embeddings", None)
        documents  = result.get("documents",  None)
        metadatas  = result.get("metadatas",  None)

        # Sparse embeddings generated per-batch by subclass (None for dense)
        sparse_embs = self._encode_sparse(documents, len(ids)) if self._sparse_slot >= 0 else None

        rows = []
        for i, doc_id in enumerate(ids):
            try:
                row = MigrationRow(self._schema.total_fields)

                # SLOT 0: ID
                row.set_field(0, str(doc_id))

                # SLOT 1: DENSE VECTOR
                # Use `is not None` — bare `if array` raises ValueError on numpy arrays
                embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else None
                if embedding is None:
                    logger.warning(f"  Record '{doc_id}' has no embedding — skipping.")
                    continue
                row.set_field(self._dense_slot, list(embedding))

                # SLOT 2 (hybrid): SPARSE VECTOR
                if self._sparse_slot >= 0 and sparse_embs is not None:
                    sp = sparse_embs[i]
                    if sp is not None and len(sp.indices) > 0:
                        row.set_field(self._sparse_slot, {
                            "indices": sp.indices.tolist(),
                            "values":  sp.values.tolist(),
                        })

                # LAST SLOT: JSON payload
                metadata: dict = (metadatas[i] if metadatas is not None and i < len(metadatas) else None) or {}
                document:  str = (documents[i]  if documents  is not None and i < len(documents)  else None) or ""

                payload = dict(metadata)
                if self.store_document_in_meta and document:
                    payload["document"] = document

                row.set_field(self._payload_slot, payload)
                rows.append(row)

            except Exception as e:
                import traceback
                logger.error(f"Error converting record '{doc_id}': {e}")
                logger.error(traceback.format_exc())
                continue

        return rows, time.time() - t0

    def _encode_sparse(self, documents: Optional[list], count: int) -> Optional[list]:
        """
        Override in ChromaHybridSource to produce sparse embeddings per record.
        Returns a list of sparse embedding objects aligned with the batch, or None.
        """
        return None

    # ── Fetch with retry ──────────────────────────────────────────────────────

    async def _fetch(self, offset: int, batch_size: int, loop) -> Dict[str, Any]:
        max_retries = 5
        include = ["embeddings", "metadatas"]
        if self._sparse_slot >= 0:
            include.append("documents")   # needed for sparse generation

        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda o=offset: self.chroma_collection.get(
                            limit   = batch_size,
                            offset  = o,
                            include = include,
                        ),
                    ),
                    timeout=60,
                )
            except Exception as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Fetch failed (attempt {attempt+1}/{max_retries}): "
                        f"{e}. Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Fetch failed after {max_retries} attempts: {e}")
                    raise

    # ── iterate_batches ───────────────────────────────────────────────────────

    async def iterate_batches(
        self, batch_size: int, initial_cursor: Any, schema: RowSchema
    ):
        """
        Async generator using ChromaDB integer offset pagination.

        Cursor: int — total records processed before this run.
        initial_cursor=0 means start from the beginning.
        Yields: (List[MigrationRow], next_cursor, {"fetch": float, "src_transform": float})
        """
        loop   = asyncio.get_running_loop()
        offset = int(initial_cursor or 0)

        logger.info(f"PRODUCER: starting ChromaDB fetch from offset={offset}")

        while True:
            t_fetch = time.time()
            result  = await self._fetch(offset, batch_size, loop)
            fetch_time = time.time() - t_fetch

            ids = result.get("ids", [])
            if not ids:
                logger.info("PRODUCER: no more data from ChromaDB")
                return

            rows, transform_time = self._convert_records(result)
            next_offset          = offset + len(ids)

            yield rows, next_offset, {"fetch": fetch_time, "src_transform": transform_time}

            if len(ids) < batch_size:
                # Partial page — we've reached the end
                logger.info("PRODUCER: reached end of ChromaDB collection (partial page)")
                return

            offset = next_offset

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Public connectors
# ══════════════════════════════════════════════════════════════════════════════

class ChromaDenseSource(ChromaBaseSource):
    """
    ChromaDB source for DENSE-ONLY migration.
    No sparse generation — just embeddings + metadata.
    """

    def _validate_schema(self):
        logger.info("Schema validated: dense-only (ChromaDB)")

    @classmethod
    def from_args(cls, args):
        return cls(
            url                    = args.source_url,
            collection             = args.source_collection,
            api_key                = args.source_api_key,
            source_path            = getattr(args, "source_path", None),
            store_document_in_meta = getattr(args, "store_document_in_meta", True),
        )


class ChromaHybridSource(ChromaBaseSource):
    """
    ChromaDB source for HYBRID migration.

    Generates sparse vectors ON THE SOURCE SIDE from stored document text
    using endee-model SparseModel (default: "endee/bm25").

    The sink creates the Endee index with sparse_model=args.sparse_model
    (e.g. "endee_bm25") — server applies IDF at search time.

    Requires: document text stored in the ChromaDB collection.
    """

    def __init__(
        self,
        url:                    str,
        collection:             str,
        api_key:                str  = "",
        source_path:            Optional[str] = None,
        store_document_in_meta: bool = True,
        sparse_model_name:      str  = "endee/bm25",
    ):
        super().__init__(
            url                    = url,
            collection             = collection,
            api_key                = api_key,
            source_path            = source_path,
            store_document_in_meta = store_document_in_meta,
        )
        self.sparse_model_name = sparse_model_name
        self._sparse_model     = None

    def connect(self):
        super().connect()
        self._load_sparse_model()

    def _load_sparse_model(self):
        try:
            from endee_model import SparseModel
        except ImportError:
            sys.exit(
                "ERROR: endee-model is not installed.  Run: pip install endee-model"
            )
        logger.info(f"Loading sparse model  '{self.sparse_model_name}'...")
        self._sparse_model = SparseModel(model_name=self.sparse_model_name)
        logger.info(f"  Sparse model ready: {self.sparse_model_name}")

    def _build_sparse_field(self) -> FieldSchema:
        return FieldSchema(
            name       = "sparse_vector",
            field_type = FieldType.SPARSE_VECTOR,
            role       = FieldRole.SPARSE_VECTOR,
        )

    def _encode_sparse(self, documents: Optional[list], count: int) -> Optional[list]:
        """
        Encode document texts into BM25 sparse embeddings.
        Uses SparseModel.embed() — the document-side BM25 function
        (TF x IDF + length normalisation).

        Returns a list of sparse embedding objects, one per record.
        Records with no document text get a zero-length embedding.
        """
        if not documents:
            logger.error(
                "No document text in this batch — cannot generate sparse vectors. "
                "ChromaDB collection must have 'documents' stored for hybrid migration."
            )
            raise RuntimeError(
                "ChromaHybridSource requires document text in the collection. "
                "Use ChromaDenseSource for collections without documents."
            )

        doc_texts = [
            (documents[i] or "") if i < len(documents) else ""
            for i in range(count)
        ]

        if all(t == "" for t in doc_texts):
            raise RuntimeError(
                "All documents in this batch are empty — cannot generate BM25 sparse vectors. "
                "Ensure the ChromaDB collection was created with 'documents' populated."
            )

        return list(self._sparse_model.embed(doc_texts))

    def _validate_schema(self):
        logger.info(
            f"Schema validated: hybrid (ChromaDB → Endee) | "
            f"sparse_model='{self.sparse_model_name}'"
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            url                    = args.source_url,
            collection             = args.source_collection,
            api_key                = args.source_api_key,
            source_path            = getattr(args, "source_path", None),
            store_document_in_meta = getattr(args, "store_document_in_meta", True),
            sparse_model_name      = getattr(args, "sparse_model", "endee/bm25"),
        )