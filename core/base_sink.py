"""
core/base_sink.py
──────────────────────────────────────────────────────────────────────────────
Abstract contract every sink connector must satisfy.

To add a NEW target database (e.g. Pinecone, Weaviate, another Endee index):
  1. Subclass BaseSink.
  2. Implement the three abstract methods.
  3. Register the class in migrate.py's SINK_REGISTRY.

Nothing else in the codebase needs to change.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .record import MigrationRecord, IndexConfig


class BaseSink(ABC):
    """
    Abstract sink connector.

    Lifecycle (called by MigrationPipeline in this order):
        connect()      → establish DB connection
        setup_index()  → get or create the target index using IndexConfig
        upsert_batch() → called once per batch during the migration loop

    upsert_batch() contract
    ───────────────────────
    • Receives a List[MigrationRecord] (the pipeline never passes an empty list).
    • Must convert each MigrationRecord to the target DB's native record format.
    • Must handle internal chunking if the target has per-request size limits.
    • Must retry transient failures with backoff before returning False.
    • Returns True  → all records upserted successfully.
    • Returns False → unrecoverable failure; the pipeline will stop.
    • Must NOT raise exceptions — catch internally and return False.
    """

    # ── Abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the target database."""

    @abstractmethod
    def setup_index(self, config: "IndexConfig") -> None:
        """
        Retrieve or create the target index.

        Called once before the migration loop starts.
        `config` comes from the source's get_index_config().
        """

    @abstractmethod
    async def upsert_batch(self, records: List["MigrationRecord"]) -> bool:
        """
        Write a batch of canonical records to the target.

        Returns True on success, False on unrecoverable failure.
        The pipeline stops immediately on False.
        """

    # ── Optional hook ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Optional explicit teardown.
        Override if you hold resources that need releasing after migration.
        """