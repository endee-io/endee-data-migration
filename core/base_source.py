"""
core/base_source.py
──────────────────────────────────────────────────────────────────────────────
Abstract contract every source connector must satisfy.

To add a NEW source database (e.g. Pinecone, Weaviate):
  1. Subclass BaseSource.
  2. Implement the four abstract methods.
  3. Register the class in migrate.py's SOURCE_REGISTRY.

Nothing else in the codebase needs to change.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, AsyncGenerator
from .schema import RowSchema, MigrationRow

# if TYPE_CHECKING:
#     from .record import MigrationRecord, IndexConfig
#     from .checkpoint import MigrationCheckpoint


class BaseSource(ABC):
    """
    Abstract source connector.

    Lifecycle (called by MigrationPipeline in this order):
        connect()        - establish DB connection
        detect_schema()  - inspect collection schema, populate internal state
        iterate_batches()  - async-generator that yields migration batches

    iterate_batches() contract

    - Must be an async generator.
    - Yields: (List[MigrationRow], next_cursor)
        - next_cursor is an opaque value stored in the checkpoint; the pipeline
          passes it back as `initial_cursor` on resume.  Use `None` to signal
          "I am done; nothing more to resume from."
        - For count-based sources (Milvus QueryIterator) cursor = total records
          processed so far (int).
        - For cursor-based sources (Qdrant scroll) cursor = scroll offset token.
    - Must honour `initial_cursor`: skip already-processed records on resume.
    - Must clean up (close iterators / connections) in a `finally` block so
      the pipeline's `break` triggers proper teardown via async-generator aclose().
    """


    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the source database."""

    @abstractmethod
    def detect_schema(self) -> None:
        """
        Inspect the collection schema and populate any internal state needed
        by get_index_config() and iterate_batches().

        E.g. for Milvus: load collection, walk fields, detect vector type,
        read HNSW index params.
        """

    # @abstractmethod
    # def get_index_config(self) -> "IndexConfig":
    #     """
    #     Return an IndexConfig describing the source collection.
    #     Called *after* detect_schema(), so all schema info is available.
    #     The pipeline passes this to the sink's setup_index().

    #     get_index_config() - return IndexConfig the sink will use
    #     """

    @abstractmethod
    async def iterate_batches(
        self,
        batch_size: int,
        initial_cursor: Any,
        schema: RowSchema,
    ) -> AsyncGenerator[Tuple[List[MigrationRow], Any, Dict[str, float]], None]:
        """
        Async generator.

        Yields
        ------
        Async generator yielding (batch_of_rows, next_cursor).
        Source fills MigrationRow slots using schema.index_of() or positionally.

        Parameters
        ----------
        batch_size     : how many records to fetch per iteration
        initial_cursor : value from checkpoint (None = fresh start)
        schema         : schema of rows
        """
        # # Subclasses implement this as an async generator.
        # # The `yield` below is intentional — it turns this into an async
        # # generator stub so type checkers don't complain, but subclasses
        # # must override and actually yield data.
        # raise NotImplementedError
        # yield  # pragma: no cover

    #  FACTORY CLASSMETHOD

    @classmethod
    @abstractmethod
    def from_args(cls, args: Any) -> "BaseSource":
        """
        Construct an instance from the parsed CLI args namespace.

        Each subclass implements this to pull only the args it needs,
        keeping migrate.py free of any source-specific argument mapping.

        Example
        -------
        @classmethod
        def from_args(cls, args):
            return cls(
                url          = args.source_url,
                token        = args.source_api_key,
                collection   = args.source_collection,
                ...
            )
        """

    # ── Optional hook ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Optional explicit teardown (called by the pipeline in a finally block).
        Override if you hold resources outside the iterator.
        """