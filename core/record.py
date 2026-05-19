"""
core/record.py
──────────────────────────────────────────────────────────────────────────────
Canonical in-flight data format.

Every source connector converts its native records into MigrationRecord.
Every sink connector consumes MigrationRecord and converts to its native format.

Neither sources nor sinks know about each other — they only speak MigrationRecord.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MigrationRecord:
    """
    Canonical record format that flows through the pipeline.

    Fields
    ------
    id              : string identifier (always stringified)
    dense_vector    : float list for the dense ANN index
    filter_data     : payload fields exposed as filterable attributes
    meta_data       : payload fields stored as opaque metadata
    sparse_indices  : token indices for hybrid search (None = dense-only record)
    sparse_values   : BM25 / TF-IDF weights parallel to sparse_indices
    """
    id: str
    dense_vector: List[float]
    filter_data: Dict[str, Any] = field(default_factory=dict)
    meta_data: Dict[str, Any] = field(default_factory=dict)
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None

    @property
    def is_hybrid(self) -> bool:
        """True if this record carries a sparse vector in addition to dense."""
        return (
            self.sparse_indices is not None
            and len(self.sparse_indices) > 0
        )


@dataclass
class IndexConfig:
    """
    Everything a sink needs to create or verify the target index.

    Produced by BaseSource.get_index_config() after schema detection.
    The sink uses this to call the target DB's index-creation API once,
    before the migration loop starts.

    Fields
    ------
    dimension       : dense vector dimensionality
    space_type      : similarity metric ('cosine', 'l2', 'ip')
    M               : HNSW M parameter
    ef_construct    : HNSW ef_construct parameter
    precision       : quantization precision (e.g. endee.Precision.INT16)
    is_hybrid       : True → index must also accept sparse vectors
    sparse_dimension: vocabulary size for the sparse index (hybrid only)
    sparse_model    : sparse model identifier passed to the target (hybrid only)
    """
    dimension: int
    space_type: str
    M: int
    ef_construct: int
    precision: Any                          # sink-specific precision enum/value
    is_hybrid: bool = False
    sparse_dimension: Optional[int] = None
    sparse_model: Optional[str] = None