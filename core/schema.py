"""
core/schema.py
──────────────────────────────────────────────────────────────────────────────
Dynamic schema system — replaces the hardcoded MigrationRecord dataclass.

RowSchema     =carries field names, types, and ROLES
               built once by source connector from source DB metadata
               shared by pipeline and target

MigrationRow  =just a positional list of values
               no field names inside — schema provides the meaning

FieldRole     = how target identifies what to do with each field
               WITHOUT hardcoding field names
               PRIMARY_KEY - id, DENSE_VECTOR - vector, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from constants import DEFAULT_SPACE_TYPE, DEFAULT_M, DEFAULT_EF_CONSTRUCT, DEFAULT_SPARSE_MODEL


# ── Field types ────────────────────────────────────────────────────────────────

class FieldType(Enum):
    STRING        = auto()   # str
    INT           = auto()   # int
    FLOAT         = auto()   # float
    BOOL          = auto()   # bool
    DENSE_VECTOR  = auto()   # List[float] — fixed dimension dense vector
    SPARSE_VECTOR = auto()   # Dict[int, float] — variable sparse vector
    JSON          = auto()   # dict — arbitrary nested payload


# ── Field roles ────────────────────────────────────────────────────────────────

class FieldRole(Enum):
    # PRIMARY_KEY   = auto()   # the unique identifier — target uses as "id"
    ID            = auto()   # the unique identifier — target uses as "id"
    DENSE_VECTOR  = auto()   # the main float[] vector — target indexes for ANN
    SPARSE_VECTOR = auto()   # sparse token weights — target uses for hybrid search
    METADATA      = auto()   # everything else — target splits into filter/meta


# ── Individual field descriptor ────────────────────────────────────────────────

@dataclass
class FieldSchema:
    """Describes one field in the row.
    example:
        name: str
            'id'/'embedding'/'title'

        field_type: FieldType
            STRING/INT/FLOAT/BOOL/FLOAT_VECTOR/SPARSE_VECTOR
        
        role: FieldRole
            ID/DENSE_VECTOR/SPARSE_VECTOR/METADATA
    """
    name:       str
    field_type: FieldType
    role:       FieldRole
    dimension:  Optional[int]  = None   # only for FLOAT_VECTOR
    # is_filterable: bool        = False  # hint: should this metadata go to filter? IT CAN BE REMOVED



# --- Full row schema ---------------------------------------------

@dataclass
class RowSchema:
    """
    The complete schema for a row.

    Built ONCE by the SOURCE CONNECTOR from the source DB's collection metadata.
    Passed to the TARGET CONNECTOR for role detection and index creation.
    Neither source NOR target hardcodes field names — they use roles and index_of().

    Also carries index creation params (replaces IndexConfig) so the sink
    can create the target index purely from this object.
    """
    # STORE SCHEMA OF EACH COLUMN
    fields: List[FieldSchema]

    # # --- INDEX CREATION PARAMS ( FROM SOURCE, REPACES INDEXCONFIG ) ---
    # space_type:   str
    # M:            int
    # ef_construct: int
    # precision:    Optional[Any]
    # sparse_model: Optional[str] = None  # e.g. "bm25" — for hybrid index creation

    #--- ROLE LOOKUPS ---------------------------------------------

    def get_primary_key(self) -> Optional[FieldSchema]:
        '''
            GET THE PRIMARY/ID COLUMN
        '''
        for f in self.fields:
            if f.role == FieldRole.ID:
                return f
        return None

    def get_dense_vector(self) -> Optional[FieldSchema]:
        '''
            GET THE PRIMARY/ID COLUMN
        '''
        for f in self.fields:
            if f.role == FieldRole.DENSE_VECTOR:
                return f
        return None

    def get_sparse_vector(self) -> Optional[FieldSchema]:
        '''
            GET THE PRIMARY/ID COLUMN
        '''
        for f in self.fields:
            if f.role == FieldRole.SPARSE_VECTOR:
                return f
        return None

    def get_metadata_fields(self) -> List[FieldSchema]:
        '''
            GET THE PRIMARY/ID COLUMN
        '''
        return [f for f in self.fields if f.role == FieldRole.METADATA]

    # ---- index access ---------------------------------------------

    def index_of(self, name: str) -> int:
        """Returns slot position for a field name, -1 if not found."""
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        return -1

    def require_index_of(self, name: str) -> int:
        """Like index_of but raises if missing — use for required fields."""
        idx = self.index_of(name)
        if idx < 0:
            available = [f.name for f in self.fields]
            raise ValueError(
                f"Required field '{name}' not found in schema. "
                f"Available: {available}"
            )
        return idx

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def total_fields(self) -> int:
        return len(self.fields)

    @property
    def is_hybrid(self) -> bool:
        return self.get_sparse_vector() is not None

    @property
    def dimension(self) -> Optional[int]:
        dense = self.get_dense_vector()
        return dense.dimension if dense else None


# ── The actual row data ─────────────────────────────────────────────────────────

class MigrationRow:
    """
    A single row of data. Stores values of one row.
    Just a positional list of values. No field names inside.
    RowSchema provides the meaning of each slot.

    Source writes:   row.set_field(0, value)   — by slot number
    Target reads:    row.get_field(schema.index_of("id"))  — by name via schema
    """
    __slots__ = ('fields',)

    def __init__(self, arity: int):
        self.fields: List[Any] = [None] * arity

    def set_field(self, pos: int, value: Any) -> None:
        self.fields[pos] = value

    def get_field(self, pos: int) -> Any:
        return self.fields[pos]

    @property
    def arity(self) -> int:
        return len(self.fields)

    def __repr__(self):
        return f"MigrationRow({self.fields!r})"