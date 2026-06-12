"""
core/type_registry.py
============================================
Canonical type constants shared across all source and target connectors.

Rules
============================================
  - This file defines canonical string constants ONLY.
  - No imports from source or target connectors.
  - No DB-specific mappings — those live in each connector file.
  - No utility functions — each connector defines its own _resolve_* helpers.

Adding a new source or target
============================================
  Define your DB-specific mappings (e.g. YOURDB_TO_WIRE_PRECISION,
  CANONICAL_TO_YOURDB_SPACE_MAPPING) inside your connector file.
  Import only the canonical constants you need from here.
"""

from __future__ import annotations


# ── Canonical space type constants ────────────────────────────────────────────
SPACE_COSINE    = "cosine"
SPACE_L2        = "l2"
SPACE_IP        = "ip"
SPACE_MANHATTAN = "manhattan"   # L1 distance


# ── Canonical precision constants ─────────────────────────────────────────────
PRECISION_FLOAT32    = "float32"
PRECISION_FLOAT16    = "float16"
PRECISION_INT16      = "int16"
PRECISION_INT8       = "int8"
PRECISION_BINARY     = "binary"
PRECISION_RAW_BINARY = "raw_binary"  # wire type for any non-float32 Milvus vector (raw bytes)

# Rank used by target connectors to compare source vs target precision.
# Higher rank = higher fidelity.
PRECISION_RANK: dict[str, int] = {
    PRECISION_BINARY:  0,
    PRECISION_INT8:    1,
    PRECISION_INT16:   2,
    PRECISION_FLOAT16: 3,
    PRECISION_FLOAT32: 4,
}


# ── Shared resolver helpers ───────────────────────────────────────────────────

import logging as _logging
import sys as _sys

_logger = _logging.getLogger(__name__)


def resolve_space(mapping: dict[str, str], raw: str) -> str:
    """Look up a raw space/metric string in the given mapping. Exits on unknown."""
    normalised = (raw or "").strip().lower()
    result = mapping.get(normalised)
    if result is None:
        _logger.error(f"Invalid space_type '{raw}'. Valid: {sorted(mapping.keys())}")
        _sys.exit(1)
    return result


def resolve_precision(mapping: dict, raw) -> any:
    """Look up a raw precision string in the given mapping. Exits on unknown."""
    normalised = (raw or "").strip().lower() if isinstance(raw, str) else raw
    result = mapping.get(normalised)
    if result is None:
        _logger.error(f"Unknown precision '{raw}'. Valid: {sorted(str(k) for k in mapping.keys())}")
        _sys.exit(1)
    return result
