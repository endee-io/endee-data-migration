"""
core/type_registry.py
============================================
Canonical intermediate type system for the migration pipeline.

Design rules
============================================

  - This file defines canonical string constants ONLY.
  - Source connectors map their native types → canonical (defined here).
  - Target connectors map canonical → their own native types (in their own files).
  - This file NEVER imports from any source or target connector.

Canonical types
============================================

  Space types  : "cosine" | "l2" | "ip"
  Precisions   : "float32" | "float16" | "int16" | "int8" | "binary"

Adding a new source
============================================

  Add a  <SOURCE>_TO_SPACE  and/or  <SOURCE>_TO_PRECISION  dict below.
  Import it in your source connector — no other file needs to change.

Adding a new target
============================================

  Create a  CANONICAL_TO_<TARGET>_SPACE  and  CANONICAL_TO_<TARGET>_PRECISION
  dict inside your target connector file — no changes needed here.
"""

from __future__ import annotations

# CANONICAL SPACE TYPE CONSTANTS
SPACE_COSINE = "cosine"
SPACE_L2     = "l2"
SPACE_IP     = "ip"


# CANONICAL PRECISION CONSTANTS
PRECISION_FLOAT32 = "float32"
PRECISION_FLOAT16 = "float16"
PRECISION_INT16   = "int16"
PRECISION_INT8    = "int8"
PRECISION_BINARY  = "binary"

PRECISION_RANK: dict[str, int] = {
    PRECISION_BINARY:  0,
    PRECISION_INT8:    1,
    PRECISION_INT16:   2,
    PRECISION_FLOAT16: 3,
    PRECISION_FLOAT32: 4,
}


# ========== SPACE TYPE MAPPING WITH TARGET =============
# ENDEE SPACE MAPPING
# KEY -> USER ENTRY / CANONICAL SPACE TYPE 
# VALUE -> ENDEE SUPPORTED SPACE TYPE
ENDEE_SPACE_MAPPING = {
    SPACE_COSINE: "cosine",
    SPACE_L2: "l2",
    SPACE_IP: "ip",
}
# =======================================================





# ============= PRECISION MAPPING =====================

# QDRANT -> CANONICAL 
QDRANT_PRECISION_MAPPING: dict[str, str] = {

}

# MILVUS -> CANONICAL
MILVUS_PRECISION_MAPPING: dict[str, str] = {
    "float_vector":    PRECISION_FLOAT32,
    "float16_vector":  PRECISION_FLOAT16,
    "binary_vector":   PRECISION_BINARY,
}


# CHROMADB -> CANONICAL
CHROMADB_PRECISION_MAPPING: dict[str, str] = {
    "float32": PRECISION_FLOAT32,   # [EXACT] only storage format ChromaDB uses
    "default": PRECISION_FLOAT32,   # [EXACT] alias — no quantization = float32
    "f32":     PRECISION_FLOAT32,   # [EXACT] shorthand alias
}




def resolve_space(mapping: dict[str, str], raw: str) -> str:
    """
    Look up a raw source space/metric string in the given mapping.
    Log Error and exit the script if not found.
    """
    normalised = (raw or "").strip().lower()
    result = mapping.get(normalised, None)
    if result is None:
        import sys
        import logging
        logger = logging.getLogger(__name__)
        logger.error("=" * 70)
        logger.error(
            f"Invalid space_type '{raw}'. "
            f"Valid values: {sorted(mapping.keys())}"
        )
        logger.error("=" * 70)
        sys.exit(1)
    return result

def resolve_precision(mapping: dict[str, str], raw: str, default: str = PRECISION_FLOAT32) -> str:
    """
    Look up a raw source dtype/precision string in the given mapping.
    Falls back to `default` and logs a warning if not found.
    """
    result = mapping.get(str(raw)) or mapping.get(str(raw).upper())
    if result is None:
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown precision/dtype '{raw}' — defaulting to '{default}'. "
            f"Known values: {list(mapping.keys())}"
        )
        return default
    return result