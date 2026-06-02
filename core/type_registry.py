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
from endee import Precision
from typing import Any
from pymilvus import DataType


# CANONICAL SPACE TYPE CONSTANTS
SPACE_COSINE = "cosine"
SPACE_L2     = "l2"
SPACE_IP     = "ip"
SPACE_MANHATTAN = "manhattan" # L1 DISTANCE




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

# ==============================================================
# CANONICAL -> ENDEE
CANONICAL_TO_ENDEE_PRECISION_MAPPING: dict[str, Precision] = {
    PRECISION_FLOAT32: Precision.FLOAT32,
    PRECISION_FLOAT16: Precision.FLOAT16,
    PRECISION_INT16: Precision.INT16,
    PRECISION_INT8: Precision.INT8,
    PRECISION_BINARY: Precision.BINARY2
} 

# ENDEE SPACE MAPPING
# KEY -> USER ENTRY / CANONICAL SPACE TYPE 
# VALUE -> ENDEE SUPPORTED SPACE TYPE
CANONICAL_TO_ENDEE_SPACE_MAPPING = {
    SPACE_COSINE: "cosine",
    SPACE_L2: "l2",
    SPACE_IP: "ip",
}

# ===============================================================

# MILVUS -> CANONICAL
MILVUS_PRECISION_MAPPING: dict[str, str] = {
    "float_vector":    PRECISION_FLOAT32,
    "float16_vector":  PRECISION_FLOAT16,
    "binary_vector":   PRECISION_BINARY,
    DataType.FLOAT_VECTOR: PRECISION_FLOAT32,
    DataType.FLOAT16_VECTOR: PRECISION_FLOAT16,
    DataType.BINARY_VECTOR: PRECISION_BINARY,
}
MILVUS_TO_CANONICAL_SPACE_MAPPING = {
    "cosine": SPACE_COSINE,
    "l2":     SPACE_L2,
    "ip":     SPACE_IP,
}

# ===============================================================

# CHROMADB -> CANONICAL
CHROMADB_PRECISION_MAPPING: dict[str, str] = {
    "float32": PRECISION_FLOAT32,   # [EXACT] only storage format ChromaDB uses
    "default": PRECISION_FLOAT32,   # [EXACT] alias — no quantization = float32
    "f32":     PRECISION_FLOAT32,   # [EXACT] shorthand alias
}

# =======================================================

QDRANT_TO_CANONICAL_SPACE_TYPE: dict[str, str] = {
    # Distance enum string forms (from qdrant_client.http.models.Distance)
    "cosine": SPACE_COSINE,
    "euclid": SPACE_L2,
    "dot":    SPACE_IP,
    "manhattan": SPACE_L2,   # [CLOSE] L1 norm, no canonical; L2 is nearest
}

# QDRANT -> CANONICAL 
QDRANT_TO_CANONICAL_PRECISION_MAPPING: dict[str, str] = {
    # ── Vector field types (no quantization) ─────────────────────────────────
    "float_vector":    PRECISION_FLOAT32,   # [EXACT] default Qdrant float32 storage
    "float16_vector":  PRECISION_FLOAT16,   # [EXACT] half-precision float16 storage
    "bfloat16_vector": PRECISION_FLOAT16,   # [CLOSE] bfloat16 is 16-bit but different exponent/mantissa split vs float16; float16 is the nearest canonical tier
    "binary_vector":   PRECISION_BINARY,    # [EXACT] 1-bit packed binary vectors
 
    # ── No quantization config present ───────────────────────────────────────
    "none":            PRECISION_FLOAT32,   # [EXACT] unquantised = float32
 
    # ── Scalar quantization ───────────────────────────────────────────────────
    # Qdrant scalar only supports int8 (float32 → uint8 internally).
    # No int16 scalar exists in Qdrant.
    "scalar":          PRECISION_INT8,      # [EXACT] float32 → uint8 = 8-bit integer
    "int8":            PRECISION_INT8,      # [EXACT] explicit sub-type alias
 
    # ── TurboQuant (available from Qdrant v1.18.0) ───────────────────────────
    # TurboQuant uses sub-byte precision levels that have no exact canonical
    # counterpart. Mapped to the nearest tier above the actual bit depth.
    #
    # Key format: "turbo:<bits_value>"
    # Top-level "turbo" key (no bits sub-field) defaults to bits4.
    "turbo":           PRECISION_INT8,      # [CLOSE] default bits4 (4-bit, 8× compression) no INT4 canonical; INT8 is nearest
    "turbo:bits4":     PRECISION_INT8,      # [CLOSE] 4-bit; INT8 (8-bit) is nearest tier above
    "turbo:bits2":     PRECISION_BINARY,    # [CLOSE] 2-bit; BINARY is nearest tier below
    "turbo:bits1_5":   PRECISION_BINARY,    # [CLOSE] 1.5-bit; BINARY is nearest
    "turbo:bits1":     PRECISION_BINARY,    # [EXACT] 1-bit; maps exactly to BINARY
 
    # ── Binary quantization ───────────────────────────────────────────────────
    # Default binary = 1-bit (exact BINARY match).
    # 2-bit and 1.5-bit encodings (available from v1.15.0) are sub-byte but
    # above 1-bit; no canonical tier exists between INT8 and BINARY, so these
    # map to BINARY (nearest lower tier).
    #
    # Key format: "binary[:<encoding_value>]"
    "binary":                    PRECISION_BINARY,  # [EXACT] 1-bit default binary
    "binary:one_bit":            PRECISION_BINARY,  # [EXACT] explicit 1-bit encoding
    "binary:two_bits":           PRECISION_BINARY,  # [CLOSE] 2-bit encoding; no 2-bit canonical; BINARY is nearest
    "binary:one_and_half_bits":  PRECISION_BINARY,  # [CLOSE] 1.5-bit encoding; same as above
 
    # ── Product quantization ──────────────────────────────────────────────────
    # Product quantization has no canonical equivalent — it is a lossy centroid
    # compression scheme with no fixed bit depth per dimension.
    # DO NOT add it here. The migration script must detect "product" in
    # quantization_config and raise a ValueError explicitly.
}

# ===============================================================





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

def resolve_precision(mapping: dict[str, str], raw: Any) -> str:
    """
    Look up a raw source dtype/precision string in the given mapping.
    Falls back to `default` and logs a warning if not found.
    """
    if isinstance(raw, str):
        normalised = ( raw or "").strip().lower()
    else:
        normalised = raw
    result = mapping.get(normalised, None)
    if result is None:
        import sys
        import logging
        logger = logging.getLogger(__name__)
        logger.error("=" * 70)
        logger.error(
            f"Unknown Precision '{raw}'. "
            f"Valid values: {sorted(mapping.keys())}"
        )
        logger.error("=" * 70)
        sys.exit(1)
    return result