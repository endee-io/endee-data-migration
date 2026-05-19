"""
migrate.py Unified CLI entrypoint for all migration types.
──────────────────────────────────────────────────────────────────────────────

Usage
─────
    python migrate.py <MIGRATION_TYPE> [OPTIONS]

Migration types
───────────────
    milvus-to-endee-dense    MilvusDenseSource  → EndeeSink
    milvus-to-endee-hybrid   MilvusHybridSource → EndeeSink
    qdrant-to-endee-dense    QdrantDenseSource  → EndeeSink
    qdrant-to-endee-hybrid   QdrantHybridSource → EndeeSink

Adding a new migration type
───────────────────────────
    1. Write a BaseSource subclass (e.g. sources/pinecone_source.py).
    2. Write a BaseSink subclass if needed (e.g. sinks/weaviate_sink.py).
    3. Add an entry to SOURCE_REGISTRY and/or SINK_REGISTRY below.
    4. Add argument parsing in _build_source() / _build_sink() as needed.
    That's it. The pipeline, checkpoint, and retry logic are reused automatically.
"""

from __future__ import annotations
from sources.milvus_source import MilvusDenseSource, MilvusHybridSource
from sources.qdrant_source import QdrantDenseSource, QdrantHybridSource
from sinks.endee_sink      import EndeeSink
from core.checkpoint       import MigrationCheckpoint
from core.pipeline         import MigrationPipeline

import argparse
import logging
import os
import sys

import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Registries ────────────────────────────────────────────────────────────────
# Map migration-type string → (SourceClass, SinkClass)
# Adding a new type = one line here + (optionally) arg parsing below.

def _get_registries():
    """Deferred import so missing optional deps don't break unrelated types."""

    SOURCE_REGISTRY = {
        "milvus-to-endee-dense":   MilvusDenseSource,
        "milvus-to-endee-hybrid":  MilvusHybridSource,
        "qdrant-to-endee-dense":   QdrantDenseSource,
        "qdrant-to-endee-hybrid":  QdrantHybridSource,
    }
    SINK_REGISTRY = {
        "milvus-to-endee-dense":   EndeeSink,
        "milvus-to-endee-hybrid":  EndeeSink,
        "qdrant-to-endee-dense":   EndeeSink,
        "qdrant-to-endee-hybrid":  QdrantDenseSource,
    }
    # The sink is always EndeeSink for current types.
    # Override here once a new sink class is added.
    SINK_REGISTRY = {k: EndeeSink for k in SOURCE_REGISTRY}
    return SOURCE_REGISTRY, SINK_REGISTRY


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified vector DB migration tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "migration_type",
        nargs="?",
        default=os.getenv("MIGRATION_TYPE"),
        help=(
            "Migration type. One of:\n"
            "  milvus-to-endee-dense\n"
            "  milvus-to-endee-hybrid\n"
            "  qdrant-to-endee-dense\n"
            "  qdrant-to-endee-hybrid\n"
        ),
    )

    # ── Source ────────────────────────────────────────────────────────────────
    src = p.add_argument_group("Source")
    src.add_argument("--source_url",        default=os.getenv("SOURCE_URL"))
    src.add_argument("--source_api_key",    default=os.getenv("SOURCE_API_KEY", ""))
    src.add_argument("--source_collection", default=os.getenv("SOURCE_COLLECTION"))
    src.add_argument("--source_port",       default=os.getenv("SOURCE_PORT"), type=int)
    src.add_argument("--source_db",         default=os.getenv("SOURCE_DB", "default"),
                     help="Milvus database name (default: 'default')")
    src.add_argument("--filter_fields",     default=os.getenv("FILTER_FIELDS", ""),
                     help="Comma-separated payload fields to expose as Endee filter attributes")
    src.add_argument("--use_https",         action="store_true",
                     default=os.getenv("USE_HTTPS", "false").lower() == "true",
                     help="Use HTTPS for Qdrant connection")

    # ── Target ────────────────────────────────────────────────────────────────
    tgt = p.add_argument_group("Target")
    tgt.add_argument("--target_url",        default=os.getenv("TARGET_URL"))
    tgt.add_argument("--target_api_key",    default=os.getenv("TARGET_API_KEY", ""))
    tgt.add_argument("--target_collection", default=os.getenv("TARGET_COLLECTION"))

    # ── Index config ──────────────────────────────────────────────────────────
    idx = p.add_argument_group("Index configuration")
    idx.add_argument("--space_type",  default=os.getenv("SPACE_TYPE", "cosine"))
    idx.add_argument("--M",           type=int,
                     default=int(os.getenv("M")) if os.getenv("M") else None,
                     help="HNSW M parameter (auto-detected from source if not set)")
    idx.add_argument("--ef_construct", type=int,
                     default=int(os.getenv("EF_CONSTRUCT")) if os.getenv("EF_CONSTRUCT") else None,
                     help="HNSW ef_construct (auto-detected from source if not set)")
    idx.add_argument("--precision",
                     default=os.getenv("PRECISION", None),
                     help="Precision override: float32 / float16 / int8 / int16 / binary")

    # ── Performance ───────────────────────────────────────────────────────────
    perf = p.add_argument_group("Performance")
    perf.add_argument("--batch_size",      type=int,
                      default=int(os.getenv("BATCH_SIZE", 1000)))
    perf.add_argument("--upsert_size",     type=int,
                      default=int(os.getenv("UPSERT_SIZE", 100)),
                      help="Endee upsert chunk size (default: 100)")
    perf.add_argument("--max_queue_size",  type=int,
                      default=int(os.getenv("MAX_QUEUE_SIZE", 5)))

    # ── Resume / checkpoint ───────────────────────────────────────────────────
    ckpt = p.add_argument_group("Checkpoint / resume")
    ckpt.add_argument("--checkpoint_file",
                      default=os.getenv("CHECKPOINT_FILE", "./migration_checkpoint.json"))
    ckpt.add_argument("--resume",
                      action="store_true",
                      default=os.getenv("RESUME", "true").lower() == "false",
                      help="Set RESUME=false (or --resume) to clear checkpoint and start fresh")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--debug", action="store_true",
                   default=os.getenv("DEBUG", "false").lower() == "true")

    return p


# ── Source factory ────────────────────────────────────────────────────────────

def _build_source(migration_type: str, args, precision):
    """Instantiate the correct source connector from CLI args."""

    if migration_type in ("milvus-to-endee-dense", "milvus-to-endee-hybrid"):
        common = dict(
            url          = args.source_url,
            token        = args.source_api_key,
            collection   = args.source_collection,
            db           = args.source_db,
            port         = args.source_port or 19530,
            space_type   = args.space_type,
            M            = args.M,
            ef_construct = args.ef_construct,
            precision    = args.precision,   # str or None
            filter_fields= args.filter_fields,
        )
        if migration_type == "milvus-to-endee-dense":
            return MilvusDenseSource(**common)
        return MilvusHybridSource(**common)

    if migration_type in ("qdrant-to-endee-dense", "qdrant-to-endee-hybrid"):
        common = dict(
            url          = args.source_url,
            collection   = args.source_collection,
            api_key      = args.source_api_key,
            port         = args.source_port,
            use_https    = args.use_https,
            space_type   = args.space_type if args.space_type != "cosine" else None,
            M            = args.M,
            ef_construct = args.ef_construct,
            precision    = args.precision,   # str or None
            filter_fields= args.filter_fields,
        )
        if migration_type == "qdrant-to-endee-dense":
            return QdrantDenseSource(**common)
        return QdrantHybridSource(**common)

    raise ValueError(f"Unknown migration type: {migration_type}")


# ── Sink factory ──────────────────────────────────────────────────────────────

def _build_sink(migration_type: str, args):
    from sinks.endee_sink import EndeeSink
    return EndeeSink(
        endee_url         = args.target_url,
        endee_api_key     = args.target_api_key,
        index_name        = args.target_collection,
        upsert_chunk_size = args.upsert_size,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    migration_type = args.migration_type
    if not migration_type:
        parser.print_help()
        print("\nError: migration_type is required (CLI arg or MIGRATION_TYPE env var)")
        sys.exit(1)

    SOURCE_REGISTRY, _ = _get_registries()
    if migration_type not in SOURCE_REGISTRY:
        print(f"\nUnknown migration type: '{migration_type}'")
        print(f"Valid types: {list(SOURCE_REGISTRY.keys())}")
        sys.exit(1)

    # ── Precision validation ──────────────────────────────────────────────────
    # Keep as a string at this layer — source connectors resolve it to
    # the endee.Precision enum themselves (keeps this file free of endee imports).
    VALID_PRECISIONS = {"float32", "float16", "int8", "int16", "binary"}
    if args.precision is not None:
        if args.precision not in VALID_PRECISIONS:
            logger.error(f"Invalid precision '{args.precision}'. Valid: {VALID_PRECISIONS}")
            sys.exit(1)
    else:
        # Qdrant sources require an explicit precision because they can't
        # safely auto-select one without risking silent data loss.
        if migration_type.startswith("qdrant-"):
            logger.warning(
                "No --precision / PRECISION set for a Qdrant source. "
                "The source will auto-detect from quantization config. "
                "Set PRECISION explicitly to suppress this warning."
            )

    # ── Build components ──────────────────────────────────────────────────────


    checkpoint = MigrationCheckpoint(args.checkpoint_file)

    if args.resume:
        logger.info("--resume / RESUME=false: clearing checkpoint for fresh start")
        checkpoint.clear()

    source = _build_source(migration_type, args, args.precision)
    sink   = _build_sink(migration_type, args)

    pipeline = MigrationPipeline(
        source           = source,
        sink             = sink,
        checkpoint       = checkpoint,
        fetch_batch_size = args.batch_size,
        max_queue_size   = args.max_queue_size,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info(f"Migration type : {migration_type}")
    logger.info(f"Source         : {args.source_collection} @ {args.source_url}")
    logger.info(f"Target         : {args.target_collection}")
    logger.info("=" * 80)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()