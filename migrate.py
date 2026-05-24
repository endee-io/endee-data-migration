"""
migrate.py  -  Unified CLI entrypoint for all migration types.
──────────────────────────────────────────────────────────────────────────────

Usage
─────
    python migrate.py --from milvus --to endee --type dense [OPTIONS]
    python migrate.py --from qdrant --to endee --type hybrid [OPTIONS]

How the registry works
──────────────────────
    SOURCE_REGISTRY[(from_db, vector_type)]  -  SourceClass
    Target_REGISTRY  [(to_db,   vector_type)]  -  TargetClass

    Both factories are single dict lookups — no if/elif chains.
    Each class owns its own from_args() classmethod, so this file
    never needs to know which args a specific DB requires.

Adding a new migration type
───────────────────────────
    1. Write a BaseSource subclass (e.g. sources/pinecone_source.py).
       Implement from_args(args) to pull whatever args your source needs.
    2. Write a BaseSink subclass if needed (e.g. sinks/weaviate_sink.py).
       Implement from_args(args) to pull whatever args your sink needs.
    3. Add --from / --to argument values to the choices= lists below.
    4. Add one entry to SOURCE_REGISTRY and/or SINK_REGISTRY.
    Nothing else changes.
"""

from __future__ import annotations

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
# Key: (database_name, vector_type)
# Value: connector class (must implement from_args(args))
#
# To add Pinecone as a source:
#   from sources.pinecone_source import PineconeDenseSource
#   SOURCE_REGISTRY[("pinecone", "dense")] = PineconeDenseSource
#
# To add Weaviate as a sink:
#   from sinks.weaviate_sink import WeaviateSink
#   SINK_REGISTRY[("weaviate", "dense")]  = WeaviateSink
#   SINK_REGISTRY[("weaviate", "hybrid")] = WeaviateSink

from sources.milvus_source import MilvusDenseSource, MilvusHybridSource
from sources.qdrant_source import QdrantDenseSource, QdrantHybridSource
from targets.endee_target import EndeeTarget

SOURCE_REGISTRY = {
    ("milvus", "dense"):  MilvusDenseSource,
    ("milvus", "hybrid"): MilvusHybridSource,
    ("qdrant", "dense"):  QdrantDenseSource,
    ("qdrant", "hybrid"): QdrantHybridSource,
}

TARGET_REGISTRY = {
    ("endee", "dense"):  EndeeTarget,
    ("endee", "hybrid"): EndeeTarget,
}


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified vector DB migration tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Migration axes ────────────────────────────────────────────────────────
    p.add_argument(
        "--from", dest="from_db",
        default=os.getenv("FROM_DB"),
        choices=["milvus", "qdrant"],       # extend when adding new sources
        required=not os.getenv("FROM_DB"),
        metavar="DB",
        help="Source database type.  Choices: milvus | qdrant",
    )
    p.add_argument(
        "--to", dest="to_db",
        default=os.getenv("TO_DB"),
        choices=["endee"],                  # extend when adding new Target
        required=not os.getenv("TO_DB"),
        metavar="DB",
        help="Target database type.  Choices: endee",
    )
    p.add_argument(
        "--type",
        default=os.getenv("VECTOR_TYPE", "dense"),
        choices=["dense", "hybrid"],
        help="Vector type of the collection (default: dense)",
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
                     help="Comma-separated payload fields to expose as filter attributes")
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
    idx.add_argument("--space_type",   default=os.getenv("SPACE_TYPE", "cosine"))
    idx.add_argument("--M",            type=int,
                     default=int(os.getenv("M")) if os.getenv("M") else None,
                     help="HNSW M (auto-detected from source if not set)")
    idx.add_argument("--ef_construct", type=int,
                     default=int(os.getenv("EF_CONSTRUCT")) if os.getenv("EF_CONSTRUCT") else None,
                     help="HNSW ef_construct (auto-detected from source if not set)")
    idx.add_argument("--precision",
                     default=os.getenv("PRECISION", None),
                     help="float32 | float16 | int8 | int16 | binary")

    # ── Performance ───────────────────────────────────────────────────────────
    perf = p.add_argument_group("Performance")
    perf.add_argument("--batch_size",     type=int,
                      default=int(os.getenv("BATCH_SIZE", 1000)))
    perf.add_argument("--upsert_size",    type=int,
                      default=int(os.getenv("UPSERT_SIZE", 100)))
    perf.add_argument("--max_queue_size", type=int,
                      default=int(os.getenv("MAX_QUEUE_SIZE", 5)))

    # ── Resume / checkpoint ───────────────────────────────────────────────────
    ckpt = p.add_argument_group("Checkpoint / resume")
    ckpt.add_argument("--checkpoint_file",
                      default=os.getenv("CHECKPOINT_FILE", "./migration_checkpoint.json"))
    ckpt.add_argument("--resume",
                      action="store_true",
                      default=os.getenv("RESUME", "true").lower() == "false",
                      help="Clear checkpoint and start fresh (set RESUME=false in env)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--debug", action="store_true",
                   default=os.getenv("DEBUG", "false").lower() == "true")

    return p


# ── Factories — no if/elif; pure registry lookups ─────────────────────────────

def _build_source(args):
    key = (args.from_db, args.type)
    SourceClass = SOURCE_REGISTRY.get(key)
    if SourceClass is None:
        logger.error(
            f"No source registered for --from={args.from_db} --type={args.type}.\n"
            f"Registered sources: {list(SOURCE_REGISTRY.keys())}"
        )
        sys.exit(1)
    return SourceClass.from_args(args)


def _build_target(args):
    key = (args.to_db, args.type)
    TargetClass = TARGET_REGISTRY.get(key)
    if TargetClass is None:
        logger.error(
            f"No Target registered for --to={args.to_db} --type={args.type}.\n"
            f"Registered Targets: {list(Target_REGISTRY.keys())}"
        )
        sys.exit(1)
    return TargetClass.from_args(args)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Precision validation ──────────────────────────────────────────────────
    VALID_PRECISIONS = {"float32", "float16", "int8", "int16", "binary"}
    if args.precision is not None and args.precision not in VALID_PRECISIONS:
        logger.error(f"Invalid --precision '{args.precision}'. Valid: {VALID_PRECISIONS}")
        sys.exit(1)
    if args.precision is None and args.from_db == "qdrant":
        logger.warning(
            "No --precision / PRECISION set for a Qdrant source. "
            "Auto-detecting from quantization config. "
            "Set PRECISION explicitly to suppress this warning."
        )

    # ── Build pipeline components ─────────────────────────────────────────────
    from core.checkpoint import MigrationCheckpoint
    from core.pipeline   import MigrationPipeline

    checkpoint = MigrationCheckpoint(args.checkpoint_file)
    if args.resume:
        logger.info("--resume / RESUME=false: clearing checkpoint for fresh start")
        checkpoint.clear()

    source   = _build_source(args)
    target     = _build_target(args)
    pipeline = MigrationPipeline(
        source           = source,
        target           = target,
        checkpoint       = checkpoint,
        fetch_batch_size = args.batch_size,
        max_queue_size   = args.max_queue_size,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info(f"From     : {args.from_db}  ({args.source_collection} @ {args.source_url})")
    logger.info(f"To       : {args.to_db}    ({args.target_collection})")
    logger.info(f"Type     : {args.type}")
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