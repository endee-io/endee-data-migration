#!/bin/bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

show_help() {
cat << EOF
${CYAN}╔══════════════════════════════════════════════════════╗
║         Endee Migration Tool v2.0                    ║
╚══════════════════════════════════════════════════════╝${NC}

${GREEN}Usage:${NC}
  docker run [docker-options] vector-migration ${YELLOW}<MIGRATION_TYPE>${NC} [OPTIONS]

${GREEN}Migration types:${NC}
  ${YELLOW}milvus-to-endee-dense${NC}    Dense Milvus collection → Endee
  ${YELLOW}milvus-to-endee-hybrid${NC}   Hybrid Milvus collection (dense+sparse) → Endee
  ${YELLOW}qdrant-to-endee-dense${NC}    Dense Qdrant collection → Endee
  ${YELLOW}qdrant-to-endee-hybrid${NC}   Hybrid Qdrant collection (dense+sparse) → Endee

${GREEN}Required:${NC}
  --source_url URL              Source database URL
  --source_collection NAME      Source collection name
  --target_api_key KEY          Endee API key
  --target_collection NAME      Endee index name
  --ef_construct N              HNSW ef_construct (auto-detected if omitted)
  --resume                      Clear checkpoint and start fresh (RESUME=false)
  --precision PREC              float32 | float16 | int8 | int16 | binary
  --M N                         HNSW M (auto-detected from source if omitted)
  --space_type TYPE             cosine | l2 | ip (default: cosine)

${GREEN}Optional:${NC}
  --source_api_key KEY          Source DB API key (default: "")
  --source_db NAME              Milvus DB name (default: "default")
  --batch_size N                Fetch batch size (default: 1000)
  --upsert_size N               Endee upsert chunk size (default: 1000)
  --max_queue_size N            Queue depth for backpressure (default: 5)
  --checkpoint_file PATH        Checkpoint file (default: ./migration_checkpoint.json)
  --debug                       Verbose logging

${GREEN}Environment variables:${NC}
  All --flags above map 1-to-1 with env vars (upper-cased, e.g. SOURCE_URL).
  MIGRATION_TYPE can replace the positional arg.

${GREEN}Examples:${NC}
  ${CYAN}# Milvus dense → Endee${NC}
  docker run vector-migration ${YELLOW}milvus-to-endee-dense${NC} \\
    --source_url http://localhost:19530 \\
    --source_collection my_collection \\
    --target_api_key "ek-..." \\
    --target_collection my_index \\
    --precision int16

  ${CYAN}# Qdrant hybrid → Endee, resume from checkpoint${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}qdrant-to-endee-hybrid${NC} \\
    --source_url http://localhost:6333 \\
    --source_collection hybrid_col \\
    --target_api_key "ek-..." \\
    --target_collection hybrid_index \\
    --precision int16

  ${CYAN}# Fresh start (clear checkpoint)${NC}
  docker run vector-migration ${YELLOW}milvus-to-endee-dense${NC} --resume ...

EOF
}

# Positional migration type (CLI beats env var)
if [ -n "$1" ] && [[ "$1" != --* ]]; then
    MIGRATION_TYPE="$1"; shift
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help; exit 0
fi

if [ -z "$MIGRATION_TYPE" ]; then
    print_error "No migration type specified."
    echo "Pass it as the first argument or set MIGRATION_TYPE in your env."
    show_help; exit 1
fi

echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         Endee Migration Tool v2.0                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
print_info "Migration Type: ${YELLOW}${MIGRATION_TYPE}${NC}"
echo ""

# Show env-based config summary
[ -n "$SOURCE_URL" ]        && echo "  Source URL        : $SOURCE_URL"
[ -n "$SOURCE_COLLECTION" ] && echo "  Source Collection : $SOURCE_COLLECTION"
[ -n "$TARGET_COLLECTION" ] && echo "  Target Collection : $TARGET_COLLECTION"
[ -n "$BATCH_SIZE" ]        && echo "  Batch Size        : $BATCH_SIZE"
[ -n "$PRECISION" ]         && echo "  Precision         : $PRECISION"
echo ""

exec python /app/migrate.py "$MIGRATION_TYPE" "$@"