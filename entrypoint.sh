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

# ─────────────────────────────────────────────────────────────────────────────
# 4. Map migration type → script
# ─────────────────────────────────────────────────────────────────────────────
case $MIGRATION_TYPE in
    milvus-to-endee-dense)
        print_info "Migration Type: ${YELLOW}Milvus (Dense)${NC} → Endee"
        SCRIPT="/app/scripts/milvus_to_endee_dense_migration.py"
        ;;
    milvus-to-endee-hybrid)
        print_info "Migration Type: ${YELLOW}Milvus (Hybrid)${NC} → Endee"
        SCRIPT="/app/scripts/milvus_to_endee_hybrid_migration.py"
        ;;
    qdrant-to-endee-dense)
        print_info "Migration Type: ${YELLOW}Qdrant (Dense)${NC} → Endee"
        SCRIPT="/app/scripts/qdrant_to_endee_dense_migration.py"
        ;;
    qdrant-to-endee-hybrid)
        print_info "Migration Type: ${YELLOW}Qdrant (Hybrid)${NC} → Endee"
        SCRIPT="/app/scripts/qdrant_to_endee_hybrid_migration.py"
        ;;
    chroma-to-endee-hybrid)
        print_info "Migration Type: ${YELLOW}ChromaDB (Dense → Hybrid)${NC} → Endee"
        print_info "Sparse vectors: ${CYAN}endee/bm25${NC} (endee-model SparseModel)"
        SCRIPT="/app/scripts/chroma_to_endee_hybrid_migration.py"
        ;;
    *)
        print_error "Unknown migration type: ${RED}$MIGRATION_TYPE${NC}"
        echo ""
        echo "Valid types:"
        echo "  • milvus-to-endee-dense"
        echo "  • milvus-to-endee-hybrid"
        echo "  • qdrant-to-endee-dense"
        echo "  • qdrant-to-endee-hybrid"
        echo "  • chroma-to-endee-hybrid"
        echo ""
        echo "Run 'docker run vector-migration --help' for more information"
        exit 1
        ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check script exists
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    print_error "Migration script not found: ${RED}$SCRIPT${NC}"
    echo ""
    print_warning "Ensure the following files exist in /app/scripts/:"
    echo "  • milvus_to_endee_dense_migration.py"
    echo "  • milvus_to_endee_hybrid_migration.py"
    echo "  • qdrant_to_endee_dense_migration.py"
    echo "  • qdrant_to_endee_hybrid_migration.py"
    echo "  • chroma_to_endee_hybrid_migration.py"
    exit 1
fi

print_success "Found migration script: $(basename $SCRIPT)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 6. Show effective configuration summary (from env vars)
# ─────────────────────────────────────────────────────────────────────────────
print_header "Effective configuration:"
[ ! -z "$SOURCE_URL" ]        && echo "  Source URL        : $SOURCE_URL"
[ ! -z "$SOURCE_PORT" ]       && echo "  Source Port       : $SOURCE_PORT"
[ ! -z "$SOURCE_PATH" ]       && echo "  Source Path       : $SOURCE_PATH"
[ ! -z "$SOURCE_COLLECTION" ] && echo "  Source Collection : $SOURCE_COLLECTION"
[ ! -z "$TARGET_URL" ]        && echo "  Target URL        : $TARGET_URL"
[ ! -z "$TARGET_COLLECTION" ] && echo "  Target Collection : $TARGET_COLLECTION"
[ ! -z "$SPACE_TYPE" ]        && echo "  Space Type        : $SPACE_TYPE"
[ ! -z "$M" ]                 && echo "  M                 : $M"
[ ! -z "$EF_CONSTRUCT" ]      && echo "  EF Construct      : $EF_CONSTRUCT"
[ ! -z "$PRECISION" ]         && echo "  Precision         : $PRECISION"
[ ! -z "$BATCH_SIZE" ]        && echo "  Batch Size        : $BATCH_SIZE"
[ ! -z "$UPSERT_SIZE" ]       && echo "  Upsert Size       : $UPSERT_SIZE"
[ ! -z "$FILTER_FIELDS" ]     && echo "  Filter Fields     : $FILTER_FIELDS"
echo ""

# Show env-based config summary
[ -n "$SOURCE_URL" ]        && echo "  Source URL        : $SOURCE_URL"
[ -n "$SOURCE_COLLECTION" ] && echo "  Source Collection : $SOURCE_COLLECTION"
[ -n "$TARGET_COLLECTION" ] && echo "  Target Collection : $TARGET_COLLECTION"
[ -n "$BATCH_SIZE" ]        && echo "  Batch Size        : $BATCH_SIZE"
[ -n "$PRECISION" ]         && echo "  Precision         : $PRECISION"
echo ""

exec python /app/migrate.py "$MIGRATION_TYPE" "$@"