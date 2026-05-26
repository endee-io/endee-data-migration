#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_header()  { echo -e "${CYAN}$1${NC}"; }

# ─────────────────────────────────────────────────────────────────────────────
show_help() {
    cat << EOF
${CYAN}╔════════════════════════════════════════════════════════════════╗
║           Endee Migration Tool v1.1.0                          ║
║          Milvus, Qdrant & ChromaDB → Endee                     ║
╚════════════════════════════════════════════════════════════════╝${NC}

${GREEN}Usage:${NC}
  docker run [docker-options] vector-migration ${YELLOW}<MIGRATION_TYPE>${NC} [OPTIONS]

${GREEN}Available Migration Types:${NC}

  ${YELLOW}milvus-to-endee-dense${NC}
    Migrate Milvus collection (dense vectors only) to Endee

  ${YELLOW}milvus-to-endee-hybrid${NC}
    Migrate Milvus collection (dense + sparse vectors) to Endee

  ${YELLOW}qdrant-to-endee-dense${NC}
    Migrate Qdrant collection (dense vectors only) to Endee

  ${YELLOW}qdrant-to-endee-hybrid${NC}
    Migrate Qdrant collection (dense + sparse vectors) to Endee

  ${YELLOW}chroma-to-endee-hybrid${NC}
    Migrate ChromaDB collection (dense) to Endee HYBRID index.
    Sparse vectors generated using Endee's own BM25 model (endee/bm25)
    from the document text stored alongside embeddings in ChromaDB.

${GREEN}Required Options:${NC}
  --source-url URL              Source database URL
  --source-collection NAME      Source collection name
  --target-api-key KEY          Endee API key
  --target-collection NAME      Endee index name
  --M VALUE                     HNSW M parameter             ${RED}[REQUIRED — no default]${NC}
  --ef-construct VALUE          HNSW ef_construct            ${RED}[REQUIRED — no default]${NC}
  --space-type TYPE             Distance metric: cosine | l2 | ip  ${RED}[REQUIRED — no default]${NC}

${GREEN}Optional Options:${NC}
  --source-api-key KEY          Source database API key (default: "")
  --batch-size SIZE             Fetch batch size (default: 1000)
  --upsert-size SIZE            Upsert chunk size (default: 1000)
  --precision VALUE             Vector precision: float32 | int16 | int8 | binary
  --filter-fields FIELDS        Comma-separated metadata fields for Endee filter
  --checkpoint-file PATH        Checkpoint file path
  --clear-checkpoint            Clear checkpoint and start fresh
  --debug                       Enable debug logging

${GREEN}ChromaDB-specific options:${NC}
  --source-port PORT            ChromaDB HTTP port (default: 8000)
  --source-path PATH            Local PersistentClient path (skips host/port)
  --no-store-document           Don't store document text in meta.document

${GREEN}Environment Variables:${NC}
  SOURCE_URL              → --source-url
  SOURCE_API_KEY          → --source-api-key
  SOURCE_COLLECTION       → --source-collection
  SOURCE_PORT             → --source-port (ChromaDB)
  SOURCE_PATH             → --source-path (ChromaDB PersistentClient)
  TARGET_URL              → --target-url
  TARGET_API_KEY          → --target-api-key
  TARGET_COLLECTION       → --target-collection
  BATCH_SIZE              → --batch-size
  UPSERT_SIZE             → --upsert-size
  M                       → --M                 [REQUIRED]
  EF_CONSTRUCT            → --ef-construct      [REQUIRED]
  SPACE_TYPE              → --space-type        [REQUIRED]
  PRECISION               → --precision
  FILTER_FIELDS           → --filter-fields
  MAX_QUEUE_SIZE          → --max-queue-size
  MIGRATION_TYPE          → migration type (alternative to CLI arg)

${GREEN}Examples:${NC}

  ${CYAN}# ChromaDB → Endee Hybrid (HTTP client)${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}chroma-to-endee-hybrid${NC} \\
    --source-url      http://localhost:8000 \\
    --source-collection my_chroma_collection \\
    --target-api-key  "endee-api-key-123" \\
    --target-collection my_hybrid_index \\
    --space-type cosine --M 16 --ef-construct 128 \\
    --precision int16 \\
    --filter-fields category,source

  ${CYAN}# ChromaDB → Endee Hybrid (local persistent storage)${NC}
  docker run -v \$(pwd)/data:/app/data -v \$(pwd)/chroma:/chroma_data \\
    vector-migration ${YELLOW}chroma-to-endee-hybrid${NC} \\
    --source-path /chroma_data \\
    --source-collection my_collection \\
    --target-api-key  "endee-api-key-123" \\
    --target-collection my_hybrid_index \\
    --space-type cosine --M 16 --ef-construct 128

  ${CYAN}# Milvus Hybrid → Endee${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}milvus-to-endee-hybrid${NC} \\
    --source-url http://localhost:19530 \\
    --source-collection my_hybrid_collection \\
    --target-api-key "endee-api-key-123" \\
    --target-collection my_index \\
    --space-type cosine --M 16 --ef-construct 128

  ${CYAN}# Qdrant Dense → Endee${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}qdrant-to-endee-dense${NC} \\
    --source-url http://localhost:6333 \\
    --source-collection my_dense_collection \\
    --target-api-key "endee-api-key-123" \\
    --target-collection my_index \\
    --space-type cosine --M 16 --ef-construct 128

  ${CYAN}# Using environment variables / .env file${NC}
  docker run --env-file .env -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}chroma-to-endee-hybrid${NC}

  ${CYAN}# Resume from checkpoint${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}chroma-to-endee-hybrid${NC} \\
    --source-collection my_collection \\
    --target-collection my_index \\
    --space-type cosine --M 16 --ef-construct 128

  ${CYAN}# Start fresh (clear checkpoint)${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}chroma-to-endee-hybrid${NC} \\
    --source-collection my_collection \\
    --target-collection my_index \\
    --space-type cosine --M 16 --ef-construct 128 \\
    --clear-checkpoint

${GREEN}Documentation:${NC}
  README:          /app/README.md
  Quick Reference: /app/QUICK_REFERENCE.md
  Deployment:      /app/DEPLOYMENT.md

${GREEN}Support:${NC}
  GitHub: https://github.com/yourusername/vector-migration
  Issues: https://github.com/yourusername/vector-migration/issues

EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read migration type — CLI arg takes priority, fall back to env var
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -z "$1" ] && [[ "$1" != --* ]]; then
    MIGRATION_TYPE=$1
    shift
fi

# 2. Help check
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# 3. Validate migration type
if [ -z "$MIGRATION_TYPE" ]; then
    print_error "No migration type specified."
    echo "Set MIGRATION_TYPE in .env or pass it as the first argument."
    echo ""
    show_help
    exit 1
fi

# Print banner
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Endee Migration Tool v1.1.0                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

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

# ─────────────────────────────────────────────────────────────────────────────
# 7. Warn if required index params are missing (script itself will also exit)
# ─────────────────────────────────────────────────────────────────────────────
MISSING_PARAMS=0

if [ -z "$SPACE_TYPE" ]; then
    print_warning "SPACE_TYPE is not set. The migration script will exit with an error."
    print_warning "Set SPACE_TYPE=cosine (or l2 / ip) in your .env file or pass --space-type."
    MISSING_PARAMS=1
fi
if [ -z "$M" ]; then
    print_warning "M is not set. The migration script will exit with an error."
    print_warning "Set M=16 (or your value) in your .env file or pass --M."
    MISSING_PARAMS=1
fi
if [ -z "$EF_CONSTRUCT" ]; then
    print_warning "EF_CONSTRUCT is not set. The migration script will exit with an error."
    print_warning "Set EF_CONSTRUCT=128 (or your value) in your .env file or pass --ef-construct."
    MISSING_PARAMS=1
fi

if [ "$MISSING_PARAMS" -eq 1 ]; then
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. Execute
# ─────────────────────────────────────────────────────────────────────────────
print_info "Executing migration..."
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

exec python "$SCRIPT" "$@"