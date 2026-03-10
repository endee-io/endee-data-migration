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

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Function to show help
show_help() {
    cat << EOF
${CYAN}╔════════════════════════════════════════════════════════════════╗
║           Endee Migration Tool v1.0.0                ║
║                    Milvus & Qdrant → Endee                     ║
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

${GREEN}Required Options:${NC}
  --source-url URL              Source database URL
  --source-collection NAME      Source collection name
  --target-api-key KEY          Endee API key
  --target-collection NAME      Endee index name

${GREEN}Optional Options:${NC}
  --source-api-key KEY          Source database API key (default: "")
  --batch-size SIZE             Fetch batch size (default: 1000)
  --upsert-size SIZE            Upsert batch size (default: 1000)
  --space-type TYPE             Distance metric: cosine, L2, IP (default: cosine)
  --M VALUE                     HNSW M parameter (default: 16)
  --ef-construct VALUE          HNSW ef_construct (default: 128)
  --checkpoint-file PATH        Checkpoint file path
  --clear-checkpoint            Clear checkpoint and start fresh
  --debug                       Enable debug logging

${GREEN}Environment Variables:${NC}
  SOURCE_URL                    Can replace --source-url
  SOURCE_API_KEY                Can replace --source-api-key
  SOURCE_COLLECTION             Can replace --source-collection
  TARGET_API_KEY                Can replace --target-api-key
  TARGET_COLLECTION             Can replace --target-collection
  BATCH_SIZE                    Can replace --batch-size
  UPSERT_SIZE                   Can replace --upsert-size

${GREEN}Examples:${NC}

  ${CYAN}# Milvus Hybrid → Endee${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}milvus-to-endee-hybrid${NC} \\
    --source-url http://localhost:19530 \\
    --source-collection my_hybrid_collection \\
    --target-api-key "endee-api-key-123" \\
    --target-collection my_index \\
    --batch-size 1000

  ${CYAN}# Qdrant Dense → Endee${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}qdrant-to-endee-dense${NC} \\
    --source-url http://localhost:6333 \\
    --source-collection my_dense_collection \\
    --target-api-key "endee-api-key-123" \\
    --target-collection my_index

  ${CYAN}# Using environment variables${NC}
  docker run --env-file .env -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}milvus-to-endee-hybrid${NC}

  ${CYAN}# Resume from checkpoint${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}milvus-to-endee-hybrid${NC} \\
    --source-collection my_collection \\
    --target-collection my_index

  ${CYAN}# Start fresh (clear checkpoint)${NC}
  docker run -v \$(pwd)/data:/app/data vector-migration \\
    ${YELLOW}milvus-to-endee-hybrid${NC} \\
    --source-collection my_collection \\
    --target-collection my_index \\
    --clear-checkpoint

${GREEN}Getting Help:${NC}
  docker run vector-migration --help
  docker run vector-migration ${YELLOW}<MIGRATION_TYPE>${NC} --help

${GREEN}Documentation:${NC}
  README:          /app/README.md
  Quick Reference: /app/QUICK_REFERENCE.md
  Deployment:      /app/DEPLOYMENT.md

${GREEN}Support:${NC}
  GitHub: https://github.com/yourusername/vector-migration
  Issues: https://github.com/yourusername/vector-migration/issues

EOF
}

# 1. Read migration type — CLI arg takes priority, fall back to env var
if [ ! -z "$1" ] && [[ "$1" != --* ]]; then
    MIGRATION_TYPE=$1
    shift
fi
# MIGRATION_TYPE from .env is already available if not set above

# 2. Now check help — only show if explicitly requested, not just because $1 is empty
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# 3. Validate
if [ -z "$MIGRATION_TYPE" ]; then
    print_error "No migration type specified."
    echo "Set MIGRATION_TYPE in .env or pass it as an argument."
    show_help
    exit 1
fi

# Print banner
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           Endee Migration Tool v1.0.0                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Determine which script to run
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
    *)
        print_error "Unknown migration type: ${RED}$MIGRATION_TYPE${NC}"
        echo ""
        echo "Valid types:"
        echo "  • milvus-to-endee-dense"
        echo "  • milvus-to-endee-hybrid"
        echo "  • qdrant-to-endee-dense"
        echo "  • qdrant-to-endee-hybrid"
        echo ""
        echo "Run 'docker run vector-migration --help' for more information"
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    print_error "Migration script not found: ${RED}$SCRIPT${NC}"
    echo ""
    print_warning "Please ensure the following files exist in scripts/:"
    echo "  • milvus_to_endee_dense_migration.py"
    echo "  • milvus_to_endee_hybrid_migration.py"
    echo "  • qdrant_to_endee_dense_migration.py"
    echo "  • qdrant_to_endee_hybrid_migration.py"
    exit 1
fi

print_success "Found migration script"
echo ""

# Show configuration summary if environment variables are set
if [ ! -z "$SOURCE_URL" ] || [ ! -z "$SOURCE_COLLECTION" ]; then
    print_header "Configuration from Environment:"
    [ ! -z "$SOURCE_URL" ] && echo "  Source URL:        $SOURCE_URL"
    [ ! -z "$SOURCE_PORT" ] && echo "  Source Port:       $SOURCE_PORT"
    [ ! -z "$SOURCE_COLLECTION" ] && echo "  Source Collection: $SOURCE_COLLECTION"
    [ ! -z "$TARGET_COLLECTION" ] && echo "  Target Collection: $TARGET_COLLECTION"
    [ ! -z "$BATCH_SIZE" ] && echo "  Batch Size:        $BATCH_SIZE"
    echo ""
fi

# Run the migration script
print_info "Executing migration..."
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Execute the script with all remaining arguments
exec python "$SCRIPT" "$@"