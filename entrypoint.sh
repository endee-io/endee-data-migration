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
  docker run vector-migration --from DB --to DB --type TYPE [OPTIONS]

${GREEN}Required:${NC}
  --from DB             Source database:  milvus | qdrant
  --to   DB             Target database:  endee
  --type TYPE           Vector type:      dense | hybrid

${GREEN}Source options:${NC}
  --source_url URL
  --source_api_key KEY
  --source_collection NAME
  --source_port PORT
  --source_db NAME          Milvus database name (default: default)
  --filter_fields F1,F2     Payload fields to expose as filter attributes
  --use_https               Use HTTPS for Qdrant

${GREEN}Target options:${NC}
  --target_url URL
  --target_api_key KEY
  --target_collection NAME

${GREEN}Index options:${NC}
  --space_type cosine|l2|ip   (default: cosine)
  --M N                       HNSW M (auto-detected if omitted)
  --ef_construct N            HNSW ef_construct (auto-detected if omitted)
  --precision float32|float16|int8|int16|binary

${GREEN}Performance:${NC}
  --batch_size N          Fetch batch size (default: 1000)
  --upsert_size N         Endee chunk size (default: 100)
  --max_queue_size N      Queue depth (default: 5)

${GREEN}Checkpoint:${NC}
  --checkpoint_file PATH  (default: ./migration_checkpoint.json)
  --resume                Clear checkpoint, start fresh (RESUME=false)

${GREEN}Env var equivalents:${NC}
  FROM_DB, TO_DB, VECTOR_TYPE, SOURCE_URL, SOURCE_API_KEY,
  SOURCE_COLLECTION, TARGET_URL, TARGET_API_KEY, TARGET_COLLECTION,
  PRECISION, BATCH_SIZE, UPSERT_SIZE, RESUME, DEBUG

${GREEN}Examples:${NC}
  ${CYAN}# Milvus dense → Endee${NC}
  docker run vector-migration \\
    --from milvus --to endee --type dense \\
    --source_url http://localhost:19530 \\
    --source_collection my_col \\
    --target_api_key "ek-..." \\
    --target_collection my_index \\
    --precision int16

  ${CYAN}# Qdrant hybrid → Endee, via env vars${NC}
  docker run --env-file .env vector-migration \\
    --from qdrant --to endee --type hybrid

  ${CYAN}# Fresh start (clear checkpoint)${NC}
  docker run vector-migration --from milvus --to endee --type dense --resume ...
EOF
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help; exit 0
fi

echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         Endee Migration Tool v2.0                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

[ -n "$SOURCE_URL" ]        && echo "  Source URL        : $SOURCE_URL"
[ -n "$SOURCE_COLLECTION" ] && echo "  Source Collection : $SOURCE_COLLECTION"
[ -n "$TARGET_COLLECTION" ] && echo "  Target Collection : $TARGET_COLLECTION"
[ -n "$PRECISION" ]         && echo "  Precision         : $PRECISION"
echo ""

exec python /app/migrate.py "$@"