# Rebuild

# # Milvus Dense
# docker run -v $(pwd)/data:/app/data vector-migration:latest \
#   milvus-to-endee-dense \
#   --source-url http://localhost:19530 \
#   --source-collection test \
#   --target-api-key "key" \
#   --target-collection index

# # Milvus Hybrid
# docker build -t vector-migration:latest .
# docker run \
#   --rm \
#   --network vector-net \
#   -v $(pwd)/data:/app/data \
#   vector-migration:latest \
#   milvus-to-endee-hybrid \
#   --source_url http://milvus-standalone \
#   --source_port 19530 \
#   --source_api_key "" \
#   --source_collection milvus_hybrid_1536 \
#   --target_url http://endee-oss:8080 \
#   --target_api_key "" \
#   --target_collection docker_milvus_hybrid_1536 \
#   --batch_size 1000 \
#   --upsert_size 1000

# # Qdrant Dense
# docker run -v $(pwd)/data:/app/data vector-migration:latest \
#   qdrant-to-endee-dense \
#   --source-url http://localhost:6333 \
#   --source-collection test \
#   --target-api-key "key" \
#   --target-collection index

# # Qdrant Hybrid
# docker run -v $(pwd)/data:/app/data vector-migration:latest \
#   qdrant-to-endee-hybrid \
#   --source-url http://localhost:6333 \
#   --source-collection test \
#   --target-api-key "key" \
#   --target-collection index

docker build -t vector-migration:latest .
docker run \
  --rm \
  --network vector-net \
  --env-file .env.dev \
  -v $(pwd)/data:/app/data \
  vector-migration:latest \
  milvus-to-endee-hybrid
  # qdrant-to-endee-hybrid
  # qdrant-to-endee-hybrid
  # milvus-to-endee-hybrid