docker network create vector-net 2>/dev/null || true
docker build -t vector-migration:latest .
docker compose up --build
# docker run \
#   --rm \
#   --network vector-net \
#   --env-file .env.dev \
#   -v $(pwd)/data:/app/data \
#   vector-migration:latest