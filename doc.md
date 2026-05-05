# Endee Migration Tool - Documentation

Migrate vector data from **Qdrant** or **Milvus** into **Endee**. The tool ships as a Docker image and supports dense-only and hybrid (dense + sparse) collections.

---



## Setup
Make sure both Endee and Source DB servers are up and reachable from machine where migration script will run.

### Deployment Scenarios

---

#### A - Migration and Endee on different servers

This is the simplest case. No shared Docker network is needed. Both servers just need to be reachable over the network.

**On the migration server:**

**Step 1 - Clone Migration Script:**
```bash
git clone https://github.com/endee-io/endee-data-migration.git
cd endee-data-migration/
```

**Step 2 - Create .env from .env.example:**
```bash
cp .env.example .env
```
de
**Step 3 - Update .env file:**
```bash
# .env
MIGRATION_TYPE=milvus-to-endee-dense
SOURCE_URL=http://<milvus-server-ip>
SOURCE_PORT=19530
SOURCE_COLLECTION=my_collection # SOURCE DATABASE COLLECTION NAME FROM WHERE DATA NEEDS TO MIGRATE
TARGET_URL=http://<endee-server-ip>:8080 # IP OF ENDEE SERVER IS RUNNING ON
TARGET_API_KEY=your-endee-api-key # IF USING ENDEE DEV SERVER GET THE API KEY (NO NEED TO SET TARGET URL IN THAT CASE )
TARGET_COLLECTION=my_index # ENDEE INDEX NAME
CLEAR_CHECKPOINT=true # FALSE IF YOU WANT TO RESUME THE MIGRATION
FILTER_FIELDS=field1,field2 # ADD FILTER_FIELDS ONLY IF YOU WANT FIELDS TO BE IN ENDEE FILTER AND NOT IN META
PRECISION=INT16 
```


**Step 4 - Make scripts executable:**
```bash
chmod +x entrypoint.sh
chmod +x cmd.sh
```

**Step 5 - Start Migration:**

```bash
./cmd.sh
```

Make sure:
- Port `6333` (Qdrant) or `19530` (Milvus) is open on the source server.
- The Endee API port is open on the Endee server.
- The migration container has outbound access to both.

---

#### B - Both on the same server, both in Docker

Both containers must be on the same Docker network so they can reach each other by container name.

**Step 1 - Create the shared network (if not already done):**

```bash
docker network create vector-net
```

**Step 2 - Clone Migration Repo:**
```bash
git clone https://github.com/endee-io/endee-data-migration.git
cd endee-data-migration/
```
**Step 3 - Start your Endee container on that network:**
Add this at end of endee docker-compose.yml
```bash
networks:
  vector-net:
    external: true
```
or run

```bash
docker run \
  --ulimit nofile=100000:100000 \
  --network vector-net \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```


**Step 4 - Run the migration on the same network, using container names as hostnames:**

```bash
# .env
MIGRATION_TYPE=milvus-to-endee-dense
SOURCE_URL=http://milvus       # container name as hostname
SOURCE_PORT=9091
SOURCE_COLLECTION=my_collection
TARGET_URL=http://endee:8080         # container name as hostname
TARGET_COLLECTION=my_index
CLEAR_CHECKPOINT=true # FALSE IF YOU WANT TO RESUME THE MIGRATION
FILTER_FIELDS=field1,field2 # ADD FILTER_FIELDS ONLY IF YOU WANT FIELDS TO BE IN ENDEE FILTER AND NOT IN META
PRECISION=INT16
```

**Step 5 - Make scripts executable:**

```bash
chmod +x entrypoint.sh
chmod +x cmd.sh
```

**Step 6 - Start Migration:**

```bash
./cmd.sh
```

The `docker-compose.yml` already joins `vector-net` by default, so if you use `docker compose up` it will work as long as the network exists and your other containers are on it.

---

#### C - Same server, Endee not in Docker

**Step 1 - Clone Migration Repo:**
```bash
git clone https://github.com/endee-io/endee-data-migration.git
cd endee-data-migration/
```


**Step 2 - Copy .env.example to .env:**
```bash
cp .env.example .env
```

**Step 3 - Update .env:**
```bash
# .env
MIGRATION_TYPE=milvus-to-endee-dense
SOURCE_URL=http://<milvus-ip>
SOURCE_PORT=9091
SOURCE_COLLECTION=my_collection
TARGET_URL=http://<IP-of-Endee>:8080   # reaches the host directly
TARGET_COLLECTION=my_index
CLEAR_CHECKPOINT=true # FALSE IF YOU WANT TO RESUME THE MIGRATION
FILTER_FIELDS=field1,field2 # ADD FILTER_FIELDS ONLY IF YOU WANT FIELDS TO BE IN ENDEE FILTER AND NOT IN META
PRECISION=INT16
```


---

## Running the Migration

### Available Migration Types

| Type | Source | Vector Mode |
|---|---|---|
| `milvus-to-endee-dense` | Milvus | Dense vectors only |
| `milvus-to-endee-hybrid` | Milvus | Dense + sparse vectors |
| `qdrant-to-endee-dense` | Qdrant | Dense vectors only |
| `qdrant-to-endee-hybrid` | Qdrant | Named dense (`dense`) + sparse (`sparse_keywords`) vectors |

Pass the type as the first argument to the container, or set `MIGRATION_TYPE` in your `.env`.

### Override precision manually
- Set `PRECISION` in your `.env` to override the auto-detected value.
- Valid values: `FLOAT32`, `FLOAT16`, `INT16`, `INT8`
  - Use INT16 if you want a good balance of accuracy and storage efficiency and the source does not have explicit quantization. 
  - Use float32 to preserve the full original precision.


### Checkpoint and Resume

The script saves progress to a JSON file after every successful batch. If the migration is interrupted (network error, container restart, manual stop), re-run the exact same command with `--clear_checkpoint` set as `False` and it will resume from where it left off.

The checkpoint file is written to `/app/data/checkpoints/migration.json` inside the container, which maps to `./data/checkpoints/migration.json` on the host when you mount `-v $(pwd)/data:/app/data`.

### FILTER FIELDS
- Each record in the source database has a payload (Qdrant) or metadata fields (Milvus). When migrated to Endee, these fields are stored in one of two places.
  - `filter` - fields you want to use for filtering queries in Endee.
  - `meta` - everything else, stored as metadata but not filterable.
- By default, all fields go into meta. To move specific fields into filter, set `FILTER_FIELDS` to a comma-separated list of field names.
- ``` FILTER_FIELDS=category,status,year```
  - With this set, category, status, and year will be stored in filter, and all remaining payload fields will go into meta.
- If you do not set `FILTER_FIELDS`, all payload fields are stored in meta and the filter field will be empty.
---

## Limitations
- OS should be Linux based
- No incremental migration - new records added to the source after migration started will not be picked up. Run a fresh migration with `--clear_checkpoint` as `True` to re-migrate.
- No real-time sync - this is a one-time, point-in-time copy only.
- Do not run the source database and Endee on the same server - resource contention will degrade performance and risk crashes.
- Multivector mode is not supported.
- Only Qdrant and Milvus are supported as sources.
- Only Endee is supported as the target.
---

## Infrastructure Requirements

### Migration + Endee on the same server

These are the combined requirements when both the migration container and Endee are running on the same machine.

| Dataset Size | RAM | CPU | Disk |
|---|---|---|---|
| 1 million records | 8 GB | 4 cores | 20 GB |
| 10 million records | 32 GB | 8 cores | 50 GB |

Disk usage covers the Endee index data, checkpoint files, and container images. Most of the RAM and CPU is consumed by Endee itself, not the migration script.

---

### Migration running on a separate server

When the migration container runs on its own dedicated server (no Endee on that machine), the resource requirements are much lighter. The script uses an async producer-consumer queue with a default max size of 5 batches. Each batch holds 1,000 records. At float32 precision and dimension 768, one batch is roughly 3 MB in memory, so the queue holds at most ~15 MB of vector data at any time. The script is network-bound, not CPU or memory-bound.

| Dataset Size | RAM | CPU | Disk |
|---|---|---|---|
| 1 million records | 4 GB | 1-2 cores | 4 GB |
| 10 million records | 8 GB | 2 cores | 8 GB |

You can reduce memory further by lowering `BATCH_SIZE` and `MAX_QUEUE_SIZE` if needed.
