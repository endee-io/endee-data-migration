CHECKPOINT_FILE = "/app/data/checkpoints/migration.json"

# Collection config keys
DIMENSION_KEY = "dimension"
SPARSE_DIMENSION_KEY = "sparse_dimension"
SPACE_TYPE_KEY = "space_type"
PRECISION_KEY = "precision"
EF_CONSTRUCT_KEY = "ef_construct"
M_KEY = "M"
RECORD_COUNTS_KEY = "record_counts"

# Checkpoint keys
PROCESSED_COUNT_KEY = "processed_count"
LAST_OFFSET_KEY = "last_offset"
BATCH_NUMBER_KEY = "batch_number"

# Stats keys
FETCHED_KEY = "fetched"
UPSERTED_KEY = "upserted"
FAILED_KEY = "failed"
BATCHES_PROCESSED_KEY = "batches_processed"
START_TIME_KEY = "start_time"
RECORDS_WITH_SPARSE_KEY = "records_with_sparse"
RECORDS_WITHOUT_SPARSE_KEY = "records_without_sparse"

# Endee record field keys
ENDEE_ID_KEY = "id"
ENDEE_VECTOR_KEY = "vector"
ENDEE_FILTER_KEY = "filter"
ENDEE_META_KEY = "meta"
ENDEE_SPARSE_INDICES_KEY = "sparse_indices"
ENDEE_SPARSE_VALUES_KEY = "sparse_values"

# Qdrant hybrid vector field names
QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_SPARSE_VECTOR_NAME = "sparse_keywords"

# API paths
ENDEE_V1_API = "/api/v1"

# Default values
DEFAULT_SPACE_TYPE = "cosine"
DEFAULT_SPARSE_MODEL = "default"
DEFAULT_VECTOR_DIMENSION = 768
DEFAULT_SPARSE_DIMENSION = 30522
DEFAULT_SPARSE_DIMENSION_FALLBACK = 30000
DEFAULT_FETCH_BATCH_SIZE = 1000
DEFAULT_UPSERT_BATCH_SIZE = 1000
DEFAULT_MILVUS_PORT = 19530
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCT = 128
DEFAULT_PROCESSED_COUNT = 0
DEFAULT_LAST_OFFSET = None
DEFAULT_BATCH_NUMBER = 0
DEFAULT_MAX_QUEUE_SIZE = 5
COMPLETED_KEY = "completed"