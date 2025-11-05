"""SQL query path definitions"""

from categorization.settings import BASE_PATH

# Base SQL directory
QUERY_PATH = BASE_PATH / "sql"

# Query subdirectories
CONTEXT_QUERIES_PATH = QUERY_PATH / "context"
CREATE_QUERY_PATH = QUERY_PATH / "create"
MERGE_QUERY_PATH = QUERY_PATH / "merge"
