"""
Constants for Sentiment Pipeline

All constants for the sentiment analysis pipeline, following CIE pattern.
"""

from categorization.settings.queries import CONTEXT_QUERIES_PATH

# Context identifier (used in registry)
SENTIMENT_CTX = "SENTIMENT"

# SQL paths
SENTIMENT_QUERIES_PATH = CONTEXT_QUERIES_PATH / "sentiment"
SENTIMENT_CONTEXT_PATH = SENTIMENT_QUERIES_PATH / "sentiment_context.sql"

# Pipeline step names (for logging)
SENTIMENT_CONTEXT_QUERY_NAME = "Sentiment Context"
SENTIMENT_PREPROCESSING_NAME = "Sentiment Preprocessing"
SENTIMENT_LABELING_NAME = "Sentiment Labeling"
SENTIMENT_PROCESS_BATCHES_NAME = "Sentiment Process Batches"

# File names - Input/Output for each pipeline step
SENTIMENT_CONTEXT_OUTPUT_FILE = "sentiment_context.parquet"
SENTIMENT_PREPROCESSED_OUTPUT_FILE = "sentiment_preprocessed.parquet"
SENTIMENT_LABELED_OUTPUT_FILE = "sentiment_labeled.parquet"

# Batch file names
SENTIMENT_BATCH_REQUESTS_FILE = "batch_requests.jsonl"
SENTIMENT_BATCH_RESULTS_FILE = "batch_results.jsonl"

# Metadata file names
SENTIMENT_METADATA_FILE = "metadata.json"
SENTIMENT_BATCH_INFO_FILE = "batch_info.json"
