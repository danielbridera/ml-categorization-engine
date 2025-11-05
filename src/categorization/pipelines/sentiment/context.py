"""
Context Query Configuration for Sentiment Pipeline

Defines BigQuery extraction configuration.
"""

from categorization.pipelines.sentiment.constants import (
    SENTIMENT_CONTEXT_OUTPUT_FILE,
    SENTIMENT_CONTEXT_PATH,
    SENTIMENT_CONTEXT_QUERY_NAME,
)

# Configuration for Step 1: BigQuery Extraction
SENTIMENT_QUERIES_CONFIG = [
    {
        "name": SENTIMENT_CONTEXT_QUERY_NAME,
        "query_path": SENTIMENT_CONTEXT_PATH,
        "output_file": SENTIMENT_CONTEXT_OUTPUT_FILE,
    }
]
