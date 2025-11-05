"""
Process Batches Configuration for Sentiment Pipeline

Defines batch results processing configuration.
"""

from categorization.pipelines.sentiment.constants import (
    SENTIMENT_BATCH_RESULTS_FILE,
    SENTIMENT_LABELED_OUTPUT_FILE,
    SENTIMENT_PREPROCESSED_OUTPUT_FILE,
    SENTIMENT_PROCESS_BATCHES_NAME,
)

# Configuration for Step 3b: Download and Process Batch Results
PROCESS_BATCHES_CONFIG = [
    {
        "name": SENTIMENT_PROCESS_BATCHES_NAME,
        "input_file": SENTIMENT_PREPROCESSED_OUTPUT_FILE,  # Original data
        "batch_results_file": SENTIMENT_BATCH_RESULTS_FILE,  # Downloaded results
        "output_file": SENTIMENT_LABELED_OUTPUT_FILE,  # Final labeled data
        "output_column": "sentiment",  # Column name for labels
    }
]
