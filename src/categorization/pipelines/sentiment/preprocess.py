"""
Preprocessing Configuration for Sentiment Pipeline

Defines data cleaning and deduplication configuration.
"""

from categorization.pipelines.sentiment.constants import (
    SENTIMENT_CONTEXT_OUTPUT_FILE,
    SENTIMENT_PREPROCESSED_OUTPUT_FILE,
    SENTIMENT_PREPROCESSING_NAME,
)

# Configuration for Step 2: Preprocessing
PREPROCESS_CONFIG = [
    {
        "name": SENTIMENT_PREPROCESSING_NAME,
        "input_file": SENTIMENT_CONTEXT_OUTPUT_FILE,  # From Step 1
        "output_file": SENTIMENT_PREPROCESSED_OUTPUT_FILE,  # For Step 3a
    }
]
