"""
Pipeline Configuration Registry

Maps context names to pipeline configurations following CIE pattern.
Allows easy addition of new contexts (e.g., "ESCALATION", "FEEDBACK").
"""

from typing import Any

from categorization.pipelines.sentiment.constants import SENTIMENT_CTX
from categorization.pipelines.sentiment.context import SENTIMENT_QUERIES_CONFIG
from categorization.pipelines.sentiment.labeling import LABELING_CONFIG
from categorization.pipelines.sentiment.preprocess import PREPROCESS_CONFIG
from categorization.pipelines.sentiment.process_batches import PROCESS_BATCHES_CONFIG

# Registry dictionaries - map context name to configuration
QUERIES_BY_CONTEXT: dict[str, list[dict[str, Any]]] = {
    SENTIMENT_CTX: SENTIMENT_QUERIES_CONFIG,
}

PREPROCESS_BY_CONTEXT: dict[str, list[dict[str, Any]]] = {
    SENTIMENT_CTX: PREPROCESS_CONFIG,
}

LABELING_BY_CONTEXT: dict[str, list[dict[str, Any]]] = {
    SENTIMENT_CTX: LABELING_CONFIG,
}

PROCESS_BATCHES_BY_CONTEXT: dict[str, list[dict[str, Any]]] = {
    SENTIMENT_CTX: PROCESS_BATCHES_CONFIG,
}
