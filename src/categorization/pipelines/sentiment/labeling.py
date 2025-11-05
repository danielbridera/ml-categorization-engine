"""
Labeling Configuration for Sentiment Pipeline

Defines OpenAI Batch API submission configuration.
"""

from categorization.pipelines.sentiment.constants import (
    SENTIMENT_BATCH_REQUESTS_FILE,
    SENTIMENT_LABELING_NAME,
    SENTIMENT_PREPROCESSED_OUTPUT_FILE,
)

# Sentiment analysis prompts
SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment analysis expert. Respond only with the sentiment category."
)

SENTIMENT_USER_PROMPT = """Analyze the sentiment of the following user message and classify it as one of these categories:
- positive: Expresses satisfaction, gratitude, happiness, or approval
- negative: Expresses frustration, anger, disappointment, or complaint
- neutral: Factual, informational, or no clear emotional tone
- mixed: Contains both positive and negative sentiments

Respond with ONLY the category name (positive, negative, neutral, or mixed), nothing else.

User message: {message}"""

# Configuration for Step 3a: Submit Batch to OpenAI
LABELING_CONFIG = [
    {
        "name": SENTIMENT_LABELING_NAME,
        "input_file": SENTIMENT_PREPROCESSED_OUTPUT_FILE,  # From Step 2
        "batch_requests_file": SENTIMENT_BATCH_REQUESTS_FILE,
        "model": "gpt-4o-mini",  # Default model
        "system_prompt": SENTIMENT_SYSTEM_PROMPT,
        "user_prompt_template": SENTIMENT_USER_PROMPT,
        "temperature": 0,  # Deterministic responses
        "max_tokens": 10,  # Only need 1 word
        "task_description": "sentiment_analysis",  # For OpenAI metadata
    }
]
