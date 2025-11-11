"""
Classifier Registry - Unified Prompt-Driven Configuration

This module contains all text classification configurations in a single unified registry.
To add a new classifier, simply add a new entry to the CLASSIFIERS dictionary below.

Each classifier is defined by its prompts, output column, and optional model parameters.
The system handles file naming, SQL queries, and preprocessing automatically.
"""

from typing import Any, TypedDict


class ClassifierConfig(TypedDict, total=False):
    """
    Configuration for a text classification task.

    Required fields:
        context_name: Unique identifier for this classifier (e.g., "SENTIMENT")
        system_prompt: System instruction for the LLM
        user_prompt_template: User prompt template with {message} placeholder
        output_column: Name of the column for classification results

    Optional fields:
        model: OpenAI model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature (default: 0 for deterministic)
        max_tokens: Maximum tokens in response (default: 10)
        use_generic_query: Use generic SQL query (default: True)
        sql_query_path: Custom SQL query path if use_generic_query is False
        task_description: Optional description for OpenAI metadata
    """
    # Required
    context_name: str
    system_prompt: str
    user_prompt_template: str
    output_column: str

    # Optional with defaults
    model: str
    temperature: float
    max_tokens: int
    use_generic_query: bool
    sql_query_path: str
    task_description: str


# ============================================================================
# CLASSIFIER REGISTRY
# ============================================================================
# To add a new classifier, add a new entry here with the required fields.
# The system will automatically handle file naming and SQL queries.

CLASSIFIERS: dict[str, ClassifierConfig] = {
    # ------------------------------------------------------------------------
    # Sentiment Analysis
    # ------------------------------------------------------------------------
    "SENTIMENT": {
        "context_name": "SENTIMENT",
        "system_prompt": (
            "You are a sentiment analysis expert. "
            "Respond only with the sentiment category."
        ),
        "user_prompt_template": """Analyze the sentiment of the following user message and classify it as one of these categories:
- positive: Expresses satisfaction, gratitude, happiness, or approval
- negative: Expresses frustration, anger, disappointment, or complaint
- neutral: Factual, informational, or no clear emotional tone
- mixed: Contains both positive and negative sentiments

Respond with ONLY the category name (positive, negative, neutral, or mixed), nothing else.

User message: {message}""",
        "output_column": "sentiment",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
        "use_generic_query": True,
    },

    # ------------------------------------------------------------------------
    # Escalation Priority Classification
    # ------------------------------------------------------------------------
    "ESCALATION": {
        "context_name": "ESCALATION",
        "system_prompt": (
            "You are a customer support expert. "
            "Classify the urgency of customer messages."
        ),
        "user_prompt_template": """Classify the urgency of this customer message:
- urgent: Requires immediate attention (account locked, payment issues, critical bugs)
- high: Important but not time-critical (feature requests, billing questions)
- medium: General inquiries (how-to questions, documentation)
- low: Non-urgent feedback or casual conversation

Respond with ONLY the urgency level (urgent, high, medium, or low), nothing else.

User message: {message}""",
        "output_column": "escalation_priority",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
        "use_generic_query": True,
    },

    # ------------------------------------------------------------------------
    # Feedback Type Classification
    # ------------------------------------------------------------------------
    "FEEDBACK": {
        "context_name": "FEEDBACK",
        "system_prompt": "You are a product feedback analyzer.",
        "user_prompt_template": """Classify this feedback into categories:
- bug_report: Technical issues or errors
- feature_request: Suggestions for new features
- improvement: Suggestions for existing features
- praise: Positive feedback without requests
- question: User asking for help or clarification

Respond with ONLY the category name (bug_report, feature_request, improvement, praise, or question), nothing else.

User message: {message}""",
        "output_column": "feedback_type",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 15,
        "use_generic_query": True,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_classifier_config(context_name: str) -> ClassifierConfig:
    """
    Get configuration for a classifier by context name.

    Args:
        context_name: The context identifier (e.g., "SENTIMENT")

    Returns:
        ClassifierConfig dictionary

    Raises:
        ValueError: If context_name is not in the registry
    """
    if context_name not in CLASSIFIERS:
        available = ", ".join(sorted(CLASSIFIERS.keys()))
        raise ValueError(
            f"Unknown context: {context_name}. "
            f"Available contexts: {available}"
        )
    return CLASSIFIERS[context_name]


def list_classifiers() -> list[str]:
    """
    Get list of all available classifier context names.

    Returns:
        Sorted list of context names
    """
    return sorted(CLASSIFIERS.keys())


def get_classifier_info(context_name: str) -> dict[str, Any]:
    """
    Get human-readable information about a classifier.

    Args:
        context_name: The context identifier

    Returns:
        Dictionary with classifier information
    """
    config = get_classifier_config(context_name)
    return {
        "context_name": config["context_name"],
        "output_column": config["output_column"],
        "model": config.get("model", "gpt-4o-mini"),
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 10),
        "uses_generic_query": config.get("use_generic_query", True),
    }
