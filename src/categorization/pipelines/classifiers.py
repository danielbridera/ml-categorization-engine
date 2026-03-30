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
    unit: str  # "message" (default) or "conversation"
    conversation_timeout: str  # e.g. "30m" — only used when unit="conversation"


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
    # Conversation Resolution Classification
    # ------------------------------------------------------------------------
    "CONVERSATION_RESOLUTION": {
        "context_name": "CONVERSATION_RESOLUTION",
        "unit": "conversation",
        "conversation_timeout": "30m",
        "system_prompt": (
            "You are a customer support quality analyst. "
            "Respond only with the resolution category."
        ),
        "user_prompt_template": """Classify whether this conversation was resolved:
- resolved: User's issue was fully addressed and they acknowledged it
- partially_resolved: Issue was addressed but user had remaining questions
- unresolved: Issue was not addressed or user left without resolution
- unclear: Not enough context to determine resolution

Respond with ONLY the category name (resolved, partially_resolved, unresolved, or unclear), nothing else.

Conversation:
{message}""",
        "output_column": "resolution_status",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 20,
        "use_generic_query": True,
    },
    # ------------------------------------------------------------------------
    # Conversation Satisfaction Classification
    # ------------------------------------------------------------------------
    "CONVERSATION_SATISFACTION": {
        "context_name": "CONVERSATION_SATISFACTION",
        "unit": "conversation",
        "conversation_timeout": "30m",
        "system_prompt": (
            "You are a customer experience analyst. "
            "Respond only with the satisfaction category."
        ),
        "user_prompt_template": """Assess customer satisfaction in this conversation:
- satisfied: Customer expressed satisfaction or issue was resolved smoothly
- neutral: Interaction was transactional with no clear satisfaction signals
- dissatisfied: Customer expressed frustration or left unresolved
- escalated: Customer requested human agent or expressed anger

Respond with ONLY the category name (satisfied, neutral, dissatisfied, or escalated), nothing else.

Conversation:
{message}""",
        "output_column": "satisfaction_level",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 20,
        "use_generic_query": True,
    },
    # ------------------------------------------------------------------------
    # Conversation Intent Classification
    # ------------------------------------------------------------------------
    "CONVERSATION_INTENT": {
        "context_name": "CONVERSATION_INTENT",
        "unit": "conversation",
        "conversation_timeout": "30m",
        "system_prompt": (
            "You are a customer intent classifier. "
            "Respond only with the intent category."
        ),
        "user_prompt_template": """Classify the primary intent of this customer conversation:
- technical_support: Reporting bugs, errors, or technical issues
- billing_and_payments: Questions about charges, invoices, or subscriptions
- account_management: Login, password, or account settings
- product_information: Questions about features or how something works
- complaint: Expressing dissatisfaction without a specific request
- general_inquiry: Other questions not fitting the above categories

Respond with ONLY the category name, nothing else.

Conversation:
{message}""",
        "output_column": "conversation_intent",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 25,
        "use_generic_query": True,
    },
    # ------------------------------------------------------------------------
    # Conversation Escalation — was the conversation escalated to a human?
    # Designed to stack on the same preprocessed CONVERSATIONS dataset.
    # ------------------------------------------------------------------------
    "CONVERSATION_ESCALATION": {
        "context_name": "CONVERSATION_ESCALATION",
        "system_prompt": (
            "You are a customer support analyst. "
            "Classify whether a chatbot conversation was escalated to a human agent."
        ),
        "user_prompt_template": """The following is a conversation between a user and an automated chatbot. Messages are labeled by speaker:
- "user:" lines are messages sent by the customer
- "flow:" lines are automated responses from the chatbot

Classify whether the user wanted or attempted to reach a live human agent. Focus on USER INTENT, not whether the bot successfully completed the transfer.

- escalated: The user explicitly requested to speak to a human agent at any point in the conversation. This includes direct requests (e.g. "asesor", "agente", "persona", "humano", "quiero hablar con alguien"), misspellings (e.g. "acesor", "assesor"), or the bot confirming a live transfer mid-chat (e.g. "te estoy transfiriendo con un agente")
- not_escalated: The user never requested a human and the conversation was handled entirely by the bot. IMPORTANT: if the bot assigns a queue turn number for a future in-person branch visit (e.g. "ya estás en la fila", "un asesor dirá tu nombre", "pronto un asesor dirá tu nombre"), this is NOT an escalation — the user is scheduling a visit, not requesting a human in this chat

Respond with ONLY the category name (escalated or not_escalated), nothing else.

Conversation:
{message}""",
        "output_column": "escalated_to_human",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
    },
    # ------------------------------------------------------------------------
    # Conversation Sentiment — user sentiment across the full conversation
    # Designed to stack on the same preprocessed CONVERSATIONS dataset.
    # ------------------------------------------------------------------------
    "CONVERSATION_SENTIMENT": {
        "context_name": "CONVERSATION_SENTIMENT",
        "system_prompt": (
            "You are a sentiment analysis expert. "
            "Respond only with the sentiment category."
        ),
        "user_prompt_template": """The following is a conversation between a user and an automated chatbot. Messages are labeled by speaker:
- "user:" lines are messages sent by the customer
- "flow:" lines are automated responses from the chatbot

Analyze the overall sentiment expressed by the user throughout the conversation and classify it as one of these categories:
- positive: Expresses satisfaction, gratitude, happiness, or approval
- negative: Expresses frustration, anger, disappointment, or complaint
- neutral: Factual, informational, or no clear emotional tone
- mixed: Contains both positive and negative sentiments

Respond with ONLY the category name (positive, negative, neutral, or mixed), nothing else.

Conversation:
{message}""",
        "output_column": "sentiment",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 10,
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
# EXTRACTION CONTEXTS
# ============================================================================
# Named data extraction contexts that are NOT classifiers — they define where
# and how to pull data, independent of any labeling task. Add entries here to
# create new first-class extraction contexts accessible via --context_name.

EXTRACTION_CONTEXTS: dict[str, dict] = {
    # Generic conversation extraction from dev-data-mlops.data_poc.v_conversations.
    # Supports workflow filtering (--workflow_names) and date range.
    "CONVERSATIONS": {
        "sql_query_path": "sql/context/conversation_context.sql",
        "unit": "message",
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
            f"Unknown context: {context_name}. Available contexts: {available}"
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
        "unit": config.get("unit", "message"),
        "conversation_timeout": config.get("conversation_timeout", "30m"),
    }
