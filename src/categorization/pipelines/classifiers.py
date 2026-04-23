"""
Classifier Registry - Unified Prompt-Driven Configuration

This module contains all text classification configurations in a single unified registry.
To add a new classifier, simply add a new entry to the CLASSIFIERS dictionary below.

Each classifier is defined by its prompts, output column, and optional model parameters.
The system handles file naming, SQL queries, and preprocessing automatically.
"""

import json
import re
from collections.abc import Callable
from typing import Any, TypedDict

import pandas as pd


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
        post_process: Optional callable for post-labeling logic.
            Signature: (df_clean: pd.DataFrame, output_column: str) -> None
            Called after labels are written and errors filtered out.
            Use for derived metrics, secondary outputs, or custom logging.
        parse_json_response: When True, treat each raw LLM response as a JSON
            object of shape ``{"<label_key>": "...", "reason": "..."}`` and
            split into two columns: ``<output_column>`` (the label) and
            ``<output_column>_reason`` (free text). Malformed responses fall
            back to the raw string + None.
        family: Classifier family tag ("monitoring", "conversations",
            "message"). Used for classifier/context compatibility checks and
            downstream writer key mapping.
        ls_eval_key: Optional LangSmith-whitelisted evaluator key for the
            monitoring family — consumed by the LangSmith-shaped writer.
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
    post_process: Callable[[pd.DataFrame, str], None]
    parse_json_response: bool
    family: str
    ls_eval_key: str


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================
# Optional post-labeling functions that can be attached to a classifier via
# the `post_process` field. Each function receives the labeled DataFrame
# (errors already filtered) and the output column name.


_JSON_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_label_and_reason(raw: str) -> tuple[str, str | None]:
    """Parse one JSON-shaped LLM response into ``(label, reason)``.

    Expects ``{"<label_field>": "...", "reason|reasoning": "..."}``. The first
    key that is not ``"reason"``/``"reasoning"`` is treated as the label
    field, which lets a single helper work across evaluators whose label key
    names differ (``intent_fulfillment``, ``apology_behavior``, etc.).

    Tolerates surrounding code fences (``\\`\\`\\`json ... \\`\\`\\```). On
    parse failure returns ``(raw, None)`` so downstream can coerce via a
    per-key fallback instead of silently dropping the row.
    """
    if not isinstance(raw, str) or raw == "error":
        return raw, None
    cleaned = _JSON_CODE_FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return raw, None
    if not isinstance(parsed, dict):
        return raw, None
    reason = parsed.get("reason") or parsed.get("reasoning")
    label_key = next(
        (k for k in parsed.keys() if k not in ("reason", "reasoning")), None
    )
    if label_key is None:
        return raw, reason
    return str(parsed[label_key]), (str(reason) if reason is not None else None)


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
# Auto-discovered specs from src/categorization/prompts/
# ============================================================================
# Newer classifiers live one-per-file under prompts/<family>/<name>.py and are
# registered via the ClassifierSpec dataclass. We merge them into CLASSIFIERS
# at import time so downstream (make_label, make_process_batches) keeps reading
# a single unified registry. On name collision, the auto-discovered spec wins —
# this lets a newer spec override a legacy entry during migration.
try:
    from categorization.prompts import discover_classifier_configs as _discover

    for _name, _cfg in _discover().items():
        CLASSIFIERS[_name] = _cfg  # type: ignore[assignment]
    del _discover, _name, _cfg
except ImportError:
    # prompts package not yet set up — silently continue with the legacy registry
    pass
except NameError:
    # No specs discovered — _name/_cfg never bound; that's fine
    pass


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
