"""
Classifier Utilities

Helper functions for working with prompt-driven classifiers.
These utilities auto-generate file names and step names based on context,
eliminating the need for hardcoded constants.
"""

from pathlib import Path


def get_classifier_files(context_name: str) -> dict[str, str]:
    """
    Generate standard file names for a classifier context.

    All data files are prefixed with the lowercase context name to support
    multi-context experiments in the same directory.

    Args:
        context_name: The context identifier (e.g., "SENTIMENT")

    Returns:
        Dictionary mapping file types to file names

    Example:
        >>> files = get_classifier_files("SENTIMENT")
        >>> files["context_output"]
        'sentiment_context.parquet'
        >>> files["labeled_output"]
        'sentiment_labeled.parquet'
    """
    prefix = context_name.lower()
    return {
        # Step 1: Context extraction
        "context_output": f"{prefix}_context.parquet",
        # Step 2: Preprocessing
        "preprocessed_output": f"{prefix}_preprocessed.parquet",
        # Step 3a: Batch labeling
        "batch_requests": "batch_requests.jsonl",
        # Step 3b: Process batches
        "batch_results": "batch_results.jsonl",
        "labeled_output": f"{prefix}_labeled.parquet",
        # Metadata files
        "metadata": "metadata.json",
        "batch_info": "batch_info.json",
    }


def get_classifier_step_names(context_name: str) -> dict[str, str]:
    """
    Generate standard step names for logging and metadata.

    Args:
        context_name: The context identifier (e.g., "SENTIMENT")

    Returns:
        Dictionary mapping step types to human-readable step names

    Example:
        >>> steps = get_classifier_step_names("SENTIMENT")
        >>> steps["context"]
        'Sentiment Context'
        >>> steps["labeling"]
        'Sentiment Labeling'
    """
    title = context_name.title()
    return {
        "context": f"{title} Context",
        "preprocess": f"{title} Preprocessing",
        "labeling": f"{title} Labeling",
        "process_batches": f"{title} Process Batches",
    }


def get_generic_query_path() -> Path:
    """
    Get the path to the generic text classification SQL query.

    Returns:
        Path to the generic SQL query file
    """
    from categorization.settings.queries import CONTEXT_QUERIES_PATH

    return CONTEXT_QUERIES_PATH / "generic_text_classification.sql"


def get_conversation_query_path() -> Path:
    """
    Get the path to the generic conversation classification SQL query.

    Returns:
        Path to the conversation SQL query file
    """
    from categorization.settings.queries import CONTEXT_QUERIES_PATH

    return CONTEXT_QUERIES_PATH / "generic_conversation_classification.sql"


def get_classifier_query_path(
    context_name: str,
    use_generic: bool = True,
    unit: str = "message",
    sql_query_path: str | None = None,
) -> Path:
    """
    Get the SQL query path for a classifier context.

    Args:
        context_name: The context identifier (e.g., "SENTIMENT")
        use_generic: If True, use the generic query. If False, use context-specific query.
        unit: "message" (default) or "conversation" — determines which generic query to use.

    Returns:
        Path to the SQL query file
    """
    if use_generic:
        if unit == "conversation":
            return get_conversation_query_path()
        return get_generic_query_path()

    # Explicit path override from classifier config
    if sql_query_path:
        from categorization.settings import BASE_PATH

        return BASE_PATH / sql_query_path

    # Auto-derive path from context name
    from categorization.settings.queries import CONTEXT_QUERIES_PATH

    context_dir = CONTEXT_QUERIES_PATH / context_name.lower()
    return context_dir / f"{context_name.lower()}_context.sql"
