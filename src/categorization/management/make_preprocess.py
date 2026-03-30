"""
Step 2: Clean and Deduplicate Messages (or Group into Conversations)

For message-unit classifiers: removes null/empty messages, filters by length,
and deduplicates within workflows.

For conversation-unit classifiers: groups messages into conversations using
a timeout-based inactivity window (same definition as CIE project).
"""

from categorization.settings.log import logger
from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_step_names,
)
from categorization.utils.conversation_utils import (
    format_conversations_from_sessions,
    group_messages_into_conversations,
)
from categorization.utils.deduplication_utils import (
    deduplicate_messages,
    filter_by_length,
    remove_null_messages,
    validate_message_quality,
)
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation, save_parquet


def make_preprocess(
    context_name: str = "CONVERSATIONS",
    workflow_names: str | None = None,
) -> str:
    """
    Clean and deduplicate messages, or group them into conversations.

    For message-unit classifiers: standard cleaning and deduplication pipeline.
    For conversation-unit classifiers: groups messages by user inactivity timeout
    and formats turn-by-turn conversation text for LLM classification.

    Args:
        context_name: Pipeline context name (e.g. "CONVERSATIONS")
        workflow_names: Comma-separated workflow names to pinpoint the right experiment

    Returns:
        Experiment ID
    """
    logger.phase_start("make_preprocess", details="Preprocessing data")

    # Load experiment metadata
    metadata = ExperimentMetadata.load(
        context_name=context_name, workflow_names=workflow_names
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Processing experiment: {experiment_id}")

    # Read unit from metadata (stored by make_context) — works for any context name
    params = metadata.get_parameters()
    unit = params.get("unit", "message")
    files = get_classifier_files(context_name)
    step_names = get_classifier_step_names(context_name)

    logger.info(f"Processing: {step_names['preprocess']} (unit={unit})")

    if unit == "conversation":
        return _make_preprocess_conversations(
            metadata, files, step_names, experiment_id
        )
    else:
        return _make_preprocess_messages(metadata, files, step_names, experiment_id)


def _make_preprocess_messages(
    metadata: ExperimentMetadata,
    files: dict,
    step_names: dict,
    experiment_id: str,
) -> str:
    """Standard message-level preprocessing: clean, filter, deduplicate."""
    # Load input data
    input_file = metadata.experiment_dir / files["context_output"]
    df = load_parquet_with_validation(
        file_path=input_file,
        step_name="preprocess",
        required_columns=["message_id", "message_text", "workflow_id"],
    )

    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} messages")

    # Step 1: Remove null/empty messages
    logger.info("Step 1: Removing null/empty messages")
    df = remove_null_messages(df, text_col="message_text")

    # Step 2: Filter by length
    params = metadata.get_parameters()
    min_length = params["min_length"]

    logger.info(f"Step 2: Filtering by length (min: {min_length})")
    df = filter_by_length(df, text_col="message_text", min_length=min_length)

    # Step 3: Deduplicate within workflows
    logger.info("Step 3: Deduplicating within workflows")
    df = deduplicate_messages(
        df,
        text_col="message_text",
        group_by_col="workflow_id",
        case_sensitive=False,
    )

    final_count = len(df)

    # Validate quality
    validate_message_quality(df, text_col="message_text")

    # Validate data is not empty after preprocessing
    if df.empty:
        raise ValueError(
            f"Preprocessing resulted in empty dataset. "
            f"Initial count: {initial_count}, Final count: 0. "
            f"Check your filters (min_length, deduplication rules)."
        )

    # Save preprocessed data
    output_file = metadata.experiment_dir / files["preprocessed_output"]
    save_parquet(df, output_file, description="preprocessed messages")

    logger.success(
        f"Preprocessing complete: {step_names['preprocess']}",
        details=f"{initial_count:,} → {final_count:,} messages "
        f"({final_count / initial_count * 100:.1f}% retained)",
    )

    # Update metadata
    metadata.update_step(
        step_name="preprocess", stats={"messages_after_preprocess": final_count}
    )
    metadata.update_file("preprocess")

    logger.phase_complete("make_preprocess", count=final_count)

    return experiment_id


def _make_preprocess_conversations(
    metadata: ExperimentMetadata,
    files: dict,
    step_names: dict,
    experiment_id: str,
) -> str:
    """Conversation-level preprocessing: group messages into conversations."""
    # Load input data (all messages — user + assistant)
    input_file = metadata.experiment_dir / files["context_output"]
    df = load_parquet_with_validation(
        file_path=input_file,
        step_name="preprocess",
        required_columns=[
            "message_id",
            "message_text",
            "workflow_id",
            "user_id",
            "event_timestamp",
            "sender_type",
        ],
    )

    initial_message_count = len(df)
    logger.info(
        f"Loaded {initial_message_count:,} raw messages for conversation grouping"
    )

    params = metadata.get_parameters()

    # If SQL already split conversations (conversation_id column present), just
    # format turns. Otherwise fall back to Python timeout-based grouping.
    if "conversation_id" in df.columns:
        logger.info("Conversations defined by SQL — formatting turns")
        conversations_df = format_conversations_from_sessions(df)
    else:
        timeout = params.get("conversation_timeout", "30m")
        conversations_df = group_messages_into_conversations(df, timeout_str=timeout)

    # Filter out conversations with very short conversation_text
    min_length = params.get("min_length", 6)
    conversations_df = conversations_df[
        conversations_df["conversation_text"].str.len() >= min_length
    ].copy()

    final_count = len(conversations_df)

    if conversations_df.empty:
        timeout = params.get("conversation_timeout", "30m")
        raise ValueError(
            f"Conversation preprocessing resulted in empty dataset. "
            f"Initial messages: {initial_message_count}. "
            f"Check date range, filters, and timeout (current: {timeout})."
        )

    # Save preprocessed conversations
    output_file = metadata.experiment_dir / files["preprocessed_output"]
    save_parquet(
        conversations_df, output_file, description="preprocessed conversations"
    )

    logger.success(
        f"Preprocessing complete: {step_names['preprocess']}",
        details=f"{initial_message_count:,} messages → {final_count:,} conversations",
    )

    # Update metadata
    metadata.update_step(
        step_name="preprocess",
        stats={
            "messages_after_preprocess": final_count,
            "raw_messages_processed": initial_message_count,
        },
    )
    metadata.update_file("preprocess")

    logger.phase_complete("make_preprocess", count=final_count)

    return experiment_id
