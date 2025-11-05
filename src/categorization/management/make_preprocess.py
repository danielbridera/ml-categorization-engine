"""
Step 2: Clean and Deduplicate Messages

Removes null/empty messages, filters by length, and deduplicates within workflows
to ensure high-quality, diverse training data.
"""

from categorization.pipelines import PREPROCESS_BY_CONTEXT
from categorization.pipelines.sentiment.constants import SENTIMENT_CTX
from categorization.settings.log import logger
from categorization.utils.deduplication_utils import (
    deduplicate_messages,
    filter_by_length,
    remove_null_messages,
    validate_message_quality,
)
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation, save_parquet


def make_preprocess(
    experiment_id: str | None = None, context_name: str = SENTIMENT_CTX
) -> str:
    """
    Clean and deduplicate messages.

    Args:
        experiment_id: Experiment ID (None = auto-detect latest)
        context_name: Pipeline context name

    Returns:
        Experiment ID
    """
    logger.phase_start("make_preprocess", details="Cleaning and deduplicating messages")

    # Load experiment metadata
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id, context_name=context_name
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Processing experiment: {experiment_id}")

    # Get configuration from registry
    preprocess_configs = PREPROCESS_BY_CONTEXT[context_name]

    # Execute each preprocessing configuration
    for preprocess_config in preprocess_configs:
        logger.info(f"Processing: {preprocess_config['name']}")

        # Load input data
        input_file = metadata.experiment_dir / preprocess_config["input_file"]
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
        output_file = metadata.experiment_dir / preprocess_config["output_file"]
        save_parquet(df, output_file, description="preprocessed messages")

        logger.success(
            f"Preprocessing complete: {preprocess_config['name']}",
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
