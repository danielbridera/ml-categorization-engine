"""
Step 3a: Submit Batch to OpenAI for Labeling

Creates batch request file and submits to OpenAI Batch API for
cost-effective sentiment labeling (50% cheaper).
"""

from categorization.clients.openai_batch import OpenAIBatchClient
from categorization.pipelines.classifiers import get_classifier_config
from categorization.settings.log import logger
from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_step_names,
)
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation


def make_labeling(
    experiment_id: str | None = None,
    model: str | None = None,
    context_name: str = "SENTIMENT",
) -> tuple[str, str]:
    """
    Submit batch to OpenAI for sentiment labeling.

    Args:
        experiment_id: Experiment ID (None = auto-detect latest)
        model: OpenAI model to use (None = use default from config)
        context_name: Pipeline context name

    Returns:
        Tuple of (experiment_id, batch_id)
    """
    logger.phase_start("make_labeling", details="Submitting batch to OpenAI")

    # Load experiment metadata
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id, context_name=context_name
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Processing experiment: {experiment_id}")

    # Check if preprocessing was completed
    if not metadata.is_step_completed("preprocess"):
        raise ValueError(
            "Preprocessing step not completed. Run 'categorization make_preprocess' first."
        )

    # Get configuration from unified registry
    classifier_config = get_classifier_config(context_name)
    unit = classifier_config.get("unit", "message")
    files = get_classifier_files(context_name)
    step_names = get_classifier_step_names(context_name)

    logger.info(f"Processing: {step_names['labeling']} (unit={unit})")

    # Use provided model or config default
    model_to_use = (
        model if model is not None else classifier_config.get("model", "gpt-4o-mini")
    )

    # Branch on unit: conversations use conversation_text, messages use message_text
    if unit == "conversation":
        text_column = "conversation_text"
        required_cols = ["conversation_text"]
        item_label = "conversations"
    else:
        text_column = "message_text"
        required_cols = ["message_text"]
        item_label = "messages"

    # Load preprocessed data
    input_file = metadata.experiment_dir / files["preprocessed_output"]
    df = load_parquet_with_validation(
        file_path=input_file,
        step_name="labeling",
        required_columns=required_cols,
    )

    messages = df[text_column].tolist()
    total_messages = len(messages)

    logger.info(f"Preparing {total_messages:,} {item_label} for batch labeling")

    # Initialize OpenAI Batch client
    batch_client = OpenAIBatchClient()

    # Create batch request file
    batch_file_path = metadata.experiment_dir / files["batch_requests"]
    batch_client.create_batch_file(
        messages=messages,
        output_path=batch_file_path,
        system_prompt=classifier_config["system_prompt"],
        user_prompt_template=classifier_config["user_prompt_template"],
        model=model_to_use,
        temperature=classifier_config.get("temperature", 0),
        max_tokens=classifier_config.get("max_tokens", 10),
    )

    # Submit batch
    logger.info("Submitting batch to OpenAI")
    task_desc = classifier_config.get(
        "task_description", f"{context_name.lower()}_classification"
    )
    batch_info = batch_client.submit_batch(
        file_path=batch_file_path,
        task_description=task_desc,
    )

    batch_id = batch_info["batch_id"]
    file_id = batch_info["file_id"]

    # Save batch info to metadata
    metadata.save_batch_info(
        batch_id=batch_id, file_id=file_id, total_requests=total_messages
    )

    logger.success(
        f"Batch submitted: {step_names['labeling']}",
        details=f"Batch ID: {batch_id}",
    )

    # Update step completion
    metadata.update_step(step_name="labeling", stats={})

    logger.phase_complete("make_labeling", count=total_messages)

    logger.info(
        "⏳ Batch processing will complete in 2-24 hours",
        details="Run 'categorization make_process_batches' to check status and download results",
    )

    return experiment_id, batch_id
