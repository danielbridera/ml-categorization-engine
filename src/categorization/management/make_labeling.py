"""
Step 3a: Submit Batch to OpenAI for Labeling

Creates batch request file and submits to OpenAI Batch API for
cost-effective sentiment labeling (50% cheaper).
"""

from categorization.clients.openai_batch import OpenAIBatchClient
from categorization.pipelines import LABELING_BY_CONTEXT
from categorization.pipelines.sentiment.constants import SENTIMENT_CTX
from categorization.settings.log import logger
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation


def make_labeling(
    experiment_id: str | None = None,
    model: str | None = None,
    context_name: str = SENTIMENT_CTX,
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

    # Get configuration from registry
    labeling_configs = LABELING_BY_CONTEXT[context_name]

    # Initialize OpenAI Batch client
    batch_client = OpenAIBatchClient()

    # Execute each labeling configuration
    for labeling_config in labeling_configs:
        logger.info(f"Processing: {labeling_config['name']}")

        # Use provided model or config default
        model_to_use = model if model is not None else labeling_config["model"]

        # Load preprocessed data
        input_file = metadata.experiment_dir / labeling_config["input_file"]
        df = load_parquet_with_validation(
            file_path=input_file,
            step_name="labeling",
            required_columns=["message_text"],
        )

        messages = df["message_text"].tolist()
        total_messages = len(messages)

        logger.info(f"Preparing {total_messages:,} messages for batch labeling")

        # Create batch request file
        batch_file_path = (
            metadata.experiment_dir / labeling_config["batch_requests_file"]
        )
        batch_client.create_batch_file(
            messages=messages,
            output_path=batch_file_path,
            system_prompt=labeling_config["system_prompt"],
            user_prompt_template=labeling_config["user_prompt_template"],
            model=model_to_use,
            temperature=labeling_config["temperature"],
            max_tokens=labeling_config["max_tokens"],
        )

        # Submit batch
        logger.info("Submitting batch to OpenAI")
        batch_info = batch_client.submit_batch(
            file_path=batch_file_path,
            task_description=labeling_config["task_description"],
        )

        batch_id = batch_info["batch_id"]
        file_id = batch_info["file_id"]

        # Save batch info to metadata
        metadata.save_batch_info(
            batch_id=batch_id, file_id=file_id, total_requests=total_messages
        )

        logger.success(
            f"Batch submitted: {labeling_config['name']}",
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
