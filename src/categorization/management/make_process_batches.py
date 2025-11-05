"""
Step 3b: Download and Process Batch Results

Checks batch status, downloads results when ready, and creates labeled dataset.
"""

from categorization.clients.openai_batch import OpenAIBatchClient
from categorization.pipelines import PROCESS_BATCHES_BY_CONTEXT
from categorization.pipelines.sentiment.constants import SENTIMENT_CTX
from categorization.settings.log import logger
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation, save_parquet


def make_process_batches(
    experiment_id: str | None = None,
    context_name: str = SENTIMENT_CTX,
) -> tuple[str, str]:
    """
    Download and process batch results from OpenAI.

    Args:
        experiment_id: Experiment ID (None = auto-detect latest)
        context_name: Pipeline context name

    Returns:
        Tuple of (experiment_id, status)
    """
    logger.phase_start("make_process_batches", details="Checking batch status")

    # Load experiment metadata
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id, context_name=context_name
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Processing experiment: {experiment_id}")

    # Check if labeling was completed
    if not metadata.is_step_completed("labeling"):
        raise ValueError(
            "Labeling step not completed. Run 'categorization make_labeling' first."
        )

    # Get batch info
    batch_info = metadata.get_batch_info()
    if not batch_info:
        raise ValueError(
            "No batch info found. Run 'categorization make_labeling' first."
        )

    batch_id = batch_info["batch_id"]

    # Get configuration from registry
    process_batches_configs = PROCESS_BATCHES_BY_CONTEXT[context_name]

    # Initialize OpenAI Batch client
    batch_client = OpenAIBatchClient()

    # Execute each process_batches configuration
    for process_batches_config in process_batches_configs:
        logger.info(f"Processing: {process_batches_config['name']}")

        # Check batch status
        logger.info(f"Checking status for batch: {batch_id}")
        status_info = batch_client.check_status(batch_id)

        status = status_info["status"]

        if status == "in_progress" or status == "validating":
            # Batch still processing
            completed = status_info["completed_requests"]
            total = status_info["total_requests"]
            progress_pct = (completed / total * 100) if total > 0 else 0

            logger.info(
                f"Batch still processing: {status}",
                details=f"Progress: {completed}/{total} ({progress_pct:.1f}%)",
            )

            # Update batch status
            metadata.update_batch_status(
                status=status,
                completed_requests=completed,
                failed_requests=status_info["failed_requests"],
            )

            return experiment_id, "in_progress"

        elif status == "completed":
            # Batch completed - download results
            logger.info("Batch completed! Downloading results...")

            # Download results
            results_file = (
                metadata.experiment_dir / process_batches_config["batch_results_file"]
            )
            stats = batch_client.download_results(
                batch_id=batch_id, output_path=results_file
            )

            logger.success(
                "Results downloaded",
                details=f"Cost: ${stats['total_cost']:.4f} "
                f"(Success: {stats['successful']}, Failed: {stats['failed']})",
            )

            # Parse labels from results
            output_column = process_batches_config["output_column"]
            logger.info(f"Parsing {output_column} labels")
            labels = batch_client.parse_labels(
                results_path=results_file, total_requests=batch_info["total_requests"]
            )

            # Load preprocessed data and add labels
            input_file = metadata.experiment_dir / process_batches_config["input_file"]
            df = load_parquet_with_validation(
                file_path=input_file,
                step_name="process_batches",
                required_columns=["message_id", "message_text"],
            )

            # Add label column (configurable column name)
            df[output_column] = labels

            # Save labeled data
            output_file = (
                metadata.experiment_dir / process_batches_config["output_file"]
            )
            save_parquet(df, output_file, description="labeled messages")

            # Update metadata with results
            labeled_count = len([label for label in labels if label != "error"])

            metadata.update_batch_status(
                status="completed",
                completed_requests=stats["successful"],
                failed_requests=stats["failed"],
                cost_info={
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_cost": stats["total_cost"],
                },
            )

            metadata.update_step(
                step_name="process_batches", stats={"messages_labeled": labeled_count}
            )
            metadata.update_file("labeled")

            logger.phase_complete("make_process_batches", count=labeled_count)

            logger.success(
                "Batch processing complete",
                details=f"Labeled: {labeled_count:,} messages, Cost: ${stats['total_cost']:.4f}",
            )

            # Show label distribution
            logger.info(f"{output_column.capitalize()} distribution:")
            df_clean = df[df[output_column] != "error"]

            if len(df_clean) > 0:
                label_counts = df_clean[output_column].value_counts()
                for label, count in label_counts.items():
                    pct = (count / len(df_clean) * 100) if len(df_clean) > 0 else 0
                    logger.info(f"  {label}: {count:,} ({pct:.1f}%)")
            else:
                logger.warning(
                    "All labels returned errors - no successful categorizations"
                )

            logger.success(
                f"Processed batch results: {process_batches_config['name']}",
                details=f"Labeled: {labeled_count:,} messages",
            )

            return experiment_id, "completed"

        elif status == "failed":
            # Batch failed
            logger.error(f"Batch failed with status: {status}")

            metadata.update_batch_status(status="failed")

            return experiment_id, "failed"

        else:
            # Unknown status
            logger.warning(f"Unknown batch status: {status}")

            return experiment_id, status
