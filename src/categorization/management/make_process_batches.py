"""
Step 4 (batch mode only): Download and Process Batch Results

Polls the OpenAI Batch API for completion, downloads results, parses labels,
and writes the labeled dataset + combined CSV.

Only needed when make_label was run with --mode batch.
"""

from categorization.clients.openai_batch import OpenAIBatchClient
from categorization.management.make_label import _rebuild_combined_csv
from categorization.pipelines.classifiers import get_classifier_config
from categorization.settings.log import logger
from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_step_names,
)
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation, save_parquet


def make_process_batches(
    context_name: str = "CONVERSATIONS",
    workflow_names: str | None = None,
) -> tuple[str, str]:
    """
    Download and process batch results from OpenAI.

    The extraction context and classifier are read from batch_info saved by
    make_label --mode batch, so no additional parameters are needed.

    Args:
        context_name: Extraction context used to find the experiment (e.g. "CONVERSATIONS")
        workflow_names: Comma-separated workflow names to pinpoint the right experiment.

    Returns:
        Tuple of (experiment_id, status) where status is one of:
        "completed", "in_progress", "failed", or the raw OpenAI status string
    """
    logger.phase_start("make_process_batches", details="Checking batch status")

    metadata = ExperimentMetadata.load(
        context_name=context_name, workflow_names=workflow_names
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Experiment: {experiment_id}")

    if not metadata.is_step_completed("labeling"):
        raise ValueError(
            "Labeling step not completed. Run 'categorization make_label --mode batch' first."
        )

    batch_info = metadata.get_batch_info()
    if not batch_info:
        raise ValueError(
            "No batch info found. Run 'categorization make_label --mode batch' first."
        )

    batch_id = batch_info["batch_id"]

    # Resolve extraction context and classifier from batch_info (stored by make_label)
    extraction_context = batch_info.get("extraction_context") or context_name
    effective_classifier = batch_info.get("classifier") or context_name

    classifier_config = get_classifier_config(effective_classifier)
    unit = classifier_config.get("unit", "message")
    classifier_files = get_classifier_files(effective_classifier)
    step_names = get_classifier_step_names(effective_classifier)

    item_label = "conversations" if unit == "conversation" else "messages"
    required_cols = (
        ["conversation_text"] if unit == "conversation" else ["message_id", "message_text"]
    )

    logger.info(
        f"Processing: {step_names['process_batches']}",
        details=f"Classifier: {effective_classifier}, unit={unit}",
    )

    batch_client = OpenAIBatchClient()

    logger.info(f"Checking status for batch: {batch_id}")
    status_info = batch_client.check_status(batch_id)
    status = status_info["status"]

    if status in ("in_progress", "validating"):
        completed = status_info["completed_requests"]
        total = status_info["total_requests"]
        progress_pct = (completed / total * 100) if total > 0 else 0

        logger.info(
            f"Batch still processing: {status}",
            details=f"Progress: {completed}/{total} ({progress_pct:.1f}%)",
        )
        metadata.update_batch_status(
            status=status,
            completed_requests=completed,
            failed_requests=status_info["failed_requests"],
        )
        return experiment_id, "in_progress"

    elif status == "completed":
        logger.info("Batch completed — downloading results")

        results_file = metadata.experiment_dir / classifier_files["batch_results"]
        stats = batch_client.download_results(batch_id=batch_id, output_path=results_file)

        logger.success(
            "Results downloaded",
            details=f"Cost: ${stats['total_cost']:.4f} "
            f"(Success: {stats['successful']}, Failed: {stats['failed']})",
        )

        output_column = classifier_config["output_column"]
        labels = batch_client.parse_labels(
            results_path=results_file, total_requests=batch_info["total_requests"]
        )

        # Load preprocessed data from extraction context
        source_files = get_classifier_files(extraction_context)
        input_file = metadata.experiment_dir / source_files["preprocessed_output"]
        df = load_parquet_with_validation(
            file_path=input_file,
            step_name="process_batches",
            required_columns=required_cols,
        )

        df[output_column] = labels
        labeled_count = sum(1 for lbl in labels if lbl != "error")

        # Save labeled parquet
        output_file = metadata.experiment_dir / classifier_files["labeled_output"]
        save_parquet(df, output_file, description=f"labeled {item_label}")

        # Save individual CSV
        df.to_csv(output_file.with_suffix(".csv"), index=False)
        logger.success("CSV export saved", details=str(output_file.with_suffix(".csv")))

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
            step_name="process_batches",
            stats={
                f"{item_label}_labeled": labeled_count,
                "classifier": effective_classifier,
            },
        )
        metadata.update_file("labeled")

        # Log label distribution
        logger.info(f"{output_column} distribution:")
        df_clean = df[df[output_column] != "error"]
        if len(df_clean) > 0:
            for label_val, count in df_clean[output_column].value_counts().items():
                pct = count / len(df_clean) * 100
                logger.info(f"  {label_val}: {count:,} ({pct:.1f}%)")
        else:
            logger.warning("All labels returned errors — no successful classifications")

        # Rebuild combined CSV
        combined_csv = _rebuild_combined_csv(metadata.experiment_dir, input_file)
        logger.success("Combined CSV updated", details=combined_csv)

        logger.phase_complete("make_process_batches", count=labeled_count)
        return experiment_id, "completed"

    elif status == "failed":
        logger.error(f"Batch failed: {status}")
        metadata.update_batch_status(status="failed")
        return experiment_id, "failed"

    else:
        logger.warning(f"Unknown batch status: {status}")
        return experiment_id, status
