"""
Step 3: Label the preprocessed dataset

Applies a classifier (LLM prompt) to the preprocessed dataset.
Two modes:
  - stream (default): concurrent Chat Completions API calls, results immediately
  - batch: submits to OpenAI Batch API (50% cheaper, 2-24h wait)

Multiple classifiers can be stacked on the same dataset. Each run adds a new
label column; combined_labeled.csv is rebuilt automatically after each run.
"""

from pathlib import Path

import pandas as pd

from categorization.clients.openai_batch import OpenAIBatchClient
from categorization.pipelines.classifiers import get_classifier_config
from categorization.settings.log import logger
from categorization.settings.pricing import estimate_cost
from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_step_names,
)
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_parquet_with_validation, save_parquet


def _rebuild_combined_csv(experiment_dir: Path, preprocessed_file: Path) -> str:
    """
    Scan the experiment directory for all *_labeled.parquet files, merge every
    label column onto the preprocessed base, and save combined_labeled.csv.

    Called automatically after each make_label run so the combined file always
    reflects every classifier applied to this experiment so far.
    """
    base = pd.read_parquet(preprocessed_file)
    base_cols = set(base.columns)

    id_col = "conversation_id" if "conversation_id" in base.columns else "message_id"

    for labeled_path in sorted(experiment_dir.glob("*_labeled.parquet")):
        labeled = pd.read_parquet(labeled_path)
        new_cols = [c for c in labeled.columns if c not in base_cols]
        if new_cols:
            base = base.merge(
                labeled[[id_col] + new_cols],
                on=id_col,
                how="left",
            )
            base_cols.update(new_cols)

    out = experiment_dir / "combined_labeled.csv"
    base.to_csv(out, index=False)
    return str(out)


def make_label(
    context_name: str = "CONVERSATIONS",
    classifier: str | None = None,
    workflow_names: str | None = None,
    mode: str = "stream",
    max_workers: int = 20,
    dry_run: bool = False,
) -> tuple[str, str]:
    """
    Label the preprocessed dataset using the specified classifier and mode.

    Args:
        context_name: Extraction context used to find the experiment (e.g. "CONVERSATIONS")
        classifier: Classifier to apply (e.g. "CONVERSATION_SENTIMENT"). Defaults to context_name.
        workflow_names: Comma-separated workflow names to pinpoint the right experiment.
        mode: "stream" (immediate, ~2x cost) or "batch" (2-24h wait, 50% cheaper)
        max_workers: Concurrent API threads for stream mode (default: 20)
        dry_run: Log cost estimate and exit without executing

    Returns:
        Tuple of (experiment_id, combined_csv_path)

    Raises:
        ValueError: If mode is invalid, classifier unknown, or preprocess step not completed
    """
    if mode not in ("stream", "batch"):
        raise ValueError(f"Invalid mode '{mode}'. Choose 'stream' or 'batch'.")

    effective_classifier = classifier or context_name

    logger.phase_start(
        "make_label",
        details=f"Classifier: {effective_classifier}, Mode: {mode}",
    )

    metadata = ExperimentMetadata.load(
        context_name=context_name, workflow_names=workflow_names
    )
    experiment_id = metadata.experiment_id

    logger.info(f"Experiment: {experiment_id}")

    if not metadata.is_step_completed("preprocess"):
        raise ValueError(
            "Preprocess step not completed. Run 'categorization make_preprocess' first."
        )

    classifier_config = get_classifier_config(effective_classifier)
    unit = classifier_config.get("unit", "message")
    classifier_files = get_classifier_files(effective_classifier)
    step_names = get_classifier_step_names(effective_classifier)

    text_col = "conversation_text" if unit == "conversation" else "message_text"
    required_cols = (
        [text_col] if unit == "conversation" else ["message_id", "message_text"]
    )
    item_label = "conversations" if unit == "conversation" else "messages"

    # Load preprocessed data from the extraction context
    source_files = get_classifier_files(context_name)
    input_file = metadata.experiment_dir / source_files["preprocessed_output"]
    df = load_parquet_with_validation(
        file_path=input_file,
        step_name="make_label",
        required_columns=required_cols,
    )

    messages = df[text_col].tolist()
    n = len(messages)
    model = classifier_config.get("model", "gpt-4o-mini")

    # Cost estimation (always logged; dry_run exits after)
    cost = estimate_cost(n, model, batch=(mode == "batch"))
    mode_label = "batch (50% off)" if mode == "batch" else "stream"
    if cost is not None:
        logger.info(
            f"Estimated cost: ~${cost:.4f}",
            details=f"{n:,} {item_label} × {model} @ {mode_label}",
        )
    else:
        logger.warning(
            f"Unknown pricing for model '{model}' — cost estimate unavailable"
        )

    if dry_run:
        logger.info("Dry run — exiting without executing")
        return experiment_id, ""

    logger.info(
        f"Processing: {step_names['labeling']} ({mode} mode, unit={unit})",
        details=f"{n:,} {item_label}",
    )

    client = OpenAIBatchClient()
    output_column = classifier_config["output_column"]

    if mode == "stream":
        labels, stats = client.label_realtime(
            messages=messages,
            system_prompt=classifier_config["system_prompt"],
            user_prompt_template=classifier_config["user_prompt_template"],
            model=model,
            temperature=classifier_config.get("temperature", 0),
            max_tokens=classifier_config.get("max_tokens", 10),
            max_workers=max_workers,
        )

        df[output_column] = labels
        labeled_count = sum(1 for lbl in labels if lbl != "error")

        # Save labeled parquet
        output_file = metadata.experiment_dir / classifier_files["labeled_output"]
        save_parquet(df, output_file, description=f"labeled {item_label}")

        # Save individual CSV
        df.to_csv(output_file.with_suffix(".csv"), index=False)

        metadata.update_step(
            step_name="labeling",
            stats={
                f"{item_label}_labeled": labeled_count,
                "classifier": effective_classifier,
            },
        )
        metadata.update_file("labeled")

        # Log label distribution
        logger.info(f"{output_column} distribution:")
        df_clean = df[df[output_column] != "error"]
        for label_val, count in df_clean[output_column].value_counts().items():
            pct = count / len(df_clean) * 100
            logger.info(f"  {label_val}: {count:,} ({pct:.1f}%)")

        post_fn = classifier_config.get("post_process")
        if post_fn:
            post_fn(df_clean, output_column)

        # Rebuild combined CSV
        combined_csv = _rebuild_combined_csv(metadata.experiment_dir, input_file)
        logger.success("Combined CSV updated", details=combined_csv)

        logger.phase_complete("make_label", count=labeled_count)
        return experiment_id, combined_csv

    else:  # batch mode
        batch_file_path = metadata.experiment_dir / classifier_files["batch_requests"]
        client.create_batch_file(
            messages=messages,
            output_path=batch_file_path,
            system_prompt=classifier_config["system_prompt"],
            user_prompt_template=classifier_config["user_prompt_template"],
            model=model,
            temperature=classifier_config.get("temperature", 0),
            max_tokens=classifier_config.get("max_tokens", 10),
        )

        task_desc = classifier_config.get(
            "task_description", f"{effective_classifier.lower()}_classification"
        )
        batch_info = client.submit_batch(
            file_path=batch_file_path,
            task_description=task_desc,
        )

        metadata.save_batch_info(
            batch_id=batch_info["batch_id"],
            file_id=batch_info["file_id"],
            total_requests=n,
            extraction_context=context_name,
            classifier=effective_classifier,
        )
        metadata.update_step(step_name="labeling", stats={})

        logger.success(
            f"Batch submitted: {step_names['labeling']}",
            details=f"Batch ID: {batch_info['batch_id']}",
        )
        logger.info(
            "Batch processing will complete in 2-24 hours",
            details="Run 'categorization make_process_batches' to check status and download results",
        )

        logger.phase_complete("make_label", count=n)
        return experiment_id, ""
