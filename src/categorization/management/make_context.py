"""
Step 1: Extract Messages from BigQuery

Queries BigQuery for user messages within specified date range and creates
a new experiment with the extracted data.
"""

import pandas as pd

from categorization.clients.bigquery import BigQueryClient
from categorization.pipelines.classifiers import get_classifier_config
from categorization.settings.log import logger
from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_query_path,
    get_classifier_step_names,
)
from categorization.utils.conversation_utils import parse_timeout
from categorization.utils.metadata import ExperimentMetadata
from categorization.utils.utils import load_query, validate_date_format


def make_context(
    start_date: str,
    end_date: str,
    n_samples: int = 1000,
    min_length: int = 6,
    context_name: str = "SENTIMENT",
    workflow_name: str | None = None,
) -> str:
    """
    Extract messages from BigQuery and create new experiment.

    Args:
        start_date: Start date for extraction (YYYY-MM-DD)
        end_date: End date for extraction (YYYY-MM-DD)
        n_samples: Number of messages to extract
        min_length: Minimum message length
        context_name: Pipeline context name
        workflow_name: Optional workflow name to filter (e.g. "my_bot"). Speeds up
                       conversation-mode queries significantly.

    Returns:
        Experiment ID
    """
    logger.phase_start("make_context", details="Extracting messages from BigQuery")

    # Validate dates
    validate_date_format(start_date)
    validate_date_format(end_date)

    if start_date >= end_date:
        raise ValueError(
            f"start_date must be before end_date: {start_date} >= {end_date}"
        )

    # Validate n_samples
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_samples > 1_000_000:
        logger.warning(f"Large n_samples may cause memory issues: {n_samples:,}")

    # Get configuration from unified registry first (needed for metadata)
    classifier_config = get_classifier_config(context_name)
    unit = classifier_config.get("unit", "message")
    conversation_timeout = classifier_config.get("conversation_timeout", "30m")

    # Create new experiment
    metadata = ExperimentMetadata.create_new(
        start_date=start_date,
        end_date=end_date,
        n_samples=n_samples,
        min_length=min_length,
        context_name=context_name,
        unit=unit,
        conversation_timeout=conversation_timeout,
        workflow_name=workflow_name,
    )

    experiment_id = metadata.experiment_id
    logger.info(f"Created experiment: {experiment_id}")

    files = get_classifier_files(context_name)
    step_names = get_classifier_step_names(context_name)

    # Get SQL query path (generic or custom), routing by unit type
    use_generic = classifier_config.get("use_generic_query", True)
    query_path = get_classifier_query_path(
        context_name, use_generic=use_generic, unit=unit
    )

    if workflow_name:
        logger.info(
            f"Processing: {step_names['context']} (unit={unit}, workflow={workflow_name})"
        )
    else:
        logger.info(f"Processing: {step_names['context']} (unit={unit})")

    # Load SQL query template
    query_template = load_query(query_path)

    # Format query with parameters
    # workflow_name_filter is only injected for conversation-mode queries
    # (generic_text_classification.sql does not have this placeholder)
    format_kwargs: dict = {
        "start_date": start_date,
        "end_date": end_date,
        "n_samples": n_samples,
        "min_length": min_length,
    }
    if unit == "conversation":
        # No table alias in CTE — plain column name
        format_kwargs["workflow_name_filter"] = (
            f"AND workflow_name = '{workflow_name}'" if workflow_name else ""
        )
        # Convert timeout string (e.g. "30m") to minutes for BigQuery TIMESTAMP_DIFF
        timeout_td = parse_timeout(conversation_timeout)
        format_kwargs["conversation_timeout_minutes"] = int(
            timeout_td.total_seconds() // 60
        )

    query = query_template.format(**format_kwargs)

    # Initialize BigQuery client
    bq_client = BigQueryClient()

    # Execute query and save to file
    output_path = metadata.experiment_dir / files["context_output"]
    bq_client.execute_to_file(query=query, output_path=str(output_path))

    # Validate file was created
    if not output_path.exists():
        raise FileNotFoundError(
            f"BigQuery export failed: output file not created at {output_path}"
        )

    # Load to get actual count
    df = pd.read_parquet(output_path)

    # Validate data is not empty
    if df.empty:
        raise ValueError(
            f"BigQuery query returned no data. Check your date range and filters.\n"
            f"Context: {context_name}\n"
            f"Date range: {start_date} to {end_date}"
        )

    actual_count = len(df)
    item_label = "messages (user + assistant)" if unit == "conversation" else "messages"

    logger.success(
        f"Query executed: {step_names['context']}",
        details=f"Extracted {actual_count:,} {item_label}",
    )

    # Update metadata
    metadata.update_step(
        step_name="context", stats={"messages_extracted": actual_count}
    )
    metadata.update_file("context")

    logger.phase_complete("make_context", count=actual_count)

    return experiment_id
