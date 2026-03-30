"""
Categorization Pipeline CLI

Command-line interface for managing batch LLM categorization experiments.
"""
# mypy: disable-error-code="no-untyped-def"

import functools
import sys

import click

from categorization.__version__ import __version__
from categorization.management.make_context import make_context
from categorization.management.make_label import make_label
from categorization.management.make_preprocess import make_preprocess
from categorization.management.make_process_batches import make_process_batches
from categorization.pipelines.classifiers import list_classifiers
from categorization.settings.log import setup_pipeline_logging
from categorization.utils.metadata import ExperimentMetadata


def handle_cli_errors(func):
    """Decorator to handle errors in CLI commands"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            click.secho(f"❌ Error: {e}", fg="red", err=True)
            sys.exit(1)

    return wrapper


@click.group()
@click.version_option(version=__version__, prog_name="categorization")
def main():
    """
    Categorization Pipeline

    Generic framework for batch LLM categorization tasks (sentiment, escalation, etc.)
    """
    # Initialize logging for CLI
    setup_pipeline_logging(level="INFO", json_enabled=False)


@main.command(name="make_context")
@click.option(
    "--start_date", required=True, help="Start date for data extraction (YYYY-MM-DD)"
)
@click.option(
    "--end_date", required=True, help="End date for data extraction (YYYY-MM-DD)"
)
@click.option(
    "--n_samples",
    default=1000,
    type=int,
    help="Number of samples to extract per workflow (default: 1000)",
)
@click.option(
    "--min_length",
    default=6,
    type=int,
    help="Minimum message length in characters (default: 6)",
)
@click.option(
    "--context_name",
    default="CONVERSATIONS",
    help="Extraction context name (e.g. 'CONVERSATIONS')",
)
@click.option(
    "--workflow_names",
    default=None,
    help="Comma-separated workflow names to filter (e.g. 'bot-a,bot-b,bot-c').",
)
@click.option(
    "--sql_query_path",
    default=None,
    help="SQL query path for fully custom extraction contexts not in the registry.",
)
@click.option(
    "--unit",
    default="message",
    help="Classification unit: 'message' (default) or 'conversation'.",
)
@handle_cli_errors
def make_context_cmd(
    start_date: str,
    end_date: str,
    n_samples: int,
    min_length: int,
    context_name: str,
    workflow_names: str | None,
    sql_query_path: str | None,
    unit: str,
):
    """STEP 1: Extract messages from BigQuery"""
    click.echo("🔄 Extracting messages from BigQuery...")
    click.echo(f"   Context: {context_name}")
    click.echo(f"   Date range: {start_date} to {end_date}")
    click.echo(f"   Samples: {n_samples:,}, Min length: {min_length}")
    if workflow_names:
        click.echo(f"   Workflows: {workflow_names}")

    experiment_id = make_context(
        start_date=start_date,
        end_date=end_date,
        n_samples=n_samples,
        min_length=min_length,
        context_name=context_name,
        workflow_names=workflow_names,
        sql_query_path=sql_query_path,
        unit=unit,
    )

    click.secho("\n✅ Context creation completed!", fg="green")
    click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
    wf_param = f" --workflow_names '{workflow_names}'" if workflow_names else ""
    click.echo(
        f"\nNext step: categorization make_preprocess --context_name {context_name}{wf_param}"
    )


@main.command(name="make_preprocess")
@click.option(
    "--context_name",
    default="CONVERSATIONS",
    help="Extraction context name (e.g. 'CONVERSATIONS')",
)
@click.option(
    "--workflow_names",
    default=None,
    help="Comma-separated workflow names to pinpoint the right experiment.",
)
@handle_cli_errors
def make_preprocess_cmd(context_name: str, workflow_names: str | None):
    """STEP 2: Clean and deduplicate messages"""
    click.echo(f"🔄 Preprocessing latest experiment ({context_name})...")

    experiment_id = make_preprocess(
        context_name=context_name, workflow_names=workflow_names
    )

    click.secho("\n✅ Preprocessing completed!", fg="green")
    click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)

    wf_param = f" --workflow_names '{workflow_names}'" if workflow_names else ""
    click.echo(
        f"\nNext step: categorization make_label --context_name {context_name}{wf_param}"
    )


@main.command(name="make_label")
@click.option(
    "--context_name",
    default="CONVERSATIONS",
    help="Extraction context used to find the experiment (e.g. 'CONVERSATIONS')",
)
@click.option(
    "--classifier",
    default=None,
    help=f"Classifier to apply (available: {', '.join(list_classifiers())}). Defaults to --context_name.",
)
@click.option(
    "--workflow_names",
    default=None,
    help="Comma-separated workflow names to pinpoint the right experiment.",
)
@click.option(
    "--mode",
    default="stream",
    type=click.Choice(["stream", "batch"]),
    help="Labeling mode: 'stream' (immediate, ~2x cost) or 'batch' (2-24h wait, 50% cheaper). Default: stream.",
)
@click.option(
    "--max_workers",
    default=20,
    type=int,
    help="Concurrent API threads for stream mode (default: 20)",
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="Print cost estimate and exit without executing.",
)
@handle_cli_errors
def make_label_cmd(
    context_name: str,
    classifier: str | None,
    workflow_names: str | None,
    mode: str,
    max_workers: int,
    dry_run: bool,
):
    """STEP 3: Label dataset with a classifier (stream or batch mode)"""
    effective = classifier or context_name
    click.echo(
        f"🔄 Labeling latest experiment ({context_name}, classifier: {effective}, mode: {mode})..."
    )

    experiment_id, csv_path = make_label(
        context_name=context_name,
        classifier=classifier,
        workflow_names=workflow_names,
        mode=mode,
        max_workers=max_workers,
        dry_run=dry_run,
    )

    if dry_run:
        return

    if mode == "stream":
        click.secho("\n✅ Labeling completed!", fg="green")
        click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
        click.secho(f"📄 CSV: {csv_path}", fg="cyan")
    else:
        click.secho("\n✅ Batch submitted!", fg="green")
        click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
        click.echo("\n⏳ Batch processing will complete in 2-24 hours")
        wf_param = f" --workflow_names '{workflow_names}'" if workflow_names else ""
        click.echo(
            f"💡 Check status: categorization make_process_batches --context_name {context_name}{wf_param}"
        )


@main.command(name="make_process_batches")
@click.option(
    "--context_name",
    default="CONVERSATIONS",
    help="Extraction context used to find the experiment (e.g. 'CONVERSATIONS')",
)
@click.option(
    "--workflow_names",
    default=None,
    help="Comma-separated workflow names to pinpoint the right experiment.",
)
@handle_cli_errors
def make_process_batches_cmd(context_name: str, workflow_names: str | None):
    """STEP 4 (batch mode only): Download and process batch results"""
    click.echo(f"🔄 Processing batch for latest experiment ({context_name})...")

    experiment_id, status = make_process_batches(
        context_name=context_name, workflow_names=workflow_names
    )

    if status == "completed":
        click.secho("\n✅ Batch processing completed!", fg="green")
        click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
        click.echo(f"\nLabeled data ready in: experiments/{experiment_id}/")
    elif status == "in_progress":
        click.secho("\n⏳ Batch still processing...", fg="yellow")
        wf_param = f" --workflow_names '{workflow_names}'" if workflow_names else ""
        click.echo(
            f"💡 Try again later: categorization make_process_batches --context_name {context_name}{wf_param}"
        )
    else:
        click.secho(f"\n⚠️  Batch status: {status}", fg="yellow")


@main.command(name="status")
@click.option(
    "--experiment_id",
    default=None,
    help="Experiment ID (shows latest if not specified)",
)
@click.option(
    "--context_name",
    default=None,
    help="Context name for auto-detection",
)
@handle_cli_errors
def status_cmd(experiment_id: str | None, context_name: str | None):
    """Show experiment status and progress"""
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id, context_name=context_name
    )
    metadata.print_status()


@main.command(name="list")
@click.option(
    "--context_name",
    default=None,
    help="Filter by context name (default: shows all contexts)",
)
@handle_cli_errors
def list_cmd(context_name: str | None):
    """List all experiments"""
    experiments = ExperimentMetadata.list_experiments(context_name=context_name)

    if not experiments:
        context_msg = f" for context '{context_name}'" if context_name else ""
        click.echo(f"No experiments found{context_msg}.")
        return

    click.echo(f"\n{'=' * 80}")
    context_msg = f" - {context_name}" if context_name else ""
    click.echo(f"All Experiments{context_msg} ({len(experiments)} total)")
    click.echo(f"{'=' * 80}\n")

    for exp in experiments:
        steps_completed = len(exp["steps_completed"])
        total_steps = 4  # context, preprocess, labeling, process_batches

        status_str = " → ".join(
            [
                "✅ context" if "context" in exp["steps_completed"] else "⏸️ context",
                "✅ preprocess"
                if "preprocess" in exp["steps_completed"]
                else "⏸️ preprocess",
                "✅ labeling" if "labeling" in exp["steps_completed"] else "⏸️ labeling",
                "✅ process_batches"
                if "process_batches" in exp["steps_completed"]
                else "⏸️ process_batches",
            ]
        )

        click.echo(f"📁 {exp['experiment_id']}")
        click.echo(f"   Created: {exp['created_at']}")
        click.echo(f"   Progress: {steps_completed}/{total_steps} steps")
        click.echo(f"   Status: {status_str}")

        if exp["batch_status"]:
            click.echo(f"   Batch: {exp['batch_status']}")

        click.echo(
            f"   Parameters: {exp['parameters']['start_date']} to {exp['parameters']['end_date']}, "
            f"{exp['parameters']['n_samples']:,} samples"
        )
        click.echo()

    click.echo(f"{'=' * 80}\n")


@main.command(name="version")
def version_cmd():
    """Show version information"""
    classifiers = ", ".join(list_classifiers())
    click.echo(f"Categorization Pipeline v{__version__}")
    click.echo(f"Available classifiers: {classifiers}")


if __name__ == "__main__":
    main()
