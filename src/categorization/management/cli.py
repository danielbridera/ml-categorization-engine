"""
Categorization Pipeline CLI

Command-line interface for managing batch LLM categorization experiments.
"""

import functools
import sys

import click

from categorization.__version__ import __version__
from categorization.management.make_context import make_context
from categorization.management.make_labeling import make_labeling
from categorization.management.make_preprocess import make_preprocess
from categorization.management.make_process_batches import make_process_batches
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
    help="Number of messages to extract (default: 1000)",
)
@click.option(
    "--min_length",
    default=6,
    type=int,
    help="Minimum message length in characters (default: 6)",
)
@click.option(
    "--context_name",
    default="SENTIMENT",
    help="Context name for categorization (default: SENTIMENT)",
)
@handle_cli_errors
def make_context_cmd(
    start_date: str, end_date: str, n_samples: int, min_length: int, context_name: str
):
    """STEP 1: Extract messages from BigQuery"""
    click.echo("🔄 Extracting messages from BigQuery...")
    click.echo(f"   Context: {context_name}")
    click.echo(f"   Date range: {start_date} to {end_date}")
    click.echo(f"   Samples: {n_samples:,}, Min length: {min_length}")

    experiment_id = make_context(
        start_date=start_date,
        end_date=end_date,
        n_samples=n_samples,
        min_length=min_length,
        context_name=context_name,
    )

    click.secho("\n✅ Context creation completed!", fg="green")
    click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
    click.echo(
        f"\nNext step: categorization make_preprocess --context_name {context_name}"
    )


@main.command(name="make_preprocess")
@click.option(
    "--experiment_id",
    default=None,
    help="Experiment ID (auto-detects latest if not specified)",
)
@click.option(
    "--context_name",
    default=None,
    help="Context name for auto-detection (default: searches all contexts)",
)
@handle_cli_errors
def make_preprocess_cmd(experiment_id: str | None, context_name: str | None):
    """STEP 2: Clean and deduplicate messages"""
    if experiment_id:
        click.echo(f"🔄 Preprocessing experiment: {experiment_id}")
    else:
        context_msg = f" ({context_name})" if context_name else ""
        click.echo(f"🔄 Preprocessing latest experiment{context_msg}...")

    experiment_id = make_preprocess(
        experiment_id=experiment_id, context_name=context_name
    )

    click.secho("\n✅ Preprocessing completed!", fg="green")
    click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)

    context_param = f" --context_name {context_name}" if context_name else ""
    click.echo(f"\nNext step: categorization make_labeling{context_param}")


@main.command(name="make_labeling")
@click.option(
    "--experiment_id",
    default=None,
    help="Experiment ID (auto-detects latest if not specified)",
)
@click.option(
    "--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)"
)
@click.option(
    "--context_name",
    default="SENTIMENT",
    help="Context name for categorization (default: SENTIMENT)",
)
@handle_cli_errors
def make_labeling_cmd(experiment_id: str | None, model: str, context_name: str):
    """STEP 3a: Submit batch to OpenAI for labeling"""
    if experiment_id:
        click.echo(f"🔄 Submitting batch for experiment: {experiment_id}")
    else:
        click.echo(f"🔄 Submitting batch for latest experiment ({context_name})...")

    experiment_id, batch_id = make_labeling(
        experiment_id=experiment_id, model=model, context_name=context_name
    )

    click.secho("\n✅ Batch submitted successfully!", fg="green")
    click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
    click.secho(f"🆔 Batch ID: {batch_id}", fg="cyan")
    click.echo("\n⏳ Batch processing will complete in 2-24 hours")
    click.echo(
        f"💡 Check status: categorization make_process_batches --context_name {context_name}"
    )


@main.command(name="make_process_batches")
@click.option(
    "--experiment_id",
    default=None,
    help="Experiment ID (auto-detects latest if not specified)",
)
@click.option(
    "--context_name",
    default="SENTIMENT",
    help="Context name for categorization (default: SENTIMENT)",
)
@handle_cli_errors
def make_process_batches_cmd(experiment_id: str | None, context_name: str):
    """STEP 3b: Download and process batch results"""
    if experiment_id:
        click.echo(f"🔄 Processing batch for experiment: {experiment_id}")
    else:
        click.echo(f"🔄 Processing batch for latest experiment ({context_name})...")

    experiment_id, status = make_process_batches(
        experiment_id=experiment_id, context_name=context_name
    )

    context_prefix = context_name.lower()

    if status == "completed":
        click.secho("\n✅ Batch processing completed!", fg="green")
        click.secho(f"📁 Experiment ID: {experiment_id}", fg="cyan", bold=True)
        click.echo(
            f"\nLabeled data ready! Check: experiments/{experiment_id}/{context_prefix}_labeled.parquet"
        )
    elif status == "in_progress":
        click.secho("\n⏳ Batch still processing...", fg="yellow")
        click.echo(
            f"💡 Try again later: categorization make_process_batches --context_name {context_name}"
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
    help="Context name for auto-detection (default: searches all contexts)",
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
        # Format status indicators
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
    click.echo(f"Sentiment Analysis Pipeline v{__version__}")


if __name__ == "__main__":
    main()
