"""
Integration test: full stream pipeline with mocked BigQuery and OpenAI.

Flow tested:
  make_context -> make_preprocess -> make_label(mode="stream")

Strategy:
- monkeypatch.chdir(tmp_path) so ExperimentMetadata uses "experiments/" under
  the temp directory (avoiding pollution of the real working directory).
- BigQueryClient.execute_to_file is mocked to write a real parquet file.
- OpenAIBatchClient.label_realtime is mocked to return fake labels.

Assertions:
- combined_labeled.csv exists after make_label
- combined_labeled.csv contains the expected output column and data
- The required pipeline steps are marked as completed in metadata
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_working_dir(monkeypatch, tmp_path):
    """
    Change the working directory to tmp_path for every test in this module.
    This ensures ExperimentMetadata's default base_dir="experiments" resolves
    under the temp directory, not the real project root.
    """
    monkeypatch.chdir(tmp_path)
    # Also create the experiments dir so it exists for make_context
    (tmp_path / "experiments").mkdir()


@pytest.fixture()
def fake_messages_df() -> pd.DataFrame:
    """A minimal DataFrame that satisfies make_context + make_preprocess requirements."""
    return pd.DataFrame(
        {
            "message_id": [f"msg_{i}" for i in range(10)],
            "message_text": [
                "This product is great, I love it!",
                "Terrible service, very disappointed.",
                "Can you help me with my order please?",
                "The shipment arrived today, thank you.",
                "I need help urgently, my account is locked.",
                "Just checking the status of my ticket.",
                "Great experience overall, will recommend.",
                "Something is broken and I am very frustrated.",
                "How do I reset my password?",
                "Works exactly as described, happy customer.",
            ],
            "workflow_id": [f"wf_{i % 3}" for i in range(10)],
            "workflow_name": [f"workflow_{i % 3}" for i in range(10)],
            "user_id": [f"user_{i}" for i in range(10)],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_side_effect(parquet_df: pd.DataFrame):
    """
    Return a side-effect function for BigQueryClient.execute_to_file that writes
    parquet_df to whatever output_path is passed.
    """
    def _side_effect(query: str, output_path: str) -> None:
        parquet_df.to_parquet(output_path, index=False, engine="pyarrow")

    return _side_effect


def _fake_label_realtime_side_effect():
    """
    Return a side-effect for OpenAIBatchClient.label_realtime that produces
    'positive' labels and a minimal stats dict.
    """
    def _side_effect(messages, system_prompt, user_prompt_template, **kwargs):
        labels = ["positive"] * len(messages)
        stats = {
            "successful": len(messages),
            "failed": 0,
            "input_tokens": 100,
            "output_tokens": 10,
            "total_cost": 0.001,
        }
        return labels, stats

    return _side_effect


def run_full_pipeline(fake_messages_df: pd.DataFrame, context_name: str = "SENTIMENT"):
    """
    Execute make_context -> make_preprocess -> make_label(mode="stream").

    BigQuery and OpenAI are mocked. Returns (metadata, experiment_id, combined_csv).
    """
    from categorization.management.make_context import make_context
    from categorization.management.make_label import make_label
    from categorization.management.make_preprocess import make_preprocess
    from categorization.utils.metadata import ExperimentMetadata

    bq_side_effect = _write_parquet_side_effect(fake_messages_df)
    label_side_effect = _fake_label_realtime_side_effect()

    with (
        patch("categorization.management.make_context.BigQueryClient") as MockBQ,
        patch("categorization.management.make_label.OpenAIBatchClient") as MockOpenAI,
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key"}),
    ):
        bq_instance = MagicMock()
        bq_instance.execute_to_file.side_effect = bq_side_effect
        MockBQ.return_value = bq_instance

        openai_instance = MagicMock()
        openai_instance.label_realtime.side_effect = label_side_effect
        MockOpenAI.return_value = openai_instance

        # Step 1: make_context (creates experiment dir + context parquet)
        experiment_id = make_context(
            start_date="2024-01-01",
            end_date="2024-01-31",
            n_samples=10,
            min_length=6,
            context_name=context_name,
        )

        # Step 2: make_preprocess (dedup + filter)
        make_preprocess(context_name=context_name)

        # Step 3: make_label (stream mode — calls label_realtime)
        returned_id, combined_csv = make_label(
            context_name=context_name,
            classifier=context_name,
            mode="stream",
        )

    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id,
    )
    return metadata, returned_id, combined_csv


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestStreamPipeline:
    """Full pipeline smoke test with mocked external I/O."""

    def test_experiment_id_returned(self, fake_messages_df):
        metadata, returned_id, _ = run_full_pipeline(fake_messages_df)
        assert returned_id == metadata.experiment_id
        assert returned_id.startswith("sentiment_")

    def test_experiment_dir_created(self, fake_messages_df):
        metadata, _, _ = run_full_pipeline(fake_messages_df)
        assert metadata.experiment_dir.exists()
        assert metadata.experiment_dir.is_dir()

    def test_metadata_json_exists(self, fake_messages_df):
        metadata, _, _ = run_full_pipeline(fake_messages_df)
        assert (metadata.experiment_dir / "metadata.json").exists()

    def test_combined_csv_path_returned(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        assert combined_csv
        assert combined_csv.endswith(".csv")

    def test_combined_csv_file_exists(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        assert Path(combined_csv).exists()

    def test_combined_csv_has_sentiment_column(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        assert "sentiment" in df.columns

    def test_combined_csv_has_message_id_column(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        assert "message_id" in df.columns

    def test_combined_csv_has_message_text_column(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        assert "message_text" in df.columns

    def test_combined_csv_row_count_matches_preprocessed(self, fake_messages_df):
        metadata, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        preprocessed_path = metadata.experiment_dir / "sentiment_preprocessed.parquet"
        preprocessed_df = pd.read_parquet(preprocessed_path)
        assert len(df) == len(preprocessed_df)

    def test_labels_are_populated(self, fake_messages_df):
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        assert df["sentiment"].notna().all()

    def test_labels_have_expected_value(self, fake_messages_df):
        """Our mock always returns 'positive'."""
        _, _, combined_csv = run_full_pipeline(fake_messages_df)
        df = pd.read_csv(combined_csv)
        assert (df["sentiment"] == "positive").all()

    def test_context_step_completed(self, fake_messages_df):
        metadata, _, _ = run_full_pipeline(fake_messages_df)
        assert metadata.is_step_completed("context")

    def test_preprocess_step_completed(self, fake_messages_df):
        metadata, _, _ = run_full_pipeline(fake_messages_df)
        assert metadata.is_step_completed("preprocess")

    def test_label_step_completed(self, fake_messages_df):
        metadata, _, _ = run_full_pipeline(fake_messages_df)
        assert metadata.is_step_completed("label")

    def test_bigquery_called_once(self, fake_messages_df):
        """Verify BigQuery.execute_to_file was called exactly once (in make_context)."""
        from categorization.management.make_context import make_context
        from categorization.management.make_preprocess import make_preprocess

        bq_side_effect = _write_parquet_side_effect(fake_messages_df)

        with (
            patch("categorization.management.make_context.BigQueryClient") as MockBQ,
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key"}),
        ):
            bq_instance = MagicMock()
            bq_instance.execute_to_file.side_effect = bq_side_effect
            MockBQ.return_value = bq_instance

            make_context(
                start_date="2024-01-01",
                end_date="2024-01-31",
                n_samples=10,
                min_length=6,
                context_name="SENTIMENT",
            )
            make_preprocess(context_name="SENTIMENT")

        bq_instance.execute_to_file.assert_called_once()

    def test_openai_called_once_per_label_run(self, fake_messages_df):
        """Verify label_realtime is called exactly once during make_label(stream)."""
        from categorization.management.make_context import make_context
        from categorization.management.make_label import make_label
        from categorization.management.make_preprocess import make_preprocess

        bq_side_effect = _write_parquet_side_effect(fake_messages_df)
        label_side_effect = _fake_label_realtime_side_effect()

        with (
            patch("categorization.management.make_context.BigQueryClient") as MockBQ,
            patch("categorization.management.make_label.OpenAIBatchClient") as MockOpenAI,
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key"}),
        ):
            bq_instance = MagicMock()
            bq_instance.execute_to_file.side_effect = bq_side_effect
            MockBQ.return_value = bq_instance

            openai_instance = MagicMock()
            openai_instance.label_realtime.side_effect = label_side_effect
            MockOpenAI.return_value = openai_instance

            make_context(
                start_date="2024-01-01",
                end_date="2024-01-31",
                n_samples=10,
                min_length=6,
                context_name="SENTIMENT",
            )
            make_preprocess(context_name="SENTIMENT")
            make_label(
                context_name="SENTIMENT",
                classifier="SENTIMENT",
                mode="stream",
            )

        openai_instance.label_realtime.assert_called_once()
