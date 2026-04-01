"""
Tests for classifier utilities (utils/classifier_utils.py).

Covers:
- get_classifier_files: naming conventions, prefix lowercasing
- get_classifier_step_names: step name formatting
- get_classifier_query_path: generic vs. custom path routing
"""

import pytest

from categorization.utils.classifier_utils import (
    get_classifier_files,
    get_classifier_step_names,
    get_classifier_query_path,
)


# ---------------------------------------------------------------------------
# get_classifier_files
# ---------------------------------------------------------------------------


class TestGetClassifierFiles:
    def test_context_output_uses_lowercase_prefix(self):
        files = get_classifier_files("SENTIMENT")
        assert files["context_output"] == "sentiment_context.parquet"

    def test_preprocessed_output_uses_lowercase_prefix(self):
        files = get_classifier_files("ESCALATION")
        assert files["preprocessed_output"] == "escalation_preprocessed.parquet"

    def test_labeled_output_uses_lowercase_prefix(self):
        files = get_classifier_files("FEEDBACK")
        assert files["labeled_output"] == "feedback_labeled.parquet"

    def test_batch_requests_is_constant(self):
        files = get_classifier_files("SENTIMENT")
        assert files["batch_requests"] == "batch_requests.jsonl"

    def test_batch_results_is_constant(self):
        files = get_classifier_files("SENTIMENT")
        assert files["batch_results"] == "batch_results.jsonl"

    def test_metadata_is_constant(self):
        files = get_classifier_files("ANYTHING")
        assert files["metadata"] == "metadata.json"

    def test_batch_info_is_constant(self):
        files = get_classifier_files("ANYTHING")
        assert files["batch_info"] == "batch_info.json"

    def test_all_expected_keys_present(self):
        files = get_classifier_files("SENTIMENT")
        expected_keys = {
            "context_output",
            "preprocessed_output",
            "batch_requests",
            "batch_results",
            "labeled_output",
            "metadata",
            "batch_info",
        }
        assert expected_keys == set(files.keys())

    def test_lowercase_context_name_also_works(self):
        files = get_classifier_files("sentiment")
        assert files["context_output"] == "sentiment_context.parquet"

    def test_conversation_classifier_naming(self):
        files = get_classifier_files("CONVERSATION_SENTIMENT")
        assert files["context_output"] == "conversation_sentiment_context.parquet"
        assert files["labeled_output"] == "conversation_sentiment_labeled.parquet"

    def test_parquet_extension_on_data_files(self):
        files = get_classifier_files("SENTIMENT")
        for key in ("context_output", "preprocessed_output", "labeled_output"):
            assert files[key].endswith(".parquet"), f"{key} should end with .parquet"

    def test_jsonl_extension_on_batch_files(self):
        files = get_classifier_files("SENTIMENT")
        for key in ("batch_requests", "batch_results"):
            assert files[key].endswith(".jsonl"), f"{key} should end with .jsonl"


# ---------------------------------------------------------------------------
# get_classifier_step_names
# ---------------------------------------------------------------------------


class TestGetClassifierStepNames:
    def test_context_step_name(self):
        steps = get_classifier_step_names("SENTIMENT")
        assert steps["context"] == "Sentiment Context"

    def test_preprocess_step_name(self):
        steps = get_classifier_step_names("SENTIMENT")
        assert steps["preprocess"] == "Sentiment Preprocessing"

    def test_labeling_step_name(self):
        steps = get_classifier_step_names("SENTIMENT")
        assert steps["labeling"] == "Sentiment Labeling"

    def test_process_batches_step_name(self):
        steps = get_classifier_step_names("SENTIMENT")
        assert steps["process_batches"] == "Sentiment Process Batches"

    def test_all_expected_keys_present(self):
        steps = get_classifier_step_names("ESCALATION")
        assert set(steps.keys()) == {"context", "preprocess", "labeling", "process_batches"}

    def test_escalation_step_names(self):
        steps = get_classifier_step_names("ESCALATION")
        assert steps["context"] == "Escalation Context"
        assert steps["labeling"] == "Escalation Labeling"

    def test_multiword_context_name_title_case(self):
        # "CONVERSATION_SENTIMENT" -> title() -> "Conversation_Sentiment"
        steps = get_classifier_step_names("CONVERSATION_SENTIMENT")
        # Python's str.title() converts "CONVERSATION_SENTIMENT" to "Conversation_Sentiment"
        assert "Conversation_Sentiment" in steps["context"]


# ---------------------------------------------------------------------------
# get_classifier_query_path
# ---------------------------------------------------------------------------


class TestGetClassifierQueryPath:
    def test_generic_message_query_path(self):
        path = get_classifier_query_path("SENTIMENT", use_generic=True, unit="message")
        assert path.name == "generic_text_classification.sql"

    def test_generic_conversation_query_path(self):
        path = get_classifier_query_path(
            "CONVERSATION_SENTIMENT", use_generic=True, unit="conversation"
        )
        assert path.name == "generic_conversation_classification.sql"

    def test_returns_path_object(self):
        from pathlib import Path
        path = get_classifier_query_path("SENTIMENT")
        assert isinstance(path, Path)

    def test_custom_sql_path_override(self):
        from pathlib import Path
        path = get_classifier_query_path(
            "CUSTOM",
            use_generic=False,
            sql_query_path="sql/context/custom.sql",
        )
        assert path.name == "custom.sql"

    def test_auto_derived_path_when_no_generic_no_override(self):
        path = get_classifier_query_path("MYCONTEXT", use_generic=False)
        assert path.name == "mycontext_context.sql"
