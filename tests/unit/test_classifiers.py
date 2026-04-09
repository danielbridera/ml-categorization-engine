"""
Tests for the classifier registry (classifiers.py).

Covers:
- CLASSIFIERS registry structure and required fields
- get_classifier_config lookups and error handling
- list_classifiers ordering
- get_classifier_info defaults
- {message} placeholder presence in every user_prompt_template
- EXTRACTION_CONTEXTS structure
"""

import pandas as pd
import pytest

from categorization.pipelines.classifiers import (
    CLASSIFIERS,
    EXTRACTION_CONTEXTS,
    _compute_icsat_score,
    get_classifier_config,
    get_classifier_info,
    list_classifiers,
)


# ---------------------------------------------------------------------------
# Registry content
# ---------------------------------------------------------------------------


class TestClassifierRegistry:
    """Registry-level structural invariants."""

    def test_known_classifiers_present(self):
        """Core classifiers must exist in the registry."""
        for name in ("SENTIMENT", "ESCALATION", "FEEDBACK"):
            assert name in CLASSIFIERS, f"Expected {name} in CLASSIFIERS"

    def test_conversation_classifiers_present(self):
        for name in (
            "CONVERSATION_RESOLUTION",
            "CONVERSATION_SATISFACTION",
            "CONVERSATION_INTENT",
            "CONVERSATION_ESCALATION",
            "CONVERSATION_SENTIMENT",
        ):
            assert name in CLASSIFIERS

    def test_all_classifiers_have_required_fields(self):
        required = {
            "context_name",
            "system_prompt",
            "user_prompt_template",
            "output_column",
        }
        for name, config in CLASSIFIERS.items():
            missing = required - set(config.keys())
            assert not missing, f"Classifier {name} missing fields: {missing}"

    def test_context_name_matches_key(self):
        for key, config in CLASSIFIERS.items():
            assert config["context_name"] == key, (
                f"context_name '{config['context_name']}' does not match key '{key}'"
            )

    def test_message_placeholder_in_every_template(self):
        """{message} must appear in every user_prompt_template."""
        for name, config in CLASSIFIERS.items():
            template = config["user_prompt_template"]
            assert "{message}" in template, (
                f"Classifier {name}: user_prompt_template missing {{message}} placeholder"
            )

    def test_conversation_unit_classifiers_have_unit_field(self):
        conv_classifiers = [
            k for k, v in CLASSIFIERS.items() if v.get("unit") == "conversation"
        ]
        assert len(conv_classifiers) >= 1, (
            "Expected at least one conversation-unit classifier"
        )
        for name in conv_classifiers:
            assert CLASSIFIERS[name].get("unit") == "conversation"

    def test_post_process_is_callable_when_present(self):
        """Any classifier that defines post_process must be callable."""
        for name, config in CLASSIFIERS.items():
            if "post_process" in config:
                assert callable(config["post_process"]), (
                    f"Classifier {name}: post_process must be callable"
                )

    def test_icsat_present_and_configured(self):
        assert "ICSAT" in CLASSIFIERS
        config = CLASSIFIERS["ICSAT"]
        assert config["output_column"] == "nps_category"
        assert config.get("unit") == "conversation"
        assert callable(config.get("post_process"))


# ---------------------------------------------------------------------------
# get_classifier_config
# ---------------------------------------------------------------------------


class TestGetClassifierConfig:
    def test_returns_config_for_known_classifier(self):
        config = get_classifier_config("SENTIMENT")
        assert config["context_name"] == "SENTIMENT"
        assert config["output_column"] == "sentiment"

    def test_escalation_output_column(self):
        config = get_classifier_config("ESCALATION")
        assert config["output_column"] == "escalation_priority"

    def test_feedback_output_column(self):
        config = get_classifier_config("FEEDBACK")
        assert config["output_column"] == "feedback_type"

    def test_raises_value_error_for_unknown_context(self):
        with pytest.raises(ValueError, match="Unknown context"):
            get_classifier_config("NONEXISTENT_CLASSIFIER")

    def test_error_message_lists_available_contexts(self):
        with pytest.raises(ValueError) as exc_info:
            get_classifier_config("BOGUS")
        assert "SENTIMENT" in str(exc_info.value)

    def test_conversation_resolution_config(self):
        config = get_classifier_config("CONVERSATION_RESOLUTION")
        assert config["unit"] == "conversation"
        assert config["output_column"] == "resolution_status"


# ---------------------------------------------------------------------------
# list_classifiers
# ---------------------------------------------------------------------------


class TestListClassifiers:
    def test_returns_sorted_list(self):
        names = list_classifiers()
        assert names == sorted(names)

    def test_all_classifier_keys_present(self):
        names = list_classifiers()
        assert set(names) == set(CLASSIFIERS.keys())

    def test_returns_list_type(self):
        assert isinstance(list_classifiers(), list)


# ---------------------------------------------------------------------------
# get_classifier_info
# ---------------------------------------------------------------------------


class TestGetClassifierInfo:
    def test_sentiment_defaults(self):
        info = get_classifier_info("SENTIMENT")
        assert info["context_name"] == "SENTIMENT"
        assert info["output_column"] == "sentiment"
        assert info["model"] == "gpt-4o-mini"
        assert info["temperature"] == 0
        assert info["max_tokens"] == 10
        assert info["unit"] == "message"

    def test_conversation_unit_reported_correctly(self):
        info = get_classifier_info("CONVERSATION_RESOLUTION")
        assert info["unit"] == "conversation"

    def test_raises_for_unknown(self):
        with pytest.raises(ValueError):
            get_classifier_info("DOES_NOT_EXIST")

    def test_uses_generic_query_default(self):
        info = get_classifier_info("SENTIMENT")
        assert info["uses_generic_query"] is True


# ---------------------------------------------------------------------------
# EXTRACTION_CONTEXTS
# ---------------------------------------------------------------------------


class TestExtractionContexts:
    def test_conversations_context_present(self):
        assert "CONVERSATIONS" in EXTRACTION_CONTEXTS

    def test_conversations_has_sql_path(self):
        ctx = EXTRACTION_CONTEXTS["CONVERSATIONS"]
        assert "sql_query_path" in ctx
        assert ctx["sql_query_path"]

    def test_conversations_unit_is_message(self):
        ctx = EXTRACTION_CONTEXTS["CONVERSATIONS"]
        assert ctx.get("unit") == "message"


# ---------------------------------------------------------------------------
# _compute_icsat_score
# ---------------------------------------------------------------------------


class TestComputeIcsatScore:
    def _make_df(self, labels: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"nps_category": labels})

    def test_score_computed_correctly(self, caplog):
        import logging

        # test with lowercase labels (as returned by the LLM)
        df = self._make_df(["promoter", "promoter", "passive", "detractor"])
        with caplog.at_level(logging.INFO):
            _compute_icsat_score(df, "nps_category")
        # (2 - 1) / 4 * 100 = 25.0
        assert any("25.0" in r.message for r in caplog.records)

    def test_score_computed_correctly_uppercase(self, caplog):
        import logging

        # test with uppercase labels too
        df = self._make_df(["PROMOTER", "PROMOTER", "PASSIVE", "DETRACTOR"])
        with caplog.at_level(logging.INFO):
            _compute_icsat_score(df, "nps_category")
        assert any("25.0" in r.message for r in caplog.records)

    def test_all_promoters(self, caplog):
        import logging

        df = self._make_df(["promoter", "promoter", "promoter"])
        with caplog.at_level(logging.INFO):
            _compute_icsat_score(df, "nps_category")
        assert any("+100.0" in r.message for r in caplog.records)

    def test_all_detractors(self, caplog):
        import logging

        df = self._make_df(["detractor", "detractor"])
        with caplog.at_level(logging.INFO):
            _compute_icsat_score(df, "nps_category")
        assert any("-100.0" in r.message for r in caplog.records)

    def test_empty_dataframe_does_not_raise(self):
        df = self._make_df([])
        _compute_icsat_score(df, "nps_category")  # should return silently
