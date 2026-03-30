"""
Tests for deduplication utilities (utils/deduplication_utils.py).

Covers:
- remove_null_messages: null values, empty strings, whitespace-only strings
- filter_by_length: min/max thresholds, whitespace stripping before length calc
- deduplicate_messages: within-group dedup, case-insensitive by default,
                        case-sensitive option, cross-group non-dedup
"""

import pandas as pd
import pytest

from categorization.utils.deduplication_utils import (
    deduplicate_messages,
    filter_by_length,
    remove_null_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(texts: list, workflow_ids: list | None = None, message_ids: list | None = None) -> pd.DataFrame:
    n = len(texts)
    return pd.DataFrame(
        {
            "message_id": pd.array(message_ids or list(range(n))),
            "message_text": pd.array(texts, dtype=pd.StringDtype()),
            "workflow_id": pd.array(workflow_ids or (["wf1"] * n), dtype=pd.StringDtype()),
        }
    )


# ---------------------------------------------------------------------------
# remove_null_messages
# ---------------------------------------------------------------------------


class TestRemoveNullMessages:
    def test_removes_none_values(self):
        df = make_df(["hello", None, "world"])
        result = remove_null_messages(df)
        assert len(result) == 2
        assert result["message_text"].notna().all()

    def test_removes_empty_strings(self):
        df = make_df(["hello", "", "world"])
        result = remove_null_messages(df)
        assert len(result) == 2

    def test_removes_whitespace_only_strings(self):
        df = make_df(["hello", "   ", "\t\n", "world"])
        result = remove_null_messages(df)
        assert len(result) == 2

    def test_keeps_valid_messages(self):
        df = make_df(["hello", "world", "test"])
        result = remove_null_messages(df)
        assert len(result) == 3

    def test_returns_copy_not_inplace(self):
        df = make_df(["hello", None])
        original_len = len(df)
        result = remove_null_messages(df)
        assert len(df) == original_len  # original unchanged
        assert len(result) == 1

    def test_empty_dataframe_returns_empty(self):
        df = make_df([])
        result = remove_null_messages(df)
        assert len(result) == 0

    def test_custom_text_col(self):
        df = pd.DataFrame({"custom_text": ["hi", None, "bye"]})
        result = remove_null_messages(df, text_col="custom_text")
        assert len(result) == 2

    def test_all_nulls_returns_empty(self):
        df = make_df([None, None, None])
        result = remove_null_messages(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_length
# ---------------------------------------------------------------------------


class TestFilterByLength:
    def test_removes_short_messages(self):
        df = make_df(["hi", "hello world", "ok"])
        result = filter_by_length(df, min_length=5)
        assert len(result) == 1
        assert result["message_text"].iloc[0] == "hello world"

    def test_keeps_messages_at_exact_min_length(self):
        # "abcde" has length 5; min_length=5 should keep it
        df = make_df(["abcde", "ab"])
        result = filter_by_length(df, min_length=5)
        assert len(result) == 1
        assert result["message_text"].iloc[0] == "abcde"

    def test_strips_whitespace_before_length_check(self):
        # "  hi  " stripped = "hi" (length 2), should be removed with min_length=4
        df = make_df(["  hi  ", "hello world"])
        result = filter_by_length(df, min_length=4)
        assert len(result) == 1
        assert result["message_text"].iloc[0] == "hello world"

    def test_whitespace_padded_long_message_passes(self):
        # "  hello  " stripped = "hello" (length 5), passes min_length=5
        df = make_df(["  hello  ", "ab"])
        result = filter_by_length(df, min_length=5)
        assert len(result) == 1

    def test_max_length_filter(self):
        df = make_df(["short", "a" * 200, "medium text"])
        result = filter_by_length(df, min_length=1, max_length=20)
        texts = result["message_text"].tolist()
        assert "a" * 200 not in texts
        assert "short" in texts
        assert "medium text" in texts

    def test_no_max_length_keeps_long_messages(self):
        df = make_df(["short", "a" * 1000])
        result = filter_by_length(df, min_length=1, max_length=None)
        assert len(result) == 2

    def test_default_min_length_is_6(self):
        df = make_df(["hello", "hello!", "abc"])
        result = filter_by_length(df)
        # "hello" len=5 < 6, "hello!" len=6 >= 6, "abc" len=3 < 6
        assert len(result) == 1
        assert result["message_text"].iloc[0] == "hello!"

    def test_empty_dataframe_returns_empty(self):
        df = make_df([])
        result = filter_by_length(df, min_length=5)
        assert len(result) == 0

    def test_custom_text_col(self):
        df = pd.DataFrame({"text": ["hi", "hello world"]})
        result = filter_by_length(df, text_col="text", min_length=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# deduplicate_messages
# ---------------------------------------------------------------------------


class TestDeduplicateMessages:
    def test_removes_exact_duplicates_within_group(self):
        df = make_df(
            ["hello", "hello", "world"],
            workflow_ids=["wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        assert len(result) == 2

    def test_case_insensitive_by_default(self):
        df = make_df(
            ["Hello", "hello", "HELLO", "world"],
            workflow_ids=["wf1", "wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, case_sensitive=False, group_by_col="workflow_id")
        assert len(result) == 2  # "Hello" (first) + "world"

    def test_case_sensitive_keeps_different_cases(self):
        df = make_df(
            ["Hello", "hello", "HELLO"],
            workflow_ids=["wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, case_sensitive=True, group_by_col="workflow_id")
        assert len(result) == 3

    def test_no_dedup_across_different_groups(self):
        # Same message text but different workflow_ids should NOT be deduped
        df = make_df(
            ["hello", "hello"],
            workflow_ids=["wf1", "wf2"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = make_df(
            ["apple", "banana", "cherry"],
            workflow_ids=["wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        assert len(result) == 3

    def test_removes_helper_columns(self):
        df = make_df(
            ["hello", "hello", "world"],
            workflow_ids=["wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        assert "_normalized_text" not in result.columns
        assert "_is_duplicate" not in result.columns

    def test_empty_dataframe_returns_empty(self):
        df = make_df([])
        result = deduplicate_messages(df)
        assert len(result) == 0

    def test_strips_whitespace_before_comparison(self):
        df = make_df(
            ["  hello  ", "hello", "world"],
            workflow_ids=["wf1", "wf1", "wf1"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        # "  hello  " and "hello" are same after strip+lower
        assert len(result) == 2

    def test_multiple_groups_with_duplicates(self):
        df = make_df(
            ["dup", "dup", "unique", "dup", "dup"],
            workflow_ids=["wf1", "wf1", "wf1", "wf2", "wf2"],
        )
        result = deduplicate_messages(df, group_by_col="workflow_id")
        assert len(result) == 3  # wf1: "dup"(kept) + "unique"; wf2: "dup"(kept)
