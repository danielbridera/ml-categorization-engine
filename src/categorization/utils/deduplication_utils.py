"""
Message Deduplication Utilities

Handles deduplication of messages within workflows to ensure
diverse training data for sentiment labeling.
"""

import pandas as pd

from categorization.settings.log import logger


def deduplicate_messages(
    df: pd.DataFrame,
    text_col: str = "message_text",
    group_by_col: str = "workflow_id",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Deduplicate messages within groups (e.g., workflows).

    Keeps the most recent occurrence of each unique message within each group.

    Args:
        df: DataFrame with messages
        text_col: Column name containing message text
        group_by_col: Column to group by for deduplication
        case_sensitive: Whether to consider case in deduplication

    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)

    logger.info("Deduplicating messages", details=f"Initial count: {initial_count:,}")

    # Create a normalized text column for comparison
    if case_sensitive:
        df["_normalized_text"] = df[text_col].str.strip()
    else:
        df["_normalized_text"] = df[text_col].str.lower().str.strip()

    # Mark duplicates within each group
    df["_is_duplicate"] = df.groupby([group_by_col, "_normalized_text"]).cumcount() > 0

    # Keep only non-duplicates
    df_deduplicated = df[~df["_is_duplicate"]].copy()

    # Drop helper columns
    df_deduplicated = df_deduplicated.drop(
        columns=["_normalized_text", "_is_duplicate"]
    )

    duplicates_removed = initial_count - len(df_deduplicated)
    dedup_rate = (duplicates_removed / initial_count * 100) if initial_count > 0 else 0

    logger.success(
        "Deduplication complete",
        details=f"Removed {duplicates_removed:,} duplicates ({dedup_rate:.1f}%)",
    )

    return df_deduplicated


def filter_by_length(
    df: pd.DataFrame,
    text_col: str = "message_text",
    min_length: int = 6,
    max_length: int | None = None,
) -> pd.DataFrame:
    """
    Filter messages by text length.

    Args:
        df: DataFrame with messages
        text_col: Column name containing message text
        min_length: Minimum character length
        max_length: Maximum character length (None = no limit)

    Returns:
        DataFrame with messages filtered by length
    """
    initial_count = len(df)

    logger.info(
        "Filtering by length",
        details=f"Min: {min_length}, Max: {max_length or 'unlimited'}",
    )

    # Calculate lengths on stripped text (ignore leading/trailing whitespace)
    lengths = df[text_col].str.strip().str.len()

    # Apply filters
    mask = lengths >= min_length
    if max_length:
        mask &= lengths <= max_length

    df_filtered = df[mask].copy()

    removed = initial_count - len(df_filtered)

    logger.success("Length filtering complete", details=f"Removed {removed:,} messages")

    return df_filtered


def remove_null_messages(
    df: pd.DataFrame, text_col: str = "message_text"
) -> pd.DataFrame:
    """
    Remove rows with null or empty message text.

    Args:
        df: DataFrame with messages
        text_col: Column name containing message text

    Returns:
        DataFrame with null messages removed
    """
    initial_count = len(df)

    # Remove nulls
    df_clean = df[df[text_col].notna()].copy()

    # Remove empty strings (after stripping whitespace)
    df_clean = df_clean[df_clean[text_col].str.strip() != ""].copy()

    removed = initial_count - len(df_clean)

    if removed > 0:
        logger.warning(f"Removed {removed:,} null/empty messages")

    return df_clean


def validate_message_quality(df: pd.DataFrame, text_col: str = "message_text") -> dict:
    """
    Validate message quality and return statistics.

    Args:
        df: DataFrame with messages
        text_col: Column name containing message text

    Returns:
        Dictionary with quality statistics
    """
    stats = {
        "total_messages": len(df),
        "null_count": df[text_col].isna().sum(),
        "empty_count": (df[text_col].str.strip() == "").sum(),
        "min_length": df[text_col].str.len().min(),
        "max_length": df[text_col].str.len().max(),
        "mean_length": df[text_col].str.len().mean(),
        "median_length": df[text_col].str.len().median(),
    }

    logger.info(
        "Message quality validation",
        details=f"Total: {stats['total_messages']:,}, Nulls: {stats['null_count']}, "
        f"Median length: {stats['median_length']:.0f}",
    )

    return stats
