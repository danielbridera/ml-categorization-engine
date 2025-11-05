"""
General Utility Functions

Common helper functions used across the sentiment analysis pipeline.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from categorization.settings.log import logger


def load_query(file_path: Path | str) -> str:
    """
    Load SQL query from file.

    Args:
        file_path: Path to SQL file

    Returns:
        SQL query string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"SQL file not found: {file_path}")

    with open(file_path, "r") as f:
        query = f.read()

    logger.debug(f"Loaded SQL query from {file_path}")

    return query


def load_parquet_with_validation(
    file_path: Path | str, step_name: str, required_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Load parquet file with validation.

    Args:
        file_path: Path to parquet file
        step_name: Name of the step for error messages
        required_columns: List of required column names

    Returns:
        DataFrame loaded from parquet

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found for {step_name}: {file_path}\n"
            f"Please run the previous step first."
        )

    logger.info(f"Loading data from {file_path.name}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file_path.name}: {missing_columns}"
            )

    return df


def save_parquet(
    df: pd.DataFrame, file_path: Path | str, description: str = "data"
) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        file_path: Output path
        description: Description of data for logging
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {description}", details=f"{len(df):,} rows → {file_path.name}")

    df.to_parquet(file_path, index=False, engine="pyarrow")

    logger.success(f"Saved {description}", details=f"{file_path}")


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string is in YYYY-MM-DD format.

    Args:
        date_str: Date string to validate

    Returns:
        True if valid

    Raises:
        ValueError: If date format is invalid
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def print_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print summary statistics for a DataFrame.

    Args:
        df: DataFrame to summarize
        name: Name of the DataFrame for display
    """
    print(f"\n{'=' * 60}")
    print(f"{name} Summary")
    print(f"{'=' * 60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn Types:")
    print(df.dtypes)
    print("\nNull Counts:")
    print(df.isnull().sum())
    print(f"{'=' * 60}\n")
