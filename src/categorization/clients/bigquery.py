# mypy: disable-error-code="no-untyped-def"
import time
from typing import Any, Callable, TypeVar

import pandas as pd
from google.api_core.exceptions import GoogleAPICallError, ServiceUnavailable
from google.cloud import bigquery

from categorization.settings.credentials import GOOGLE_CLOUD_PROJECT
from categorization.settings.log import logger

_BQ_RETRYABLE = (ServiceUnavailable, ConnectionError, TimeoutError)

_T = TypeVar("_T")


def _retry_bq(func: Callable[[], _T], max_retries: int = 3, initial_delay: float = 2.0) -> _T:
    """Retry a BigQuery call with exponential backoff on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return func()
        except _BQ_RETRYABLE as e:
            if attempt == max_retries:
                raise
            logger.warning(
                f"BigQuery transient error (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)
            delay *= 2
        except GoogleAPICallError as e:
            # Non-retryable API errors (bad query, permissions, etc.)
            logger.error(f"BigQuery API error: {e}")
            raise
    raise RuntimeError("Unreachable")


class BigQueryClient:
    def __init__(self) -> None:
        logger.info("Initializing BigQuery client with Application Default Credentials")

        # Use Application Default Credentials (ADC)
        self.client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)

    def execute(self, query: str) -> bigquery.QueryJob:
        """
        Executes query in BigQuery.

            Args:
                query (str): query to be executed.

            Returns:
                bigquery.QueryJob: The query job object.
        """
        logger.info(f"Executing query: {query[:100]}...")
        job = self.client.query(query)
        return job

    def fetchone(self, query: str) -> dict[str, Any] | None:
        """
        Executes query in BigQuery and fetches one record.

            Args:
                query (str): query to be executed.

            Returns:
                Dict with first record of the query result, or None if no results.
        """
        job = self.execute(query)
        results = job.result()

        for row in results:
            # Convert Row to dict and return first record
            return dict(row.items())
        return None

    def fetchall(self, query: str) -> list[dict[str, Any]]:
        """
        Executes query in BigQuery and fetches all records.

            Args:
                query (str): query to be executed.

            Returns:
                List of all records (dicts) of the query result.
        """
        job = self.execute(query)
        results = job.result()

        records = []
        for row in results:
            records.append(dict(row.items()))

        logger.info(f"Fetched {len(records)} records")
        return records

    def execute_to_file(
        self, query: str, output_path: str, columns: list[str] | None = None
    ) -> None:
        """
        Executes query in BigQuery and writes results to a Parquet file.

            Args:
                query (str): query to be executed.
                output_path (str): output file path (should end with .parquet).
                columns (List): order in which columns are going to be written to the file.
                    If None, the order of the query is kept.
                    Defaults to None.
        """
        logger.info(f"Executing query and converting to DataFrame for: {output_path}")

        def _run():
            job = self.client.query(query)
            return job.to_dataframe()

        df = _retry_bq(_run)

        if df.empty:
            logger.warning("No records found in query results")
            # Still save an empty parquet file to indicate completion
            df.to_parquet(output_path, index=False)
            return

        # Reorder columns if specified
        if columns:
            # Only use columns that exist in the DataFrame
            available_columns = [col for col in columns if col in df.columns]
            if available_columns:
                df = df[available_columns]
            else:
                logger.warning(
                    "None of the specified columns found in results. Using all columns."
                )

        # Save to Parquet format
        logger.info(f"Saving {len(df)} records to Parquet: {output_path}")
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(f"Successfully wrote {len(df)} records to {output_path}")

    def execute_to_dataframe(self, query: str) -> pd.DataFrame:
        """
        Executes query in BigQuery and returns results as a pandas DataFrame.

            Args:
                query (str): query to be executed.

            Returns:
                pandas.DataFrame: DataFrame containing the query results.
        """
        logger.info("Executing query and converting to DataFrame")

        def _run():
            job = self.client.query(query)
            return job.to_dataframe()

        df = _retry_bq(_run)

        logger.info(
            f"Retrieved DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        return df

    def get_table_schema(self, table_id: str) -> list[dict[str, Any]]:
        """
        Gets the schema of a BigQuery table.

            Args:
                table_id (str): Table ID in format 'dataset.table' or 'project.dataset.table'.

            Returns:
                List of dictionaries containing field information.
        """
        table_ref = self.client.get_table(table_id)
        schema = []

        for field in table_ref.schema:
            schema.append(
                {
                    "name": field.name,
                    "field_type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
            )

        return schema

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        table_id: str,
        write_disposition: str = "WRITE_APPEND",
        partition_field: str | None = None,
        cluster_fields: list[str] | None = None,
    ) -> int:
        """
        Upload a pandas DataFrame to BigQuery table.

        Note: Assumes table already exists in BigQuery. Use make_tables to create tables.

        Args:
            df: DataFrame to upload
            table_id: Full table ID in format 'project.dataset.table' or 'dataset.table'
            write_disposition: Write mode ('WRITE_APPEND', 'WRITE_TRUNCATE', 'WRITE_EMPTY')
            partition_field: Field name for table partitioning (optional)
            cluster_fields: List of field names for table clustering (optional)

        Returns:
            Number of rows uploaded

        Raises:
            ValueError: If table_id format is invalid or DataFrame is empty
            google.cloud.exceptions.GoogleCloudError: If upload fails
        """
        if df.empty:
            logger.warning(f"DataFrame is empty, skipping upload to {table_id}")
            return 0

        # Parse table_id using BigQuery's built-in parser
        try:
            # Handle both "dataset.table" and "project.dataset.table" formats
            if table_id.count(".") == 1:
                # Format: "dataset.table" - prepend project
                full_table_id = f"{self.client.project}.{table_id}"
            else:
                # Format: "project.dataset.table" - use as-is
                full_table_id = table_id

            # Validate by creating a Table reference
            bigquery.Table.from_string(full_table_id)
        except Exception as e:
            raise ValueError(
                f"Invalid table_id format: {table_id}. Expected 'dataset.table' or 'project.dataset.table'. Error: {e}"
            ) from e

        logger.info(f"Uploading {len(df)} rows to {full_table_id}")

        # Configure job
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            autodetect=True,  # Auto-detect schema from DataFrame
        )

        # Configure partitioning (if table supports it)
        if partition_field:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY, field=partition_field
            )
            logger.info(f"Partitioning by field: {partition_field}")

        # Configure clustering (if table supports it)
        if cluster_fields:
            job_config.clustering_fields = cluster_fields
            logger.info(f"Clustering by fields: {cluster_fields}")

        # Upload data
        try:
            job = self.client.load_table_from_dataframe(
                df, full_table_id, job_config=job_config
            )

            # Wait for job to complete
            job.result()

            logger.info(f"Successfully uploaded {len(df)} rows to {full_table_id}")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to upload data to {full_table_id}: {e!s}")
            raise
