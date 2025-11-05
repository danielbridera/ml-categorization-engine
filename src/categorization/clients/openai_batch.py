"""
OpenAI Batch API Client

Generic client for batch categorization tasks using OpenAI's Batch API (50% cheaper).
Supports sentiment analysis, escalation detection, and other classification tasks.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from categorization.settings.credentials import OPENAI_API_KEY
from categorization.settings.log import logger

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> T:
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        OpenAIError: If all retries exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise
            logger.warning(
                f"Rate limit hit. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay *= backoff_factor
        except (APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries:
                logger.error(f"Network error after {max_retries} retries: {e}")
                raise
            logger.warning(
                f"Network error. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay *= backoff_factor
        except OpenAIError as e:
            # Don't retry other OpenAI errors (invalid requests, auth, etc.)
            logger.error(f"OpenAI API error: {e}")
            raise


class OpenAIBatchClient:
    """Generic client for OpenAI Batch API categorization operations"""

    def __init__(self):
        """Initialize OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Initialized OpenAI Batch API client")

    def create_batch_file(
        self,
        messages: list[str],
        output_path: str | Path,
        system_prompt: str,
        user_prompt_template: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 10,
    ) -> str:
        """
        Create JSONL batch request file for OpenAI Batch API.

        Args:
            messages: List of message texts to categorize
            output_path: Path to save JSONL file
            system_prompt: System message for the model
            user_prompt_template: User prompt template with {message} placeholder
            model: OpenAI model to use
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Path to created batch file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Creating batch request file",
            details=f"{len(messages)} messages → {output_path}",
        )

        with open(output_path, "w") as f:
            for idx, message in enumerate(messages):
                request = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt_template.format(message=message),
                            },
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                }
                f.write(json.dumps(request) + "\n")

        logger.success("Batch file created", details=f"{len(messages)} requests")

        return str(output_path)

    def submit_batch(
        self, file_path: str | Path, task_description: str = "batch_categorization"
    ) -> dict[str, Any]:
        """
        Submit batch file to OpenAI for processing.

        Args:
            file_path: Path to JSONL batch file
            task_description: Description of the task for metadata

        Returns:
            Dictionary with batch_id and file_id
        """
        file_path = Path(file_path)

        logger.info("Uploading batch file to OpenAI")

        # Upload file with retry logic
        def upload_file():
            with open(file_path, "rb") as f:
                return self.client.files.create(file=f, purpose="batch")

        batch_input_file = retry_with_backoff(upload_file)

        logger.success("File uploaded", details=f"File ID: {batch_input_file.id}")

        # Create batch job with retry logic
        logger.info("Creating batch job")

        def create_batch():
            return self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": task_description, "file_name": file_path.name},
            )

        batch_job = retry_with_backoff(create_batch)

        logger.success("Batch job created", details=f"Batch ID: {batch_job.id}")

        return {
            "batch_id": batch_job.id,
            "file_id": batch_input_file.id,
            "status": batch_job.status,
        }

    def check_status(self, batch_id: str) -> dict[str, Any]:
        """
        Check status of a batch job.

        Args:
            batch_id: OpenAI batch ID

        Returns:
            Dictionary with status information
        """

        def retrieve_status():
            return self.client.batches.retrieve(batch_id)

        batch_status = retry_with_backoff(retrieve_status)

        status_info = {
            "batch_id": batch_id,
            "status": batch_status.status,
            "total_requests": batch_status.request_counts.total,
            "completed_requests": batch_status.request_counts.completed,
            "failed_requests": batch_status.request_counts.failed,
            "output_file_id": batch_status.output_file_id,
            "error_file_id": batch_status.error_file_id,
        }

        logger.info(
            f"Batch status: {batch_status.status}",
            details=f"Completed: {batch_status.request_counts.completed}/{batch_status.request_counts.total}",
        )

        return status_info

    def download_results(
        self, batch_id: str, output_path: str | Path
    ) -> dict[str, Any]:
        """
        Download completed batch results.

        Args:
            batch_id: OpenAI batch ID
            output_path: Path to save results JSONL file

        Returns:
            Dictionary with download statistics and cost info
        """
        output_path = Path(output_path)

        # Check status first with retry logic
        def retrieve_status():
            return self.client.batches.retrieve(batch_id)

        batch_status = retry_with_backoff(retrieve_status)

        if batch_status.status != "completed":
            raise ValueError(
                f"Batch not completed yet. Current status: {batch_status.status}"
            )

        if not batch_status.output_file_id:
            raise ValueError("No output file available for this batch")

        logger.info("Downloading batch results")

        # Download results with retry logic
        result_file_id = batch_status.output_file_id

        def download_file():
            return self.client.files.content(result_file_id)

        result = retry_with_backoff(download_file)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(result.content)

        logger.success("Results downloaded", details=f"Saved to {output_path}")

        # Parse results for statistics
        stats = self._parse_results_stats(output_path)

        return stats

    def _parse_results_stats(self, results_path: Path) -> dict[str, Any]:
        """
        Parse results file to extract statistics.

        Args:
            results_path: Path to results JSONL file

        Returns:
            Dictionary with statistics
        """
        total_input_tokens = 0
        total_output_tokens = 0
        successful = 0
        failed = 0

        with open(results_path, "r") as f:
            for line in f:
                result = json.loads(line)

                if result.get("error") is not None:
                    failed += 1
                else:
                    successful += 1
                    usage = result["response"]["body"]["usage"]
                    total_input_tokens += usage["prompt_tokens"]
                    total_output_tokens += usage["completion_tokens"]

        # Calculate cost with 50% Batch API discount
        input_cost = (total_input_tokens / 1_000_000) * 0.150 * 0.5
        output_cost = (total_output_tokens / 1_000_000) * 0.600 * 0.5
        total_cost = input_cost + output_cost

        stats = {
            "successful": successful,
            "failed": failed,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

        logger.info(
            "Results parsed",
            details=f"Success: {successful}, Failed: {failed}, Cost: ${total_cost:.4f}",
        )

        return stats

    def parse_labels(self, results_path: Path, total_requests: int) -> list[str]:
        """
        Parse category labels from results file.

        Args:
            results_path: Path to results JSONL file
            total_requests: Expected total number of requests

        Returns:
            List of category labels (ordered by request index)
        """
        labels_dict = {}
        failed_requests = []

        with open(results_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")

                    # Safely parse index from custom_id
                    try:
                        parts = custom_id.split("-")
                        if len(parts) < 2:
                            raise ValueError(f"Invalid custom_id format: {custom_id}")
                        idx = int(parts[1])
                    except (ValueError, IndexError):
                        logger.error(
                            f"Line {line_num}: Invalid custom_id format: {custom_id}"
                        )
                        continue

                    if result.get("error") is not None:
                        # Request failed
                        labels_dict[idx] = "error"
                        failed_requests.append((idx, str(result["error"])))
                    else:
                        # Extract label with safe navigation
                        try:
                            label = (
                                result["response"]["body"]["choices"][0]["message"][
                                    "content"
                                ]
                                .strip()
                                .lower()
                            )
                            labels_dict[idx] = label
                        except (KeyError, IndexError, AttributeError) as e:
                            logger.error(
                                f"Line {line_num}: Failed to extract label: {e}"
                            )
                            labels_dict[idx] = "error"
                            failed_requests.append((idx, f"Parse error: {e}"))

                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON: {e}")
                    continue

        # Create ordered list (fill missing with 'error')
        labels = []
        for i in range(total_requests):
            if i in labels_dict:
                labels.append(labels_dict[i])
            else:
                labels.append("error")
                failed_requests.append((i, "Missing from results"))

        if failed_requests:
            logger.warning(
                f"{len(failed_requests)} requests failed or missing",
                details=f"First failures: {failed_requests[:3]}",
            )

        return labels

    # Backward compatibility alias
    def parse_sentiments(self, results_path: Path, total_requests: int) -> list[str]:
        """
        Legacy method for parsing sentiment labels. Use parse_labels() instead.

        Args:
            results_path: Path to results JSONL file
            total_requests: Expected total number of requests

        Returns:
            List of sentiment labels (ordered by request index)
        """
        return self.parse_labels(results_path, total_requests)
