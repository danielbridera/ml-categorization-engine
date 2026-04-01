"""
Pipeline Logger - Structured JSON Logging with File Persistence

Provides consistent structured logging for the entire pipeline using standard library
logging with JSON formatting for production aggregation tools (GCO compatible).
Includes file persistence for debugging and auditing.
"""

import logging
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pythonjsonlogger import jsonlogger


def setup_pipeline_logging(
    level: str = "INFO",
    json_enabled: bool = True,
    log_file: str | None = None,
    workflow_id: str | None = None,
    console_enabled: bool = True,
) -> str:
    """Setup global structured logging with dual output (console + file)

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_enabled: Use JSON formatting for structured logs
        log_file: Custom log file path (optional)
        workflow_id: Workflow identifier for file naming
        console_enabled: Enable console output (False for quiet mode)

    Returns:
        str: Path to the log file being used
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Determine log file path
    if log_file:
        log_file_path = Path(log_file)
    else:
        log_file_path = _get_default_log_file_path(workflow_id)

    # Ensure log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup formatters
    formatter: jsonlogger.JsonFormatter | logging.Formatter
    if json_enabled:
        # JSON for aggregation (MLOps feedback)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(module)s %(lineno)d"
        )  # Queryable in GCO
    else:
        # Fallback for dev
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler (stdout for GCO) - only if console_enabled
    if console_enabled:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # File handler with rotation (for persistence and debugging)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,  # Keep 5 backup files
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    # File gets DEBUG level for detailed logging
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    root_logger.propagate = False

    # Log the initialization
    init_logger = logging.getLogger(__name__)
    init_logger.info(
        f"📁 Logging system initialized Console: {level}, File: DEBUG → {log_file_path}"
    )

    return str(log_file_path)


class PipelineLogger:
    """Wrapper for structured logging; maintains current API (success/info/etc.) without passing"""

    def __init__(self, name: str | None = None):
        self.logger = logging.getLogger(name or __name__)  # No passing: auto per module

    def info(self, message: str, emoji: str = "ℹ", details: str = "") -> None:
        """Log info message with optional emoji and details."""
        msg = f"{emoji} {message}"
        if details:
            msg += f" {details}"
        self.logger.info(msg)

    def success(self, message: str, details: str = "") -> None:
        """Log success message with green checkmark emoji."""
        msg = f"✅ {message}"
        if details:
            msg += f" {details}"
        self.logger.info(msg)

    def warning(self, message: str, details: str = "") -> None:
        """Log warning message with warning emoji."""
        msg = f"⚠️ {message}"
        if details:
            msg += f" {details}"
        self.logger.warning(msg)

    def error(self, message: str, details: str = "") -> None:
        """Log error message with red X emoji."""
        msg = f"❌ {message}"
        if details:
            msg += f" {details}"
        self.logger.error(msg)

    def debug(self, message: str, details: str = "") -> None:
        """Log debug message with magnifying glass emoji."""
        msg = f"🔍 {message}"
        if details:
            msg += f" {details}"
        self.logger.debug(msg)

    def phase_start(self, phase_name: str, details: str = "") -> None:
        """Log the start of a pipeline phase."""
        msg = f"🚀 Phase start: {phase_name}"
        if details:
            msg += f" {details}"
        self.logger.info(msg)

    def phase_complete(self, phase_name: str, count: int) -> None:
        """Log completion of a pipeline phase with item count."""
        self.logger.info(f"✅ Phase complete: {phase_name} {count:,} items processed")

    def batch_progress(self, current: int, total: int, operation: str) -> None:
        """Log batch processing progress with current/total counts."""
        self.logger.info(f"🔄 {operation} batch {current}/{total}")

    def pipeline_start(self, workflow_id: str, date_range: str, samples: int) -> None:
        """Log pipeline startup with workflow details."""
        self.logger.info(
            f"🚀 Pipeline starting Workflow: {workflow_id}, Range: {date_range}, Samples: {samples:,}"
        )

    def pipeline_complete(self, workflow_id: str, total_time: float) -> None:
        """Log pipeline completion with timing information."""
        self.logger.info(
            f"🎉 Pipeline completed Workflow: {workflow_id}, Time: {total_time:.1f}s"
        )

    def save_file(self, file_type: str, count: int, path: str) -> None:
        """Log file save operation with record count and path."""
        self.logger.info(f"💾 Saved {file_type} {count:,} records to {path}")

    def load_cache(self, cache_type: str, details: str | None = None) -> None:
        """Log cache loading operation."""
        msg = f"Loading cached {cache_type}"
        if details:
            msg += f" {details}"
        else:
            msg = f"Using cached {cache_type}"
        self.logger.info(f"📦 {msg}")

    def evaluation_batch_complete(
        self, batch_num: int, total_batches: int, records: int
    ) -> None:
        """Log successful completion of evaluation batch."""
        self.logger.info(
            f"✅ Batch completed {batch_num}/{total_batches} ({records} metric results)"
        )

    def evaluation_batch_failed(
        self, batch_num: int, total_batches: int, error: str
    ) -> None:
        """Log failed evaluation batch with error details."""
        self.logger.error(f"❌ Batch failed {batch_num}/{total_batches} Error: {error}")


def _get_default_log_file_path(workflow_id: str | None = None) -> Path:
    """Generate default log file path with timestamp and workflow ID"""
    logs_dir = Path("logs")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if workflow_id:
        # Workflow-specific log file
        filename = f"workflow_{workflow_id}_{timestamp}.log"
    else:
        # General pipeline log file
        filename = f"pipeline_{timestamp}.log"

    return logs_dir / filename


def get_latest_log_file() -> Path | None:
    """Get the most recent log file"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None

    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        return None

    # Return the most recently modified log file
    return max(log_files, key=lambda f: f.stat().st_mtime)


def cleanup_old_logs(days_to_keep: int = 30) -> int:
    """Clean up log files older than specified days

    Args:
        days_to_keep: Number of days to keep log files

    Returns:
        Number of files deleted
    """

    logs_dir = Path("logs")
    if not logs_dir.exists():
        return 0

    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    deleted_count = 0

    for log_file in logs_dir.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted_count += 1

    return deleted_count


logger = PipelineLogger()
