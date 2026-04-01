"""
Experiment Metadata Management

Handles creation, loading, and updating of experiment metadata.
Enables automatic discovery of experiments and tracking of pipeline progress.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from categorization.settings.log import logger


class ExperimentMetadata:
    """Manages experiment metadata and state tracking"""

    def __init__(self, experiment_dir: Path):
        """Initialize with existing experiment directory"""
        self.experiment_dir = Path(experiment_dir)
        self.metadata_file = self.experiment_dir / "metadata.json"
        self.batch_info_file = self.experiment_dir / "batch_info.json"

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found in {experiment_dir}")

        self.metadata = self._load_metadata()
        self.experiment_id = self.metadata["experiment_id"]

    @classmethod
    def create_new(
        cls,
        start_date: str,
        end_date: str,
        n_samples: int,
        min_length: int,
        context_name: str,
        unit: str = "message",
        conversation_timeout: str = "30m",
        workflow_names: str | None = None,
        base_dir: str = "experiments",
    ) -> "ExperimentMetadata":
        """
        Create new experiment with unique ID.

        Args:
            start_date: Start date for data extraction (YYYY-MM-DD)
            end_date: End date for data extraction (YYYY-MM-DD)
            n_samples: Number of samples to extract
            min_length: Minimum message length
            context_name: Context name for experiment ID prefix
            unit: Classification unit — "message" (default) or "conversation"
            conversation_timeout: Inactivity timeout for conversation grouping (e.g. "30m")
            workflow_names: Comma-separated workflow names filter (None = all workflows)
            base_dir: Base directory for experiments

        Returns:
            ExperimentMetadata instance for the new experiment
        """
        # Generate unique experiment ID with timestamp using context name as prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{context_name.lower()}_{timestamp}"

        # Create experiment directory
        experiment_dir = Path(base_dir) / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata: dict[str, Any] = {
            "experiment_id": experiment_id,
            "context_name": context_name,
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "n_samples": n_samples,
                "min_length": min_length,
                "unit": unit,
                "conversation_timeout": conversation_timeout,
                "workflow_names": workflow_names,
            },
            "steps_completed": [],
            "batch_info": None,
            "files": {
                f"{context_name.lower()}_context": None,
                f"{context_name.lower()}_preprocessed": None,
                f"{context_name.lower()}_labeled": None,
            },
            "stats": {
                f"{context_name.lower()}_extracted": None,
                f"{context_name.lower()}_after_preprocess": None,
                f"{context_name.lower()}_labeled": None,
            },
        }

        # Save metadata
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(
            f"Created experiment: {experiment_id}",
            details=f"Directory: {experiment_dir}",
        )

        return cls(experiment_dir)

    @classmethod
    def load(
        cls,
        experiment_id: str | None = None,
        base_dir: str = "experiments",
        context_name: str | None = None,
        workflow_names: str | None = None,
    ) -> "ExperimentMetadata":
        """
        Load existing experiment.

        Args:
            experiment_id: Specific experiment ID to load (None = latest)
            base_dir: Base directory for experiments
            context_name: Optional context name for filtering latest experiment
            workflow_names: Comma-separated workflow names filter (None = all workflows)

        Returns:
            ExperimentMetadata instance
        """
        if experiment_id:
            experiment_dir = Path(base_dir) / experiment_id
            if not experiment_dir.exists():
                raise ValueError(f"Experiment not found: {experiment_id}")
        else:
            # Auto-discover latest experiment filtered by context and optionally workflow_names
            found = cls._find_latest_experiment(base_dir, context_name, workflow_names)
            if not found:
                context_msg = f" for context '{context_name}'" if context_name else ""
                raise ValueError(
                    f"No experiments found{context_msg} in {base_dir}. Run make_context first."
                )
            experiment_dir = found

        return cls(experiment_dir)

    @staticmethod
    def _find_latest_experiment(
        base_dir: str = "experiments",
        context_name: str | None = None,
        workflow_names: str | None = None,
    ) -> Path | None:
        """
        Find most recent experiment folder, optionally filtered by context and workflow_names.

        Args:
            base_dir: Base directory for experiments
            context_name: Optional context name to filter by (e.g., "CONVERSATIONS")
            workflow_names: Optional comma-separated workflow names to match against metadata

        Returns:
            Path to latest matching experiment or None
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return None

        pattern = f"{context_name.lower()}_*" if context_name else "*_*"
        candidates = sorted(base_path.glob(pattern))

        if not workflow_names:
            return candidates[-1] if candidates else None

        # Filter by workflow_names stored in metadata
        normalized = ",".join(sorted(n.strip() for n in workflow_names.split(",")))
        for experiment_dir in reversed(candidates):
            metadata_file = experiment_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            with open(metadata_file) as f:
                meta = json.load(f)
            stored = meta.get("parameters", {}).get("workflow_names", "")
            if stored:
                stored_normalized = ",".join(
                    sorted(n.strip() for n in stored.split(","))
                )
                if stored_normalized == normalized:
                    return experiment_dir

        # Fall back to latest if no workflow_names match
        return candidates[-1] if candidates else None

    @staticmethod
    def list_experiments(
        base_dir: str = "experiments", context_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List all experiments with their status.

        Args:
            base_dir: Base directory for experiments
            context_name: Optional context name to filter by (e.g., "SENTIMENT")

        Returns:
            List of experiment info dictionaries
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        # Use context-specific pattern or generic pattern
        pattern = f"{context_name.lower()}_*" if context_name else "*_*"
        experiments = []
        for exp_dir in sorted(base_path.glob(pattern), reverse=True):
            metadata_file = exp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                experiments.append(
                    {
                        "experiment_id": metadata["experiment_id"],
                        "created_at": metadata["created_at"],
                        "steps_completed": metadata["steps_completed"],
                        "batch_status": metadata.get("batch_info", {}).get("status")
                        if metadata.get("batch_info")
                        else None,
                        "parameters": metadata["parameters"],
                    }
                )

        return experiments

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from file"""
        with open(self.metadata_file, "r") as f:
            return cast(dict[str, Any], json.load(f))

    def _save_metadata(self) -> None:
        """Save metadata to file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def update_step(self, step_name: str, stats: dict[str, Any]) -> None:
        """
        Mark step as completed and update stats.

        Args:
            step_name: Name of the completed step
            stats: Dictionary of statistics to update
        """
        # Add to completed steps if not already there
        if step_name not in self.metadata["steps_completed"]:
            self.metadata["steps_completed"].append(step_name)

        # Update stats
        self.metadata["stats"].update(stats)

        # Update timestamp
        self.metadata[f"{step_name}_completed_at"] = datetime.now().isoformat()

        self._save_metadata()

        logger.success(f"Step completed: {step_name}", details=f"Stats: {stats}")

    def save_batch_info(
        self,
        batch_id: str,
        file_id: str,
        total_requests: int,
        extraction_context: str | None = None,
        classifier: str | None = None,
    ) -> None:
        """
        Save batch submission details.

        Args:
            batch_id: OpenAI batch ID
            file_id: OpenAI file ID
            total_requests: Number of requests in batch
            extraction_context: Context name used for data extraction (e.g. "CONVERSATIONS")
            classifier: Classifier applied (e.g. "CONVERSATION_SENTIMENT")
        """
        batch_info = {
            "batch_id": batch_id,
            "file_id": file_id,
            "submitted_at": datetime.now().isoformat(),
            "completed_at": None,
            "status": "submitted",
            "total_requests": total_requests,
            "completed_requests": 0,
            "failed_requests": 0,
            "estimated_completion": datetime.fromtimestamp(
                datetime.now().timestamp() + 24 * 3600
            ).isoformat(),
            "cost": {"input_tokens": None, "output_tokens": None, "total_cost": None},
            "extraction_context": extraction_context,
            "classifier": classifier,
        }

        # Save to separate batch_info.json file
        with open(self.batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        # Also update metadata
        self.metadata["batch_info"] = {
            "batch_id": batch_id,
            "status": "submitted",
            "submitted_at": batch_info["submitted_at"],
            "extraction_context": extraction_context,
            "classifier": classifier,
        }
        self._save_metadata()

        logger.success(
            "Batch info saved",
            details=f"Batch ID: {batch_id}, Requests: {total_requests}",
        )

    def get_batch_info(self) -> dict[str, Any] | None:
        """Get batch submission details"""
        if not self.batch_info_file.exists():
            return None

        with open(self.batch_info_file, "r") as f:
            return cast(dict[str, Any], json.load(f))

    def update_batch_status(
        self,
        status: str,
        completed_requests: int = 0,
        failed_requests: int = 0,
        cost_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Update batch processing status.

        Args:
            status: Current status (in_progress, completed, failed)
            completed_requests: Number of completed requests
            failed_requests: Number of failed requests
            cost_info: Cost information dictionary
        """
        batch_info = self.get_batch_info()
        if not batch_info:
            raise ValueError("No batch info found")

        batch_info["status"] = status
        batch_info["completed_requests"] = completed_requests
        batch_info["failed_requests"] = failed_requests

        if status == "completed":
            batch_info["completed_at"] = datetime.now().isoformat()

        if cost_info:
            batch_info["cost"].update(cost_info)

        # Save updated batch info
        with open(self.batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        # Update metadata
        self.metadata["batch_info"]["status"] = status
        self._save_metadata()

        logger.info(
            f"Batch status updated: {status}",
            details=f"Completed: {completed_requests}/{batch_info['total_requests']}",
        )

    def get_file_path(self, file_type: str) -> Path:
        """Get path for a specific file type"""
        # Get context name from metadata (with fallback for legacy experiments)
        context_prefix = self.metadata.get("context_name", "MESSAGES").lower()

        file_mapping = {
            "context": f"{context_prefix}_context.parquet",
            "preprocess": f"{context_prefix}_preprocessed.parquet",
            "labeled": f"{context_prefix}_labeled.parquet",
            "batch_requests": "batch_requests.jsonl",
            "batch_results": "batch_results.jsonl",
        }

        if file_type not in file_mapping:
            raise ValueError(f"Unknown file type: {file_type}")

        return self.experiment_dir / file_mapping[file_type]

    def update_file(self, file_type: str) -> None:
        """Update metadata to track that a file was created"""
        # Get context name from metadata (with fallback for legacy experiments)
        context_prefix = self.metadata.get("context_name", "MESSAGES").lower()

        self.metadata["files"][f"{context_prefix}_{file_type}"] = self.get_file_path(
            file_type
        ).name
        self._save_metadata()

    def get_parameters(self) -> dict[str, Any]:
        """Get experiment parameters"""
        return cast(dict[str, Any], self.metadata["parameters"])

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed"""
        return step_name in self.metadata["steps_completed"]

    def get_stats(self) -> dict[str, Any]:
        """Get experiment statistics"""
        return cast(dict[str, Any], self.metadata["stats"])

    def print_status(self) -> None:
        """Print detailed experiment status"""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.experiment_id}")
        print(f"{'=' * 60}")
        print(f"\nCreated: {self.metadata['created_at']}")

        print("\nParameters:")
        for key, value in self.metadata["parameters"].items():
            print(f"  {key}: {value}")

        print("\nSteps Completed:")
        for step in ["context", "preprocess", "labeling", "process_batches"]:
            status = "✅" if step in self.metadata["steps_completed"] else "⏸️"
            print(f"  {status} {step}")

        if self.metadata.get("batch_info"):
            batch_status = self.metadata["batch_info"]["status"]
            print(f"\nBatch Status: {batch_status}")
            if batch_status != "submitted":
                batch_info = self.get_batch_info()
                if batch_info:
                    print(
                        f"  Completed: {batch_info['completed_requests']}/{batch_info['total_requests']}"
                    )
                    if batch_info["cost"]["total_cost"]:
                        print(f"  Cost: ${batch_info['cost']['total_cost']:.4f}")

        print("\nStatistics:")
        for key, value in self.metadata["stats"].items():
            if value is not None:
                print(f"  {key}: {value:,}")

        print(f"\nDirectory: {self.experiment_dir}")
        print(f"{'=' * 60}\n")
