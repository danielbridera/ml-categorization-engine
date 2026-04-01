"""
Tests for ExperimentMetadata (utils/metadata.py).

All tests use real temp directories via tmp_path — no file I/O is mocked.

Covers:
- create_new: directory creation, metadata.json contents, experiment_id format
- __init__: raises when metadata.json missing
- load: by explicit ID, autodiscovery (latest), context filtering
- _find_latest_experiment: workflow_names matching
- update_step / is_step_completed / get_stats
- save_batch_info / get_batch_info / update_batch_status
- get_file_path naming conventions
- update_file tracking
- list_experiments
"""

from pathlib import Path

import pytest

from categorization.utils.metadata import ExperimentMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_experiment(tmp_path: Path, context: str = "SENTIMENT", **kwargs) -> ExperimentMetadata:
    """Create a new experiment in tmp_path and return its ExperimentMetadata."""
    defaults = dict(
        start_date="2024-01-01",
        end_date="2024-01-31",
        n_samples=100,
        min_length=6,
        context_name=context,
        base_dir=str(tmp_path),
    )
    defaults.update(kwargs)
    return ExperimentMetadata.create_new(**defaults)


# ---------------------------------------------------------------------------
# create_new
# ---------------------------------------------------------------------------


class TestCreateNew:
    def test_creates_directory(self, tmp_path):
        meta = make_experiment(tmp_path)
        assert meta.experiment_dir.exists()
        assert meta.experiment_dir.is_dir()

    def test_metadata_json_created(self, tmp_path):
        meta = make_experiment(tmp_path)
        assert meta.metadata_file.exists()

    def test_experiment_id_uses_context_prefix(self, tmp_path):
        meta = make_experiment(tmp_path, context="SENTIMENT")
        assert meta.experiment_id.startswith("sentiment_")

    def test_experiment_id_lowercase(self, tmp_path):
        meta = make_experiment(tmp_path, context="ESCALATION")
        assert meta.experiment_id.startswith("escalation_")

    def test_metadata_contains_required_keys(self, tmp_path):
        meta = make_experiment(tmp_path)
        required = {"experiment_id", "context_name", "created_at", "parameters", "steps_completed", "files", "stats"}
        assert required.issubset(meta.metadata.keys())

    def test_parameters_stored(self, tmp_path):
        meta = make_experiment(tmp_path, n_samples=500, min_length=10)
        params = meta.get_parameters()
        assert params["n_samples"] == 500
        assert params["min_length"] == 10
        assert params["start_date"] == "2024-01-01"

    def test_steps_completed_empty_on_creation(self, tmp_path):
        meta = make_experiment(tmp_path)
        assert meta.metadata["steps_completed"] == []

    def test_unit_stored_in_parameters(self, tmp_path):
        meta = make_experiment(tmp_path, unit="conversation")
        assert meta.get_parameters()["unit"] == "conversation"

    def test_workflow_names_stored_when_provided(self, tmp_path):
        meta = make_experiment(tmp_path, workflow_names="bot1,bot2")
        assert meta.get_parameters()["workflow_names"] == "bot1,bot2"

    def test_files_initialized_to_none(self, tmp_path):
        meta = make_experiment(tmp_path, context="SENTIMENT")
        files = meta.metadata["files"]
        # All file entries should start as None
        assert all(v is None for v in files.values())

    def test_stats_initialized_to_none(self, tmp_path):
        meta = make_experiment(tmp_path, context="SENTIMENT")
        stats = meta.metadata["stats"]
        assert all(v is None for v in stats.values())


# ---------------------------------------------------------------------------
# __init__ (direct instantiation)
# ---------------------------------------------------------------------------


class TestInit:
    def test_raises_when_metadata_file_missing(self, tmp_path):
        empty_dir = tmp_path / "no_metadata"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            ExperimentMetadata(empty_dir)

    def test_loads_experiment_id_from_file(self, tmp_path):
        meta = make_experiment(tmp_path)
        reloaded = ExperimentMetadata(meta.experiment_dir)
        assert reloaded.experiment_id == meta.experiment_id


# ---------------------------------------------------------------------------
# load (autodiscovery)
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_by_explicit_id(self, tmp_path):
        meta = make_experiment(tmp_path)
        loaded = ExperimentMetadata.load(
            experiment_id=meta.experiment_id, base_dir=str(tmp_path)
        )
        assert loaded.experiment_id == meta.experiment_id

    def test_load_raises_for_unknown_id(self, tmp_path):
        with pytest.raises(ValueError, match="Experiment not found"):
            ExperimentMetadata.load(
                experiment_id="nonexistent_123", base_dir=str(tmp_path)
            )

    def test_load_raises_when_no_experiments(self, tmp_path):
        empty_base = tmp_path / "empty"
        empty_base.mkdir()
        with pytest.raises(ValueError, match="No experiments found"):
            ExperimentMetadata.load(base_dir=str(empty_base))

    def test_load_latest_without_filter(self, tmp_path):
        import time
        _ = make_experiment(tmp_path, context="SENTIMENT")
        time.sleep(1.1)  # ensure distinct second-level timestamps
        meta2 = make_experiment(tmp_path, context="SENTIMENT")
        loaded = ExperimentMetadata.load(base_dir=str(tmp_path))
        # Should be the alphabetically last (latest timestamp)
        assert loaded.experiment_id == meta2.experiment_id

    def test_load_filtered_by_context_name(self, tmp_path):
        sent = make_experiment(tmp_path, context="SENTIMENT")
        _ = make_experiment(tmp_path, context="ESCALATION")
        loaded = ExperimentMetadata.load(
            context_name="SENTIMENT", base_dir=str(tmp_path)
        )
        assert loaded.experiment_id == sent.experiment_id

    def test_load_by_workflow_names_match(self, tmp_path):
        meta_a = make_experiment(tmp_path, context="SENTIMENT", workflow_names="botA")
        _ = make_experiment(tmp_path, context="SENTIMENT", workflow_names="botB")
        loaded = ExperimentMetadata.load(
            context_name="SENTIMENT",
            workflow_names="botA",
            base_dir=str(tmp_path),
        )
        assert loaded.experiment_id == meta_a.experiment_id

    def test_load_workflow_names_order_independent(self, tmp_path):
        """workflow_names matching should be order-independent (sorted comparison)."""
        meta = make_experiment(tmp_path, context="SENTIMENT", workflow_names="botB,botA")
        loaded = ExperimentMetadata.load(
            context_name="SENTIMENT",
            workflow_names="botA,botB",
            base_dir=str(tmp_path),
        )
        assert loaded.experiment_id == meta.experiment_id


# ---------------------------------------------------------------------------
# update_step / is_step_completed / get_stats
# ---------------------------------------------------------------------------


class TestUpdateStep:
    def test_marks_step_as_completed(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.update_step("context", {"messages_extracted": 500})
        assert meta.is_step_completed("context")

    def test_updates_stats(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.update_step("context", {"messages_extracted": 42})
        assert meta.get_stats()["messages_extracted"] == 42

    def test_step_not_duplicated_on_rerun(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.update_step("context", {})
        meta.update_step("context", {})
        assert meta.metadata["steps_completed"].count("context") == 1

    def test_persists_to_disk(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.update_step("preprocess", {"messages_after_preprocess": 300})
        reloaded = ExperimentMetadata(meta.experiment_dir)
        assert reloaded.is_step_completed("preprocess")
        assert reloaded.get_stats()["messages_after_preprocess"] == 300

    def test_is_step_completed_false_before_update(self, tmp_path):
        meta = make_experiment(tmp_path)
        assert not meta.is_step_completed("context")


# ---------------------------------------------------------------------------
# save_batch_info / get_batch_info / update_batch_status
# ---------------------------------------------------------------------------


class TestBatchInfo:
    def test_save_batch_info_creates_file(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.save_batch_info(
            batch_id="batch_abc123",
            file_id="file_xyz456",
            total_requests=200,
        )
        assert meta.batch_info_file.exists()

    def test_get_batch_info_returns_correct_data(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.save_batch_info(
            batch_id="batch_abc123",
            file_id="file_xyz456",
            total_requests=200,
            extraction_context="CONVERSATIONS",
            classifier="CONVERSATION_SENTIMENT",
        )
        info = meta.get_batch_info()
        assert info is not None
        assert info["batch_id"] == "batch_abc123"
        assert info["file_id"] == "file_xyz456"
        assert info["total_requests"] == 200
        assert info["status"] == "submitted"
        assert info["extraction_context"] == "CONVERSATIONS"
        assert info["classifier"] == "CONVERSATION_SENTIMENT"

    def test_get_batch_info_returns_none_when_no_file(self, tmp_path):
        meta = make_experiment(tmp_path)
        assert meta.get_batch_info() is None

    def test_metadata_batch_info_populated_after_save(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.save_batch_info("bid", "fid", 50)
        assert meta.metadata["batch_info"] is not None
        assert meta.metadata["batch_info"]["batch_id"] == "bid"
        assert meta.metadata["batch_info"]["status"] == "submitted"

    def test_update_batch_status(self, tmp_path):
        meta = make_experiment(tmp_path)
        meta.save_batch_info("bid", "fid", 100)
        meta.update_batch_status(
            status="completed",
            completed_requests=100,
            failed_requests=0,
        )
        info = meta.get_batch_info()
        assert info["status"] == "completed"
        assert info["completed_requests"] == 100
        assert info["completed_at"] is not None

    def test_update_batch_status_raises_without_batch_info(self, tmp_path):
        meta = make_experiment(tmp_path)
        with pytest.raises(ValueError, match="No batch info found"):
            meta.update_batch_status("completed")


# ---------------------------------------------------------------------------
# get_file_path
# ---------------------------------------------------------------------------


class TestGetFilePath:
    def test_context_file_uses_context_prefix(self, tmp_path):
        meta = make_experiment(tmp_path, context="SENTIMENT")
        path = meta.get_file_path("context")
        assert path.name == "sentiment_context.parquet"

    def test_preprocess_file_uses_context_prefix(self, tmp_path):
        meta = make_experiment(tmp_path, context="ESCALATION")
        path = meta.get_file_path("preprocess")
        assert path.name == "escalation_preprocessed.parquet"

    def test_labeled_file_uses_context_prefix(self, tmp_path):
        meta = make_experiment(tmp_path, context="FEEDBACK")
        path = meta.get_file_path("labeled")
        assert path.name == "feedback_labeled.parquet"

    def test_batch_requests_file_name(self, tmp_path):
        meta = make_experiment(tmp_path)
        path = meta.get_file_path("batch_requests")
        assert path.name == "batch_requests.jsonl"

    def test_raises_for_unknown_file_type(self, tmp_path):
        meta = make_experiment(tmp_path)
        with pytest.raises(ValueError, match="Unknown file type"):
            meta.get_file_path("nonexistent")

    def test_path_is_inside_experiment_dir(self, tmp_path):
        meta = make_experiment(tmp_path, context="SENTIMENT")
        path = meta.get_file_path("context")
        assert path.parent == meta.experiment_dir


# ---------------------------------------------------------------------------
# list_experiments
# ---------------------------------------------------------------------------


class TestListExperiments:
    def test_empty_base_dir_returns_empty_list(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert ExperimentMetadata.list_experiments(base_dir=str(empty)) == []

    def test_returns_all_experiments(self, tmp_path):
        make_experiment(tmp_path, context="SENTIMENT")
        make_experiment(tmp_path, context="ESCALATION")
        result = ExperimentMetadata.list_experiments(base_dir=str(tmp_path))
        assert len(result) == 2

    def test_filters_by_context_name(self, tmp_path):
        make_experiment(tmp_path, context="SENTIMENT")
        make_experiment(tmp_path, context="ESCALATION")
        result = ExperimentMetadata.list_experiments(
            base_dir=str(tmp_path), context_name="SENTIMENT"
        )
        assert len(result) == 1
        assert result[0]["experiment_id"].startswith("sentiment_")

    def test_experiment_info_structure(self, tmp_path):
        make_experiment(tmp_path)
        result = ExperimentMetadata.list_experiments(base_dir=str(tmp_path))
        keys = {"experiment_id", "created_at", "steps_completed", "batch_status", "parameters"}
        assert keys.issubset(result[0].keys())
