"""
Build the per-account `intent_taxonomy_<slug>.json` deliverable from a re-clustered experiment.

Cheap (no LLM calls). Reads `labeled_with_final_action.parquet`,
`canonical_vocabulary_<workflow>.json`, and `metadata.json`, then emits one
deliverable per workflow / account slug.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from categorization.pipelines.intent_taxonomy_builder import (
    DEFAULT_MODEL,
    build_taxonomy_json,
)
from categorization.settings.log import logger
from categorization.utils.metadata import ExperimentMetadata

RECLUSTER_PARQUET_FILENAME = "labeled_with_final_action.parquet"


def make_intent_taxonomy_json(
    experiment_id: str | None = None,
    context_name: str | None = None,
    workflow_names: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    """Emit `intent_taxonomy_<slug>.json` for each workflow with a discovered vocab."""
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id,
        context_name=context_name,
        workflow_names=workflow_names,
    )
    experiment_dir: Path = metadata.experiment_dir

    parquet_path = experiment_dir / RECLUSTER_PARQUET_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Re-clustered parquet not found: {parquet_path}. "
            f"Run `categorization make_intent_recluster` first."
        )

    df = pd.read_parquet(parquet_path)

    vocab_files = sorted(experiment_dir.glob("canonical_vocabulary_*.json"))
    if not vocab_files:
        raise FileNotFoundError(
            f"No canonical_vocabulary_*.json files in {experiment_dir}. "
            f"Run `categorization make_intent_recluster` first."
        )

    written: list[Path] = []
    for vocab_path in vocab_files:
        vocab = json.loads(vocab_path.read_text())
        workflow_name = vocab.get("workflow_name")
        if not workflow_name:
            logger.warning(
                f"Skipping {vocab_path.name}: missing `workflow_name` key in vocab."
            )
            continue
        if workflow_name not in df["workflow_name"].unique():
            logger.warning(
                f"Skipping {vocab_path.name}: workflow `{workflow_name}` "
                f"absent from re-clustered parquet."
            )
            continue

        out_path = build_taxonomy_json(
            experiment_dir=experiment_dir,
            workflow_name=workflow_name,
            vocab=vocab,
            df=df,
            model=model,
        )
        wf_total = int((df["workflow_name"] == workflow_name).sum())
        logger.info(
            f"[{workflow_name}] wrote {out_path.name} ({wf_total} conversations)"
        )
        written.append(out_path)

    metadata.update_step(
        "intent_taxonomy_json",
        stats={"intent_taxonomy_json_files": len(written)},
    )

    return written
