"""
Re-cluster freeform user-intent labels into a per-workflow canonical action vocabulary.

Two LLM passes per workflow (vocabulary discovery + chunked label mapping),
then a coverage cutoff. Writes four artifacts into the experiment dir:
  - canonical_vocabulary_<workflow>.json (one per workflow)
  - taxonomy_mapping_action.json
  - intent_frequencies_action.csv
  - labeled_with_final_action.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openai import OpenAI

from categorization.pipelines.intent_taxonomy_builder import (
    DEFAULT_MODEL,
    TOP_N_FOR_DISCOVERY,
    apply_coverage_cutoff,
    discover_vocabulary,
    map_freeform_to_canonical,
    write_recluster_artifacts,
)
from categorization.settings.credentials import OPENAI_API_KEY
from categorization.settings.log import logger
from categorization.utils.metadata import ExperimentMetadata

FREEFORM_PARQUET_FILENAME = "user_intent_freeform_labeled.parquet"
INTENT_COLUMN = "primary_intent"


def make_intent_recluster(
    experiment_id: str | None = None,
    context_name: str | None = None,
    workflow_names: str | None = None,
    top_n_for_discovery: int = TOP_N_FOR_DISCOVERY,
    model: str = DEFAULT_MODEL,
) -> dict[str, Path]:
    """Run vocabulary discovery + label mapping for every workflow in the experiment."""
    metadata = ExperimentMetadata.load(
        experiment_id=experiment_id,
        context_name=context_name,
        workflow_names=workflow_names,
    )
    experiment_dir: Path = metadata.experiment_dir

    freeform_path = experiment_dir / FREEFORM_PARQUET_FILENAME
    if not freeform_path.exists():
        raise FileNotFoundError(
            f"Freeform-labeled parquet not found: {freeform_path}. "
            f"Run `categorization make_label --classifier USER_INTENT_FREEFORM` first."
        )

    df = pd.read_parquet(freeform_path)
    logger.info(
        f"Loaded {len(df):,} labeled conversations "
        f"({df['workflow_name'].nunique()} workflows; "
        f"{df[INTENT_COLUMN].nunique()} unique raw intents)"
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    df["primary_intent_canonical"] = df[INTENT_COLUMN]
    mapping_per_workflow: dict[str, dict[str, str]] = {}
    vocab_per_workflow: dict[str, dict] = {}

    for workflow, group in df.groupby("workflow_name"):
        counts = group[INTENT_COLUMN].value_counts()
        logger.info(
            f"[{workflow}] {len(counts)} raw labels — running Pass A (vocab discovery)…"
        )
        vocab = discover_vocabulary(
            client,
            workflow_name=str(workflow),
            freeform_counts=counts,
            top_n=top_n_for_discovery,
            model=model,
        )
        logger.info(
            f"[{workflow}] discovered {len(vocab['actions'])} actions "
            f"(slug: {vocab['account_slug']}) — running Pass B (label mapping)…"
        )
        mapping = map_freeform_to_canonical(
            client,
            workflow_name=str(workflow),
            vocab=vocab,
            freeform_counts=counts,
            model=model,
        )

        vocab_per_workflow[str(workflow)] = vocab
        mapping_per_workflow[str(workflow)] = mapping

        mask = df["workflow_name"] == workflow
        df.loc[mask, "primary_intent_canonical"] = df.loc[mask, INTENT_COLUMN].map(
            mapping
        )
        canon_n = df.loc[mask, "primary_intent_canonical"].nunique()
        logger.info(f"[{workflow}] {len(counts)} raw → {canon_n} canonical")

    apply_coverage_cutoff(df)

    paths = write_recluster_artifacts(
        experiment_dir=experiment_dir,
        df=df,
        mapping_per_workflow=mapping_per_workflow,
        vocab_per_workflow=vocab_per_workflow,
    )

    for workflow in df["workflow_name"].unique():
        wf_df = df[df["workflow_name"] == workflow]
        total = len(wf_df)
        head = wf_df["primary_intent_final"].value_counts()
        general_pct = (wf_df["primary_intent_final"] == "user_intent_general").mean()
        logger.info(
            f"[{workflow}] total={total} | "
            f"head_labels={(head.index != 'user_intent_general').sum()} | "
            f"user_intent_general={general_pct:.1%}"
        )

    metadata.update_step(
        "intent_recluster",
        stats={
            "intent_recluster_workflows": int(df["workflow_name"].nunique()),
            "intent_recluster_conversations": int(len(df)),
            "intent_recluster_raw_labels": int(df[INTENT_COLUMN].nunique()),
        },
    )

    return paths
