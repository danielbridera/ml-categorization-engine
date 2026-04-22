"""
Extraction contexts — named configurations for the first pipeline step.

An ``ExtractionContext`` bundles everything ``make_context`` needs to decide:

- which SQL query template to load
- whether the downstream unit is per-message or per-conversation
- whether to apply the ``has_oris = TRUE`` filter by default
- what classifier family is expected (prevents running ORIS evals on
  non-ORIS-filtered data)

Two canonical contexts are defined here:

- ``MONITORING``    — per-conversation, ``has_oris=TRUE`` by default,
  paired with ``monitoring`` family classifiers (the ORIS eval suite).
- ``CONVERSATIONS_V2`` — per-conversation, no ``has_oris`` filter,
  paired with ``conversations`` family classifiers.

(The ``_V2`` suffix is temporary. The legacy ``CONVERSATIONS`` entry still
lives in ``pipelines.classifiers.EXTRACTION_CONTEXTS`` pointing at
``v_conversations``; once the per-conversation classifier migration lands
we'll retire that and drop the ``_V2``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["monitoring", "conversations", "message"]
Granularity = Literal["conversation", "turn", "message"]


@dataclass(frozen=True)
class ExtractionContext:
    """Parameter bundle for one pipeline context."""

    name: str
    sql_query_path: str
    unit: Literal["message", "conversation"]
    granularity: Granularity
    classifier_family: Family
    has_oris_default: bool
    turn_filter_step_name: str | None = None
    description: str = ""


# Canonical contexts. Both default to per-conversation granularity — per-turn
# is still supported at the preprocess layer but is not the default path here.
CONTEXTS: dict[str, ExtractionContext] = {
    "MONITORING": ExtractionContext(
        name="MONITORING",
        sql_query_path="sql/context/mart_conversations.sql",
        unit="conversation",
        granularity="conversation",
        classifier_family="monitoring",
        has_oris_default=True,
        turn_filter_step_name=None,
        description=(
            "ORIS monitoring evals over full conversations. "
            "Defaults to has_oris=TRUE; override with --no-has-oris if needed."
        ),
    ),
    "CONVERSATIONS_V2": ExtractionContext(
        name="CONVERSATIONS_V2",
        sql_query_path="sql/context/mart_conversations.sql",
        unit="conversation",
        granularity="conversation",
        classifier_family="conversations",
        has_oris_default=False,
        turn_filter_step_name=None,
        description=(
            "General conversation analysis. No has_oris filter by default; "
            "override with --has-oris if you want ORIS-only conversations."
        ),
    ),
}


def get_context(name: str) -> ExtractionContext | None:
    """Return the named context, or ``None`` if not registered."""
    return CONTEXTS.get(name)


def list_contexts() -> list[str]:
    """Sorted list of registered context names."""
    return sorted(CONTEXTS.keys())
