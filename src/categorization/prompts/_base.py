"""
ClassifierSpec — canonical declaration of one classifier.

Each classifier file under `src/categorization/prompts/<family>/<name>.py`
exports a module-level ``CLASSIFIER`` of this type. The package's ``__init__``
auto-discovers and registers them into the global ``CLASSIFIERS`` registry at
import time, so adding a classifier means dropping one file — no edits to
``classifiers.py``.

Why a dataclass (not a TypedDict) here: gives ergonomic defaults, validation,
and a single ``to_config()`` conversion to the legacy ``ClassifierConfig``
TypedDict that the pipeline consumes. Downstream keeps reading the TypedDict
so ``make_label`` / ``make_process_batches`` don't need to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Family = Literal["monitoring", "conversations", "message"]


# Family defaults — applied when a field is left unset on the spec.
_DEFAULTS_BY_FAMILY: dict[Family, dict[str, Any]] = {
    "monitoring": {
        "unit": "conversation",
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_tokens": 300,
        "parse_json_response": True,
    },
    "conversations": {
        "unit": "conversation",
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_tokens": 300,
        "parse_json_response": True,
    },
    "message": {
        "unit": "message",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 10,
        "parse_json_response": False,
    },
}


@dataclass(frozen=True)
class ClassifierSpec:
    """Declarative classifier definition.

    Required fields:
        name: Globally unique identifier (also the CLI ``--classifier`` value)
        family: "monitoring" | "conversations" | "message"
        output_column: Column name that will hold the label
        system_prompt: System instruction for the LLM
        user_prompt_template: User prompt template — must contain ``{message}``

    Optional (family defaults apply if unset):
        model, temperature, max_tokens, parse_json_response, unit
        post_process: Callable(df_clean, output_column) -> None — runs after
            labels are parsed; use for derived metrics (e.g. iCSAT score).
        ls_eval_key: LangSmith-shaped feedback key (for monitoring evals);
            used by the writer's EVAL_KEY_MAP.
    """

    name: str
    family: Family
    output_column: str
    system_prompt: str
    user_prompt_template: str = "{message}"

    # Optional — fall back to family defaults when None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    parse_json_response: bool | None = None
    unit: Literal["message", "conversation"] | None = None

    post_process: Callable[[pd.DataFrame, str], None] | None = None
    ls_eval_key: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if "{message}" not in self.user_prompt_template:
            raise ValueError(
                f"ClassifierSpec {self.name!r}: user_prompt_template must "
                "contain the '{message}' placeholder"
            )
        if self.family not in _DEFAULTS_BY_FAMILY:
            raise ValueError(
                f"ClassifierSpec {self.name!r}: unknown family {self.family!r}"
            )

    def to_config(self) -> dict[str, Any]:
        """Materialize a ClassifierConfig-shaped dict for the legacy registry.

        Backfills any unset optional field from ``_DEFAULTS_BY_FAMILY``.
        """
        defaults = _DEFAULTS_BY_FAMILY[self.family]
        cfg: dict[str, Any] = {
            "context_name": self.name,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "output_column": self.output_column,
            "model": self.model if self.model is not None else defaults["model"],
            "temperature": self.temperature
            if self.temperature is not None
            else defaults["temperature"],
            "max_tokens": self.max_tokens
            if self.max_tokens is not None
            else defaults["max_tokens"],
            "parse_json_response": self.parse_json_response
            if self.parse_json_response is not None
            else defaults["parse_json_response"],
            "unit": self.unit if self.unit is not None else defaults["unit"],
            # Track provenance so downstream can do family-compatibility checks
            "family": self.family,
        }
        if self.post_process is not None:
            cfg["post_process"] = self.post_process
        if self.ls_eval_key is not None:
            cfg["ls_eval_key"] = self.ls_eval_key
        if self.tags:
            cfg["tags"] = tuple(self.tags)
        return cfg
