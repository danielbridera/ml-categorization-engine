"""
ICSAT — Inferred Customer Satisfaction (NPS-style classifier).

Classifies each conversation as PROMOTER / PASSIVE / DETRACTOR, then
computes a batch-level iCSAT score = (promoters − detractors) / total × 100.
"""

from __future__ import annotations

import pandas as pd

from categorization.prompts._base import ClassifierSpec
from categorization.settings.log import logger


def compute_icsat_score(df: pd.DataFrame, output_column: str) -> None:
    """Log the iCSAT score for the batch.

    Promoters/detractors counted case-insensitively against `output_column`.
    Skipped silently on empty input.
    """
    total = len(df)
    if total == 0:
        return
    normalized = df[output_column].str.upper()
    promoters = int((normalized == "PROMOTER").sum())
    detractors = int((normalized == "DETRACTOR").sum())
    score = round((promoters - detractors) * 100 / total, 1)
    logger.info(
        f"iCSAT score: {score:+.1f}",
        details=f"promoters={promoters}, detractors={detractors}, total={total}",
    )


_SYSTEM_PROMPT = """You are a customer satisfaction classifier. Respond only with a JSON object.

Classify the conversation as exactly one of:

- PROMOTER:  User completed a transactional goal OR explicitly expressed
             satisfaction. Simply receiving information without reacting is
             NOT enough.
- PASSIVE:   User browsed, asked questions, or received info without clear
             satisfaction or dissatisfaction. Info delivered correctly but
             user just moved on.
- DETRACTOR: Explicit bad experience — user complained, bot looped, user
             answered a survey negatively, or the bot failed with a technical
             error.

OUTPUT FORMAT

Respond ONLY with a JSON object matching exactly this shape:

{"nps_category": "PROMOTER" | "PASSIVE" | "DETRACTOR", "reason": "<brief justification>"}"""


CLASSIFIER = ClassifierSpec(
    name="ICSAT",
    family="conversations",
    output_column="nps_category",
    system_prompt=_SYSTEM_PROMPT,
    user_prompt_template="Conversation:\n{message}",
    post_process=compute_icsat_score,
)
