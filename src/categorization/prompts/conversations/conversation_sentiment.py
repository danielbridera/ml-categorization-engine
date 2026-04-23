"""
CONVERSATION_SENTIMENT — overall sentiment of a full conversation.

Migrated from the legacy CLASSIFIERS dict; now emits JSON with a reason
alongside the label, matching the unified label+reason contract.
"""

from categorization.prompts._base import ClassifierSpec

_SYSTEM_PROMPT = """You are a sentiment analysis expert. Respond only with a JSON object.

Analyze the sentiment of the following conversation and classify it as one of:
- positive: Expresses satisfaction, gratitude, happiness, or approval
- negative: Expresses frustration, anger, disappointment, or complaint
- neutral:  Factual, informational, or no clear emotional tone
- mixed:    Contains both positive and negative sentiments

OUTPUT FORMAT

Respond ONLY with a JSON object matching exactly this shape:

{"sentiment": "positive" | "negative" | "neutral" | "mixed", "reason": "<brief justification>"}"""


CLASSIFIER = ClassifierSpec(
    name="CONVERSATION_SENTIMENT",
    family="conversations",
    output_column="sentiment",
    system_prompt=_SYSTEM_PROMPT,
    user_prompt_template="Conversation:\n{message}",
)
