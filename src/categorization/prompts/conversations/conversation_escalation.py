"""
CONVERSATION_ESCALATION — was the conversation escalated to a human agent?

Migrated from the legacy CLASSIFIERS dict; emits label + reason JSON.
"""

from categorization.prompts._base import ClassifierSpec

_SYSTEM_PROMPT = """You are a customer support quality analyst. Respond only with a JSON object.

Determine whether this conversation involved escalation to a human agent:
- escalated:      The user asked for a human agent, expressed frustration with
                   the bot and requested escalation, OR the bot explicitly handed
                   the conversation off to a human.
- not_escalated:  The conversation was handled entirely by the bot with no
                   request for (or handoff to) a human agent.

OUTPUT FORMAT

Respond ONLY with a JSON object matching exactly this shape:

{"escalated_to_human": "escalated" | "not_escalated", "reason": "<brief justification>"}"""


CLASSIFIER = ClassifierSpec(
    name="CONVERSATION_ESCALATION",
    family="conversations",
    output_column="escalated_to_human",
    system_prompt=_SYSTEM_PROMPT,
    user_prompt_template="Conversation:\n{message}",
)
