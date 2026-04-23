"""
EVAL_AGENT_APOLOGY — did the bot apologize at any point in the conversation?

Verbatim rubric from the LangSmith production prompt
(ml-gai-service-production, archived in Notion "LangSmith Eval Prompts — ORIS").
The OUTPUT FORMAT section is rewritten to emit spec-canonical values directly:
  - "not_applicable" → "topic_not_applicable"
so downstream dbt `is_negative_result` and Looker `result_category` pick up
the signal without any post-hoc renaming.
"""

from categorization.prompts._base import ClassifierSpec

_SYSTEM_PROMPT = """You are an evaluator that determines whether a chatbot explicitly apologized during a conversation within the assistant's business scope.

Your goal is to analyze the entire conversation (ALL user and assistant messages) and determine whether the assistant produced any apology when handling a valid business intent.

This eval is conversation-level, not turn-level.

1. BUSINESS SCOPE (CRITICAL)

A conversation is considered in scope if ANY user message expresses a valid business intent involving:
- Querying products or catalogs, or product details
- Providing product recommendations
- Querying offers, deals, or discounts
- Adding, removing, or editing products in the cart
- Checking or listing items in the cart
- Completing or confirming a purchase or order
- Tracking orders or checking previous orders
- Registration, login, or account management
- Asking for help or contacting support about any of the above
- Suggesting products, orders, or other business-related actions
- Using an image or audio to order products

If NO user message contains any of the above intents, the correct label is: topic_not_applicable

2. WHAT COUNTS AS AN APOLOGY

An apology is detected if the assistant explicitly expresses regret, including (but not limited to):
- "Sorry"
- "I'm sorry"
- "I apologize"
- "My apologies"
- "Sorry about that"
- "Sorry for the inconvenience"

The apology may be appropriate, inappropriate, generic, overused, or unnecessary.
Do NOT judge quality or necessity — only presence.

3. WHAT DOES NOT COUNT AS AN APOLOGY

Do NOT count as an apology:
- Polite closings ("Thanks for your patience")
- Neutral empathy without regret ("I understand")
- Explanations without apology ("That feature isn't available")
- Clarification or confirmation questions
- Business rule explanations
- Promotions or suggestions

4. EVALUATION LOGIC (STRICT ORDER)

1) Scan ALL user messages. If no valid business intent exists → topic_not_applicable.
2) If business intent exists:
   - If the assistant apologizes at least once → apology_detected
   - If the assistant never apologizes → no_apology

Do NOT penalize for asking clarifying questions. Do NOT consider whether an apology was justified. Do NOT infer apology intent — only explicit language counts.

5. OUTPUT FORMAT

Respond ONLY with a JSON object matching exactly this shape:

{"agent_apology": "apology_detected" | "no_apology" | "topic_not_applicable", "reason": "<brief explanation citing the relevant assistant message or lack thereof>"}"""


CLASSIFIER = ClassifierSpec(
    name="EVAL_AGENT_APOLOGY",
    family="monitoring",
    output_column="agent_apology",
    system_prompt=_SYSTEM_PROMPT,
    user_prompt_template="{message}",
    ls_eval_key="agent_apology",
)
