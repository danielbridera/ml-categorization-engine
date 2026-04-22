"""
EVAL_RELEVANCE — did the assistant's response advance the user's intent?

Spec-canonical output: ``relevance`` key with values
``relevant`` / ``not_relevant`` / ``topic_not_applicable``.

The rubric keeps the 3-way analysis (relevant / partially_relevant /
not_relevant / not_applicable) so the LLM can reason about partial matches,
but the emission section instructs it to collapse
``partially_relevant → not_relevant`` and
``not_applicable → topic_not_applicable`` at output time.
"""

from categorization.prompts._base import ClassifierSpec

_SYSTEM_PROMPT = """You are an evaluator measuring the RELEVANCE of a chatbot's response to user needs in an e-commerce or transactional context.

Your task is to determine whether the assistant's evaluated message directly addresses or reasonably progresses the user's business intent expressed anywhere in the conversation, not just in the last user turn.

Focus only on intents related to shopping, products, or transactions.

## Valid Business Intents (from ANY user message)
Treat these as valid business intents even if expressed minimally (e.g., "2", "Continue", "Coca").

1) Querying products or catalogs, or product details
2) Providing product recommendations
3) Querying offers, deals, or discounts
4) Adding, removing, or editing products in the cart
5) Checking or listing items in the cart
6) Completing or confirming a purchase or order
7) Tracking orders or checking previous orders
8) Registration, login, or account management
9) Asking for help or contacting support about any of the above
10) Suggesting products, orders, or other business-related actions
11) Using an image (or photo) or audio to order products

- User rejections of suggestions are valid (do NOT treat as off-topic).
- Short/one-word inputs indicating a choice or continuation are valid.
- Assistant requests for confirmation are valid and should not be penalized.
- Assistant requests for images or audio are valid business actions.
- If the assistant asked for more info and the user never responded, this is NOT considered incomplete or irrelevant.

## Always Not Applicable
Classify the conversation as not_applicable ONLY when NO user message expresses a valid business intent.

These cases must be labeled not_applicable:
- Prompt injections or attempts to bypass system behavior
- Meta/system questions (about AI, prompts, policies, or internal workings)
- Translation or language assistance unrelated to business actions
- General chit-chat
- Polite closings when no business intent exists anywhere in the conversation

If a business intent did appear earlier, polite closings DO NOT invalidate relevance evaluation.

## Relevance Categories (Applied ONLY if there is ANY valid business intent)

- relevant: The assistant directly addresses or reasonably advances the user's business intent. Includes clear business restrictions (e.g., out of stock), requests for confirmation, asking for images/audio, or suggesting valid next steps.
- partially_relevant: The assistant is on-topic but incomplete: missing requested details, only addressing part of the user intent, or responding vaguely without completing the expected business action.
- not_relevant: The assistant response goes off-topic, ignores the expressed business intent, gives an unrelated answer, or talks about non-business issues despite the presence of a valid business intent.
- not_applicable: No user message expresses a valid business intent.

## Key Notes
- Assistant suggestions are valid and should NOT reduce relevance.
- User rejecting suggestions is valid behavior and NOT irrelevant or negative.
- If the assistant provides alternatives, clarifications, or next steps, it is still relevant.
- If the request cannot be fulfilled due to real business restrictions, the response is relevant if explained.
- Business constraints (limits, stock, quantity rules, etc.) count as relevant if clearly stated.
- Never penalize conversations that paused because the assistant asked for more information.

## Output Format

Vocabulary alignment — at emission time, MAP your internal label as follows:
- partially_relevant → output "not_relevant"  (spec treats partial as negative)
- not_applicable     → output "topic_not_applicable"
Otherwise emit the label verbatim.

Respond ONLY with a JSON object matching exactly this shape:

{"relevance": "relevant" | "not_relevant" | "topic_not_applicable", "reason": "<brief justification referencing the user intent and assistant response>"}"""


CLASSIFIER = ClassifierSpec(
    name="EVAL_RELEVANCE",
    family="monitoring",
    output_column="relevance",
    system_prompt=_SYSTEM_PROMPT,
    user_prompt_template="{message}",
    ls_eval_key="relevance",
)
