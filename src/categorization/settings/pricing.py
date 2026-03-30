"""
Model Pricing Configuration

Cost per 1M tokens in USD. Used for estimating labeling costs before submission.
Batch API applies a 50% discount on top of these rates.

Update these when OpenAI changes pricing.
Source: https://openai.com/pricing
"""

# Per-million token rates (input, output) in USD
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.500, "output": 10.000},
    "gpt-4o-2024-11-20": {"input": 2.500, "output": 10.000},
    "gpt-4-turbo": {"input": 10.000, "output": 30.000},
    "gpt-3.5-turbo": {"input": 0.500, "output": 1.500},
}

BATCH_DISCOUNT = 0.5  # OpenAI Batch API is 50% cheaper than real-time


def get_model_pricing(model: str) -> dict[str, float] | None:
    """
    Return pricing for a model, or None if unknown.

    Args:
        model: OpenAI model name (e.g. "gpt-4o-mini")

    Returns:
        Dict with "input" and "output" keys ($ per 1M tokens), or None
    """
    return PRICING.get(model)


def estimate_cost(
    n_requests: int,
    model: str,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 10,
    batch: bool = False,
) -> float | None:
    """
    Estimate total labeling cost.

    Args:
        n_requests: Number of messages to label
        model: OpenAI model name
        avg_input_tokens: Estimated tokens per request (prompt + message)
        avg_output_tokens: Estimated tokens per response (label)
        batch: Whether to apply batch API discount

    Returns:
        Estimated cost in USD, or None if model pricing is unknown
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_cost = (n_requests * avg_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (n_requests * avg_output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + output_cost

    if batch:
        total *= BATCH_DISCOUNT

    return total
