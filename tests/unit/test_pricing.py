"""
Tests for model pricing utilities (settings/pricing.py).

Covers:
- PRICING dict has expected models
- get_model_pricing: known models, unknown model returns None
- estimate_cost: correct formula, batch discount, unknown model returns None
- BATCH_DISCOUNT constant value
"""

import pytest

from categorization.settings.pricing import (
    BATCH_DISCOUNT,
    PRICING,
    estimate_cost,
    get_model_pricing,
)


# ---------------------------------------------------------------------------
# PRICING dict
# ---------------------------------------------------------------------------


class TestPricingDict:
    def test_gpt_4o_mini_present(self):
        assert "gpt-4o-mini" in PRICING

    def test_gpt_4o_present(self):
        assert "gpt-4o" in PRICING

    def test_each_entry_has_input_and_output_keys(self):
        for model, rates in PRICING.items():
            assert "input" in rates, f"{model} missing 'input' key"
            assert "output" in rates, f"{model} missing 'output' key"

    def test_prices_are_positive(self):
        for model, rates in PRICING.items():
            assert rates["input"] > 0, f"{model} input price must be positive"
            assert rates["output"] > 0, f"{model} output price must be positive"

    def test_gpt_4o_mini_pricing_values(self):
        """Spot-check the known pricing for gpt-4o-mini."""
        rates = PRICING["gpt-4o-mini"]
        assert rates["input"] == pytest.approx(0.150)
        assert rates["output"] == pytest.approx(0.600)


# ---------------------------------------------------------------------------
# BATCH_DISCOUNT
# ---------------------------------------------------------------------------


class TestBatchDiscount:
    def test_batch_discount_is_0_5(self):
        assert BATCH_DISCOUNT == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_model_pricing
# ---------------------------------------------------------------------------


class TestGetModelPricing:
    def test_returns_dict_for_known_model(self):
        result = get_model_pricing("gpt-4o-mini")
        assert isinstance(result, dict)
        assert "input" in result
        assert "output" in result

    def test_returns_none_for_unknown_model(self):
        assert get_model_pricing("gpt-99-ultra") is None

    def test_returns_none_for_empty_string(self):
        assert get_model_pricing("") is None

    @pytest.mark.parametrize("model", list(PRICING.keys()))
    def test_all_registered_models_return_pricing(self, model):
        result = get_model_pricing(model)
        assert result is not None


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_basic_estimate_for_known_model(self):
        cost = estimate_cost(1000, "gpt-4o-mini", avg_input_tokens=500, avg_output_tokens=10)
        assert cost is not None
        assert cost > 0

    def test_returns_none_for_unknown_model(self):
        assert estimate_cost(100, "unknown-model-xyz") is None

    def test_batch_flag_halves_cost(self):
        cost_realtime = estimate_cost(1000, "gpt-4o-mini", batch=False)
        cost_batch = estimate_cost(1000, "gpt-4o-mini", batch=True)
        assert cost_realtime is not None
        assert cost_batch is not None
        assert cost_batch == pytest.approx(cost_realtime * BATCH_DISCOUNT)

    def test_zero_requests_returns_zero(self):
        cost = estimate_cost(0, "gpt-4o-mini")
        assert cost == pytest.approx(0.0)

    def test_cost_scales_linearly_with_requests(self):
        cost_100 = estimate_cost(100, "gpt-4o-mini")
        cost_200 = estimate_cost(200, "gpt-4o-mini")
        assert cost_100 is not None
        assert cost_200 is not None
        assert cost_200 == pytest.approx(cost_100 * 2)

    def test_manual_calculation_gpt4o_mini(self):
        """
        With gpt-4o-mini: input=$0.150/1M, output=$0.600/1M
        1000 requests * 500 input tokens = 500_000 input tokens = $0.075
        1000 requests * 10 output tokens = 10_000 output tokens = $0.006
        Total = $0.081
        """
        cost = estimate_cost(1000, "gpt-4o-mini", avg_input_tokens=500, avg_output_tokens=10, batch=False)
        expected = (1000 * 500 / 1_000_000) * 0.150 + (1000 * 10 / 1_000_000) * 0.600
        assert cost == pytest.approx(expected)

    def test_manual_calculation_with_batch_discount(self):
        cost = estimate_cost(1000, "gpt-4o-mini", avg_input_tokens=500, avg_output_tokens=10, batch=True)
        expected_full = (1000 * 500 / 1_000_000) * 0.150 + (1000 * 10 / 1_000_000) * 0.600
        assert cost == pytest.approx(expected_full * 0.5)

    def test_higher_token_counts_increase_cost(self):
        cost_low = estimate_cost(100, "gpt-4o-mini", avg_input_tokens=100)
        cost_high = estimate_cost(100, "gpt-4o-mini", avg_input_tokens=1000)
        assert cost_high > cost_low
