"""Static price table and cost calculation.

Illustrative prices, update for your providers. Prices move often, so
treat these as placeholders and override them in your config file
under the ``pricing`` key. Unknown models cost 0.0 so the gateway
never blocks on a missing price, but the miss is visible in logs.
"""

from __future__ import annotations

from .config import ModelPrice
from .schemas import Usage

# Illustrative prices, update for your providers.
# All values are USD per 1K tokens.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(input_per_1k=0.0025, output_per_1k=0.01),
    "gpt-4o-mini": ModelPrice(input_per_1k=0.00015, output_per_1k=0.0006),
    "claude-sonnet-4-5": ModelPrice(input_per_1k=0.003, output_per_1k=0.015),
    "claude-haiku-4-5": ModelPrice(input_per_1k=0.001, output_per_1k=0.005),
    # The mock provider is free by definition.
    "mock-small": ModelPrice(input_per_1k=0.0, output_per_1k=0.0),
    "mock-large": ModelPrice(input_per_1k=0.0, output_per_1k=0.0),
}


class PriceTable:
    """Default prices merged with config overrides."""

    def __init__(self, overrides: dict[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(DEFAULT_PRICES)
        if overrides:
            self._prices.update(overrides)

    def price_for(self, model: str) -> ModelPrice | None:
        return self._prices.get(model)

    def cost_usd(self, model: str, usage: Usage) -> float:
        """Cost of one request in USD. Unknown models cost zero."""
        price = self._prices.get(model)
        if price is None:
            return 0.0
        cost = (
            usage.prompt_tokens / 1000.0 * price.input_per_1k
            + usage.completion_tokens / 1000.0 * price.output_per_1k
        )
        return round(cost, 8)
