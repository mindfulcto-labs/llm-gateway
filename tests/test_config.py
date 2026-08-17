from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import GatewayConfig, load_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "routes.yaml"


def test_example_config_loads_and_validates() -> None:
    config = load_config(EXAMPLE_CONFIG)
    assert set(config.routes) == {"fast", "quality", "balanced"}
    assert config.routes["balanced"].strategy == "round_robin"
    assert config.budgets.default_daily_usd == 5.0
    assert config.budgets.keys["demo-key"] == 1.0
    assert config.evals.enabled is True


def test_route_defaults_applied() -> None:
    config = GatewayConfig.model_validate(
        {"routes": {"r": {"targets": [{"provider": "mock", "model": "m"}]}}}
    )
    route = config.routes["r"]
    assert route.strategy == "priority"
    assert route.retry.attempts == 1
    assert route.retry.timeout_seconds == 30.0


def test_invalid_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(
            {
                "routes": {
                    "r": {
                        "strategy": "random",
                        "targets": [{"provider": "mock", "model": "m"}],
                    }
                }
            }
        )


def test_route_requires_at_least_one_target() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"routes": {"r": {"targets": []}}})


def test_budget_limit_lookup_prefers_key_entry() -> None:
    config = GatewayConfig.model_validate(
        {
            "routes": {"r": {"targets": [{"provider": "mock", "model": "m"}]}},
            "budgets": {"default_daily_usd": 5.0, "keys": {"special": 0.5}},
        }
    )
    assert config.budgets.limit_for("special") == 0.5
    assert config.budgets.limit_for("anyone-else") == 5.0
