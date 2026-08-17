from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app_from_config
from llm_gateway.config import GatewayConfig


def base_config_dict(db_path: str) -> dict[str, Any]:
    """A config that exercises routing, fallback, budgets, and evals offline."""
    return {
        "routes": {
            "fast": {
                "strategy": "priority",
                "retry": {"attempts": 1, "timeout_seconds": 5},
                "targets": [{"provider": "mock", "model": "mock-small"}],
            },
            "quality": {
                "strategy": "priority",
                "retry": {"attempts": 1, "timeout_seconds": 5},
                "targets": [{"provider": "mock", "model": "mock-large"}],
            },
            "flaky": {
                "strategy": "priority",
                "retry": {"attempts": 2, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-error"},
                    {"provider": "mock", "model": "mock-small"},
                ],
            },
            "doomed": {
                "strategy": "priority",
                "retry": {"attempts": 1, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-error"},
                    {"provider": "mock", "model": "mock-bad-request"},
                ],
            },
            "spread": {
                "strategy": "round_robin",
                "retry": {"attempts": 1, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-small"},
                    {"provider": "mock", "model": "mock-large"},
                ],
            },
        },
        "budgets": {"default_daily_usd": None, "keys": {}},
        "pricing": {
            "mock-small": {"input_per_1k": 1.0, "output_per_1k": 2.0},
        },
        "evals": {"enabled": True, "evaluators": ["empty_or_refusal"]},
        "database_path": db_path,
    }


@pytest.fixture()
def config_dict(tmp_path: Path) -> dict[str, Any]:
    return base_config_dict(str(tmp_path / "usage.db"))


@pytest.fixture()
def gateway_config(config_dict: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig.model_validate(config_dict)


@pytest.fixture()
def client(gateway_config: GatewayConfig) -> Iterator[TestClient]:
    app = create_app_from_config(gateway_config)
    with TestClient(app) as test_client:
        yield test_client
    app.state.usage_store.close()


def chat_payload(model: str = "fast", content: str = "Hello there") -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}]}
