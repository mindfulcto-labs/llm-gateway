from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.conftest import base_config_dict, chat_payload

from llm_gateway.app import create_app_from_config
from llm_gateway.config import GatewayConfig


def budget_client(tmp_path: Path, budgets: dict[str, Any]) -> Iterator[TestClient]:
    config_dict = base_config_dict(str(tmp_path / "usage.db"))
    config_dict["budgets"] = budgets
    app = create_app_from_config(GatewayConfig.model_validate(config_dict))
    with TestClient(app) as client:
        yield client
    app.state.usage_store.close()


@pytest.fixture()
def tight_budget_client(tmp_path: Path) -> Iterator[TestClient]:
    # mock-small is priced at 1.0/2.0 USD per 1K tokens in the test
    # config, so a couple of requests cross this limit quickly.
    yield from budget_client(
        tmp_path,
        {"default_daily_usd": 0.02, "keys": {"rich-key": 100.0, "broke-key": 0.0}},
    )


def spend_until_blocked(client: TestClient, key: str, max_requests: int = 50) -> int:
    """Send requests until a 429 arrives. Returns how many succeeded."""
    for i in range(max_requests):
        response = client.post(
            "/v1/chat/completions",
            json=chat_payload("fast", "words " * 50),
            headers={"Authorization": f"Bearer {key}"},
        )
        if response.status_code == 429:
            return i
        assert response.status_code == 200
    raise AssertionError("budget never triggered")


def test_request_under_budget_succeeds(tight_budget_client: TestClient) -> None:
    response = tight_budget_client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer some-key"},
    )
    assert response.status_code == 200


def test_breaching_daily_budget_returns_429(tight_budget_client: TestClient) -> None:
    succeeded = spend_until_blocked(tight_budget_client, "some-key")
    assert succeeded >= 1
    response = tight_budget_client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer some-key"},
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "daily_budget_exceeded"


def test_zero_budget_key_blocked_from_first_paid_request(
    tight_budget_client: TestClient,
) -> None:
    # With a 0.0 limit the very first check trips (0.0 >= 0.0), so
    # the key can never spend anything.
    response = tight_budget_client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer broke-key"},
    )
    assert response.status_code == 429


def test_per_key_limit_overrides_default(tight_budget_client: TestClient) -> None:
    # Exhaust the default budget on one key.
    spend_until_blocked(tight_budget_client, "some-key")
    # A key with its own generous limit still gets through.
    response = tight_budget_client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer rich-key"},
    )
    assert response.status_code == 200


def test_budgets_are_isolated_per_key(tight_budget_client: TestClient) -> None:
    spend_until_blocked(tight_budget_client, "key-a")
    response = tight_budget_client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer key-b"},
    )
    assert response.status_code == 200


def test_missing_authorization_uses_anonymous_key(tmp_path: Path) -> None:
    gen = budget_client(tmp_path, {"default_daily_usd": None, "keys": {}})
    client = next(gen)
    response = client.post("/v1/chat/completions", json=chat_payload("fast"))
    assert response.status_code == 200
    usage = client.get("/v1/usage").json()["usage"]
    assert usage[0]["api_key"] == "anonymous"
