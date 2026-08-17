from __future__ import annotations

from pathlib import Path

from tests.conftest import chat_payload

from llm_gateway.pricing import PriceTable
from llm_gateway.schemas import Usage
from llm_gateway.usage import UsageStore


def test_usage_recorded_per_key_per_day(tmp_path: Path) -> None:
    store = UsageStore(tmp_path / "u.db")
    store.record("key-a", Usage(prompt_tokens=100, completion_tokens=50), 0.01, day="2026-01-01")
    store.record("key-a", Usage(prompt_tokens=200, completion_tokens=100), 0.02, day="2026-01-01")
    store.record("key-a", Usage(prompt_tokens=10, completion_tokens=5), 0.001, day="2026-01-02")
    store.record("key-b", Usage(prompt_tokens=1, completion_tokens=1), 0.5, day="2026-01-01")

    report = store.report("key-a")
    assert len(report) == 2
    day_one = next(r for r in report if r["day"] == "2026-01-01")
    assert day_one["requests"] == 2
    assert day_one["prompt_tokens"] == 300
    assert day_one["completion_tokens"] == 150
    assert abs(day_one["cost_usd"] - 0.03) < 1e-9
    store.close()


def test_spend_for_day_defaults_to_zero(tmp_path: Path) -> None:
    store = UsageStore(tmp_path / "u.db")
    assert store.spend_for_day("never-seen") == 0.0
    store.close()


def test_usage_endpoint_reports_spend(client) -> None:
    client.post(
        "/v1/chat/completions",
        json=chat_payload("fast"),
        headers={"Authorization": "Bearer report-key"},
    )
    body = client.get("/v1/usage", params={"key": "report-key"}).json()
    assert len(body["usage"]) == 1
    row = body["usage"][0]
    assert row["api_key"] == "report-key"
    assert row["requests"] == 1
    assert row["prompt_tokens"] > 0
    assert row["cost_usd"] > 0


def test_usage_endpoint_filters_by_key(client) -> None:
    for key in ("alpha", "beta"):
        client.post(
            "/v1/chat/completions",
            json=chat_payload("fast"),
            headers={"Authorization": f"Bearer {key}"},
        )
    all_rows = client.get("/v1/usage").json()["usage"]
    assert {row["api_key"] for row in all_rows} == {"alpha", "beta"}
    alpha_rows = client.get("/v1/usage", params={"key": "alpha"}).json()["usage"]
    assert {row["api_key"] for row in alpha_rows} == {"alpha"}


def test_cost_calculation_uses_price_table() -> None:
    prices = PriceTable()
    usage = Usage(prompt_tokens=1000, completion_tokens=2000)
    cost = prices.cost_usd("gpt-4o-mini", usage)
    assert abs(cost - (0.00015 + 2 * 0.0006)) < 1e-9


def test_unknown_model_costs_zero() -> None:
    prices = PriceTable()
    usage = Usage(prompt_tokens=1000, completion_tokens=1000)
    assert prices.cost_usd("no-such-model", usage) == 0.0


def test_config_pricing_overrides_defaults() -> None:
    from llm_gateway.config import ModelPrice

    prices = PriceTable({"gpt-4o-mini": ModelPrice(input_per_1k=1.0, output_per_1k=1.0)})
    usage = Usage(prompt_tokens=1000, completion_tokens=1000)
    assert abs(prices.cost_usd("gpt-4o-mini", usage) - 2.0) < 1e-9
