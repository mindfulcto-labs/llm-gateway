from __future__ import annotations

import pytest
from tests.conftest import chat_payload

from llm_gateway.config import GatewayConfig
from llm_gateway.providers import build_default_providers
from llm_gateway.providers.mock import MockProvider
from llm_gateway.router import AllTargetsFailedError, Router, UnknownRouteError
from llm_gateway.schemas import ChatCompletionRequest


def make_router(routes: dict, providers=None) -> Router:
    config = GatewayConfig.model_validate({"routes": routes})
    return Router(config, providers or build_default_providers())


def make_request(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(chat_payload(model))


async def test_failing_target_falls_back_to_next() -> None:
    router = make_router(
        {
            "r": {
                "retry": {"attempts": 1, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-error"},
                    {"provider": "mock", "model": "mock-small"},
                ],
            }
        }
    )
    result = await router.dispatch(make_request("r"))
    assert result.model == "mock-small"
    assert result.fallback_hops == 1
    assert result.attempts == 2
    assert len(result.failed_attempts) == 1
    assert result.failed_attempts[0].model == "mock-error"


async def test_retryable_error_consumes_all_attempts_before_fallback() -> None:
    router = make_router(
        {
            "r": {
                "retry": {"attempts": 3, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-error"},
                    {"provider": "mock", "model": "mock-small"},
                ],
            }
        }
    )
    result = await router.dispatch(make_request("r"))
    assert result.model == "mock-small"
    # Three failed tries on the first target, then one success.
    assert result.attempts == 4
    assert len(result.failed_attempts) == 3


async def test_non_retryable_error_skips_remaining_attempts() -> None:
    router = make_router(
        {
            "r": {
                "retry": {"attempts": 3, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-bad-request"},
                    {"provider": "mock", "model": "mock-small"},
                ],
            }
        }
    )
    result = await router.dispatch(make_request("r"))
    assert result.model == "mock-small"
    # One failed try on the first target (no retries), then one success.
    assert result.attempts == 2
    assert len(result.failed_attempts) == 1


async def test_all_targets_failing_raises_with_attempt_details() -> None:
    router = make_router(
        {
            "r": {
                "retry": {"attempts": 2, "timeout_seconds": 5},
                "targets": [
                    {"provider": "mock", "model": "mock-error"},
                    {"provider": "mock", "model": "mock-bad-request"},
                ],
            }
        }
    )
    with pytest.raises(AllTargetsFailedError) as excinfo:
        await router.dispatch(make_request("r"))
    # Two retryable failures plus one non-retryable failure.
    assert len(excinfo.value.attempts) == 3


async def test_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_gateway.providers.mock.MOCK_SLOW_SECONDS", 0.5)
    router = make_router(
        {
            "r": {
                "retry": {"attempts": 1, "timeout_seconds": 0.05},
                "targets": [
                    {"provider": "mock", "model": "mock-slow"},
                    {"provider": "mock", "model": "mock-small"},
                ],
            }
        }
    )
    result = await router.dispatch(make_request("r"))
    assert result.model == "mock-small"
    assert result.fallback_hops == 1
    assert "timeout" in result.failed_attempts[0].error


async def test_unknown_route_raises() -> None:
    router = make_router(
        {"r": {"targets": [{"provider": "mock", "model": "mock-small"}]}}
    )
    with pytest.raises(UnknownRouteError):
        await router.dispatch(make_request("missing"))


def test_router_rejects_unknown_provider_at_startup() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        make_router(
            {"r": {"targets": [{"provider": "nope", "model": "m"}]}},
            providers={"mock": MockProvider()},
        )


def test_gateway_returns_502_when_all_targets_fail(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("doomed"))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "all_targets_failed"


def test_gateway_serves_from_fallback_target(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("flaky"))
    assert response.status_code == 200
    assert response.json()["model"] == "mock-small"
