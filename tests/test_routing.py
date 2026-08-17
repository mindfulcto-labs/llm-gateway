from __future__ import annotations

from tests.conftest import chat_payload


def test_alias_resolves_to_configured_target(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("fast"))
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock-small"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_second_alias_resolves_to_its_own_target(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("quality"))
    assert response.status_code == 200
    assert response.json()["model"] == "mock-large"


def test_unknown_alias_returns_404(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("no-such-route"))
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert "no-such-route" in error["message"]


def test_round_robin_rotates_targets_across_requests(client) -> None:
    first = client.post("/v1/chat/completions", json=chat_payload("spread"))
    second = client.post("/v1/chat/completions", json=chat_payload("spread"))
    third = client.post("/v1/chat/completions", json=chat_payload("spread"))
    models = [r.json()["model"] for r in (first, second, third)]
    assert models == ["mock-small", "mock-large", "mock-small"]


def test_mock_provider_is_deterministic(client) -> None:
    a = client.post("/v1/chat/completions", json=chat_payload("fast", "same input"))
    b = client.post("/v1/chat/completions", json=chat_payload("fast", "same input"))
    assert a.json()["choices"] == b.json()["choices"]
    assert a.json()["usage"] == b.json()["usage"]


def test_streaming_rejected_with_400(client) -> None:
    payload = chat_payload("fast")
    payload["stream"] = True
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "stream_not_supported"
