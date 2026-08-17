from __future__ import annotations

import json

import pytest
from tests.conftest import chat_payload


def test_healthz(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_routes_endpoint_lists_configured_routes(client) -> None:
    body = client.get("/v1/routes").json()
    assert "fast" in body["routes"]
    fast = body["routes"]["fast"]
    assert fast["strategy"] == "priority"
    assert fast["targets"] == [{"provider": "mock", "model": "mock-small"}]
    assert fast["retry"]["attempts"] == 1


def test_chat_completion_matches_openai_shape(client) -> None:
    response = client.post("/v1/chat/completions", json=chat_payload("fast"))
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_request_without_messages_rejected(client) -> None:
    response = client.post("/v1/chat/completions", json={"model": "fast", "messages": []})
    assert response.status_code == 422


def test_request_log_line_is_structured_json(client, capsys: pytest.CaptureFixture) -> None:
    client.post("/v1/chat/completions", json=chat_payload("fast"))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    completed = [json.loads(line) for line in lines if '"request_completed"' in line]
    assert completed, "expected a request_completed log line"
    entry = completed[0]
    for field in (
        "request_id",
        "route",
        "provider",
        "model",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "fallback_hops",
    ):
        assert field in entry
    assert entry["route"] == "fast"
    assert entry["provider"] == "mock"


def test_fallback_hops_visible_in_logs(client, capsys: pytest.CaptureFixture) -> None:
    client.post("/v1/chat/completions", json=chat_payload("flaky"))
    lines = capsys.readouterr().out.splitlines()
    completed = [json.loads(line) for line in lines if '"request_completed"' in line]
    assert completed[0]["fallback_hops"] == 1
    assert completed[0]["attempts"] == 3
