from __future__ import annotations

from llm_gateway.providers.anthropic import (
    DEFAULT_MAX_TOKENS,
    request_to_anthropic,
    response_from_anthropic,
)
from llm_gateway.providers.openai import build_payload
from llm_gateway.schemas import ChatCompletionRequest


def make_request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "fast",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def test_anthropic_request_moves_system_to_top_level() -> None:
    payload = request_to_anthropic(make_request(), "claude-sonnet-4-5")
    assert payload["system"] == "Be brief."
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["model"] == "claude-sonnet-4-5"


def test_anthropic_request_applies_default_max_tokens() -> None:
    payload = request_to_anthropic(make_request(), "claude-sonnet-4-5")
    assert payload["max_tokens"] == DEFAULT_MAX_TOKENS


def test_anthropic_request_respects_explicit_max_tokens() -> None:
    payload = request_to_anthropic(make_request(max_tokens=99), "claude-sonnet-4-5")
    assert payload["max_tokens"] == 99


def test_anthropic_request_translates_stop_to_stop_sequences() -> None:
    payload = request_to_anthropic(make_request(stop="END"), "m")
    assert payload["stop_sequences"] == ["END"]
    payload = request_to_anthropic(make_request(stop=["A", "B"]), "m")
    assert payload["stop_sequences"] == ["A", "B"]


def test_anthropic_request_joins_multiple_system_messages() -> None:
    request = make_request(
        messages=[
            {"role": "system", "content": "One."},
            {"role": "system", "content": "Two."},
            {"role": "user", "content": "Hi"},
        ]
    )
    payload = request_to_anthropic(request, "m")
    assert payload["system"] == "One.\n\nTwo."


def test_anthropic_response_translated_to_openai_shape() -> None:
    data = {
        "id": "msg_123",
        "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    response = response_from_anthropic(data, "claude-sonnet-4-5")
    assert response.object == "chat.completion"
    assert response.first_text() == "Hello world"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 14


def test_anthropic_max_tokens_stop_reason_maps_to_length() -> None:
    data = {
        "id": "msg_1",
        "content": [{"type": "text", "text": "cut off"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    response = response_from_anthropic(data, "m")
    assert response.choices[0].finish_reason == "length"


def test_openai_payload_swaps_alias_for_concrete_model() -> None:
    payload = build_payload(make_request(temperature=0.3), "gpt-4o-mini")
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0.3
    assert "stream" not in payload
    assert payload["messages"][1]["content"] == "Hello"
