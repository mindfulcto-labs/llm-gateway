"""Anthropic adapter.

Translates between the OpenAI chat completions format and the
Anthropic Messages API in both directions. The translation functions
are pure so they can be tested without any network access.

Differences handled here:

- System messages move from the messages list to the ``system`` field.
- ``max_tokens`` is required by Anthropic, so a default is applied.
- ``stop`` becomes ``stop_sequences``.
- Response content blocks are joined into a single string.
- ``stop_reason`` maps to an OpenAI ``finish_reason``.
- ``input_tokens``/``output_tokens`` map to prompt/completion tokens.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)
from .base import Provider, ProviderError

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def request_to_anthropic(request: ChatCompletionRequest, model: str) -> dict[str, Any]:
    """Translate an OpenAI-format request into an Anthropic Messages payload."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role in ("user", "assistant"):
            messages.append({"role": message.role, "content": message.content or ""})
        else:
            # Tool messages have no direct equivalent in this minimal
            # adapter; forward them as user turns so context survives.
            messages.append({"role": "user", "content": message.content or ""})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop is not None:
        stop = request.stop
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    return payload


def response_from_anthropic(data: dict[str, Any], model: str) -> ChatCompletionResponse:
    """Translate an Anthropic Messages response into the OpenAI format."""
    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    stop_reason = data.get("stop_reason")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop") if stop_reason else "stop"
    usage_data = data.get("usage", {})
    prompt_tokens = int(usage_data.get("input_tokens", 0))
    completion_tokens = int(usage_data.get("output_tokens", 0))
    return ChatCompletionResponse(
        id=data.get("id", "chatcmpl-anthropic"),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=60.0)

    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        if not self._api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set", provider=self.name, retryable=False
            )
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json=request_to_anthropic(request, model),
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Anthropic request failed: {exc}", provider=self.name, retryable=True
            ) from exc

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise ProviderError(
                f"Anthropic returned {response.status_code}: {response.text[:500]}",
                provider=self.name,
                status_code=response.status_code,
                retryable=retryable,
            )
        return response_from_anthropic(response.json(), model)
