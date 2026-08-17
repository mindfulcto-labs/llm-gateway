"""OpenAI adapter.

The gateway already speaks the OpenAI wire format, so this adapter
only swaps the route alias for the concrete model, forwards the
request, and validates the response shape.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from .base import Provider, ProviderError

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def build_payload(request: ChatCompletionRequest, model: str) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True, exclude={"stream"})
    payload["model"] = model
    return payload


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self._client = client or httpx.AsyncClient(timeout=60.0)

    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        if not self._api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set", provider=self.name, retryable=False
            )
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=build_payload(request, model),
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI request failed: {exc}", provider=self.name, retryable=True
            ) from exc

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise ProviderError(
                f"OpenAI returned {response.status_code}: {response.text[:500]}",
                provider=self.name,
                status_code=response.status_code,
                retryable=retryable,
            )
        return ChatCompletionResponse.model_validate(response.json())
