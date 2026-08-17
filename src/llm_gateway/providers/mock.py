"""A deterministic offline provider.

Lets the gateway run with no network and no API keys, for demos and
tests. Responses are derived from the request content, so the same
input always produces the same output text and token counts.

Special model names, useful for exercising failure paths:

- ``mock-error``: always raises a retryable ProviderError.
- ``mock-bad-request``: always raises a non-retryable ProviderError.
- ``mock-slow``: sleeps for MOCK_SLOW_SECONDS before answering, to
  trigger route timeouts.
"""

from __future__ import annotations

import asyncio
import hashlib

from ..schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage
from .base import Provider, ProviderError

MOCK_SLOW_SECONDS = 2.0


class MockProvider(Provider):
    name = "mock"

    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        if model == "mock-error":
            raise ProviderError(
                "mock upstream failure", provider=self.name, status_code=500, retryable=True
            )
        if model == "mock-bad-request":
            raise ProviderError(
                "mock rejected request", provider=self.name, status_code=400, retryable=False
            )
        if model == "mock-slow":
            await asyncio.sleep(MOCK_SLOW_SECONDS)

        prompt_text = " ".join(m.content or "" for m in request.messages)
        user_text = request.last_user_content()
        digest = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]

        if user_text.strip() == "[empty]":
            content = ""
        elif user_text.strip() == "[refuse]":
            content = "I cannot help with that request."
        else:
            content = f"Mock reply from {model} (ref {digest}): {user_text[:200]}"

        prompt_tokens = max(1, len(prompt_text.split()))
        completion_tokens = max(1, len(content.split())) if content else 0
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{digest}",
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
