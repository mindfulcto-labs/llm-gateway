"""Common provider interface.

Every adapter takes an OpenAI-format request plus a concrete model
name and returns an OpenAI-format response. Translation to and from
other wire formats happens inside the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Raised by adapters on any upstream failure.

    ``retryable`` controls fallback behaviour: retryable errors are
    retried on the same target up to the route's attempt limit, then
    the router moves to the next target. Non-retryable errors skip
    straight to the next target.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(message)


class Provider(ABC):
    """A chat completions backend."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Run one chat completion against ``model`` and return the result."""
        raise NotImplementedError
