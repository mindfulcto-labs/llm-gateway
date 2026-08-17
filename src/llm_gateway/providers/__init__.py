"""Provider registry."""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import Provider, ProviderError
from .mock import MockProvider
from .openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "MockProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "build_default_providers",
]


def build_default_providers() -> dict[str, Provider]:
    """The three built-in providers, keyed by the name used in route configs."""
    return {
        "mock": MockProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }
