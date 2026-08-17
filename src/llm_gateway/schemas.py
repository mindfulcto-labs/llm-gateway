"""OpenAI-compatible request and response schemas.

The gateway speaks the OpenAI chat completions wire format on both
sides. Provider adapters translate to and from other provider formats.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    """A standard OpenAI chat completions request.

    ``model`` is treated as a route alias by the gateway, for example
    ``fast`` or ``quality``. The router replaces it with the concrete
    provider model before dispatch.
    """

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    user: str | None = None

    model_config = ConfigDict(extra="allow")

    def last_user_content(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user" and message.content:
                return message.content
        return ""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)

    model_config = ConfigDict(extra="allow")

    def first_text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI-style error envelope."""

    error: ErrorDetail


def estimate_tokens(text: str) -> int:
    """A rough token estimate for providers that do not report usage.

    Uses the common four characters per token approximation. Good
    enough for budget accounting when the provider gives us nothing.
    """
    return max(1, len(text) // 4)
