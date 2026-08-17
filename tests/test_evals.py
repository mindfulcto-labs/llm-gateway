from __future__ import annotations

import pytest
from tests.conftest import chat_payload

from llm_gateway.evals import EmptyOrRefusalEvaluator, build_evaluators
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
)


def make_response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="mock-small",
        choices=[Choice(message=ChatMessage(role="assistant", content=content))],
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(chat_payload("fast"))


def test_empty_response_scores_zero() -> None:
    result = EmptyOrRefusalEvaluator().score(make_request(), make_response(""))
    assert result.score == 0.0
    assert result.detail == "empty response"


def test_refusal_scores_low() -> None:
    result = EmptyOrRefusalEvaluator().score(
        make_request(), make_response("I cannot help with that request.")
    )
    assert result.score == 0.2
    assert "refusal marker" in (result.detail or "")


def test_normal_answer_scores_one() -> None:
    result = EmptyOrRefusalEvaluator().score(
        make_request(), make_response("The capital of France is Paris.")
    )
    assert result.score == 1.0


def test_build_evaluators_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown evaluator"):
        build_evaluators(["does_not_exist"])


def test_eval_scores_logged_for_each_request(client, capsys: pytest.CaptureFixture) -> None:
    client.post("/v1/chat/completions", json=chat_payload("fast", "[refuse]"))
    captured = capsys.readouterr().out
    assert '"event": "eval_scored"' in captured
    assert '"evaluator": "empty_or_refusal"' in captured
    assert '"score": 0.2' in captured
