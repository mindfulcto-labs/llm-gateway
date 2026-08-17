"""Post-response evaluation hooks.

An evaluator scores a completed request and response pair. The gateway
runs configured evaluators after each successful completion and logs
the scores. Evaluators are offline by design: they must not make
external calls, and a failing evaluator never fails the request.

To add one, implement the ``Evaluator`` protocol and register it in
``EVALUATOR_REGISTRY``, then list its name under ``evals.evaluators``
in the config file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .schemas import ChatCompletionRequest, ChatCompletionResponse


class EvalResult(BaseModel):
    name: str
    score: float
    detail: str | None = None


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def score(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ) -> EvalResult: ...


class EmptyOrRefusalEvaluator:
    """Flags empty answers and blanket refusals.

    Scores 0.0 for an empty response, 0.2 when the response opens with
    a refusal phrase, and 1.0 otherwise. A cheap first line of signal
    on answer quality that needs no model calls.
    """

    name = "empty_or_refusal"

    REFUSAL_MARKERS = (
        "i cannot",
        "i can't",
        "i am unable",
        "i'm unable",
        "i am not able",
        "i'm not able",
        "as an ai",
        "i won't",
    )

    def score(
        self, request: ChatCompletionRequest, response: ChatCompletionResponse
    ) -> EvalResult:
        text = response.first_text().strip()
        if not text:
            return EvalResult(name=self.name, score=0.0, detail="empty response")
        opening = text[:120].lower()
        for marker in self.REFUSAL_MARKERS:
            if marker in opening:
                return EvalResult(
                    name=self.name, score=0.2, detail=f"refusal marker: {marker!r}"
                )
        return EvalResult(name=self.name, score=1.0, detail=None)


EVALUATOR_REGISTRY: dict[str, type] = {
    EmptyOrRefusalEvaluator.name: EmptyOrRefusalEvaluator,
}


def build_evaluators(names: list[str]) -> list[Evaluator]:
    evaluators: list[Evaluator] = []
    for name in names:
        try:
            evaluators.append(EVALUATOR_REGISTRY[name]())
        except KeyError:
            raise ValueError(
                f"Unknown evaluator {name!r}. Available: {', '.join(sorted(EVALUATOR_REGISTRY))}."
            ) from None
    return evaluators
