"""FastAPI application.

Endpoints:

- ``POST /v1/chat/completions``: OpenAI-compatible chat completions.
- ``GET /v1/usage``: spend per API key per day.
- ``GET /v1/routes``: configured routes and their targets.
- ``GET /healthz``: liveness check.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse

from .config import GatewayConfig, load_config
from .evals import Evaluator, build_evaluators
from .logs import log_event
from .pricing import PriceTable
from .providers import Provider, build_default_providers
from .router import AllTargetsFailedError, Router, UnknownRouteError
from .schemas import ChatCompletionRequest, ChatCompletionResponse
from .usage import BudgetEnforcer, BudgetExceededError, UsageStore

DEFAULT_CONFIG_PATH = "examples/routes.yaml"
ANONYMOUS_KEY = "anonymous"


def _error_response(status_code: int, message: str, error_type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": code}},
    )


def _api_key_from_header(authorization: str | None) -> str:
    if not authorization:
        return ANONYMOUS_KEY
    if authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
        return key or ANONYMOUS_KEY
    return authorization.strip() or ANONYMOUS_KEY


def create_app_from_config(
    config: GatewayConfig,
    providers: dict[str, Provider] | None = None,
) -> FastAPI:
    """Build the app from an in-memory config. Used directly by tests."""
    providers = providers or build_default_providers()
    router = Router(config, providers)
    prices = PriceTable(config.pricing)
    store = UsageStore(config.database_path)
    budget = BudgetEnforcer(config.budgets, store)
    evaluators: list[Evaluator] = (
        build_evaluators(config.evals.evaluators) if config.evals.enabled else []
    )

    app = FastAPI(title="llm-gateway", version="0.1.0")
    app.state.config = config
    app.state.usage_store = store

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/routes")
    async def routes() -> dict[str, object]:
        summary = {}
        for alias, route in config.routes.items():
            summary[alias] = {
                "strategy": route.strategy,
                "retry": route.retry.model_dump(),
                "targets": [t.model_dump() for t in route.targets],
            }
        return {"routes": summary}

    @app.get("/v1/usage")
    async def usage(
        key: str | None = Query(default=None, description="Filter by API key."),
    ) -> dict[str, object]:
        return {"usage": store.report(api_key=key)}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        request_id = uuid.uuid4().hex[:16]
        api_key = _api_key_from_header(authorization)
        started = time.monotonic()

        if payload.stream:
            return _error_response(
                400,
                "Streaming is not supported in this release. Set stream to false.",
                "invalid_request_error",
                "stream_not_supported",
            )

        try:
            budget.check(api_key)
        except BudgetExceededError as exc:
            log_event(
                "budget_exceeded",
                level=logging.WARNING,
                request_id=request_id,
                api_key=api_key,
                spent_usd=exc.spent_usd,
                limit_usd=exc.limit_usd,
            )
            return _error_response(
                429,
                str(exc),
                "insufficient_quota",
                "daily_budget_exceeded",
            )

        try:
            result = await router.dispatch(payload)
        except UnknownRouteError as exc:
            return _error_response(404, str(exc), "invalid_request_error", "model_not_found")
        except AllTargetsFailedError as exc:
            latency_ms = round((time.monotonic() - started) * 1000, 1)
            log_event(
                "request_failed",
                level=logging.ERROR,
                request_id=request_id,
                api_key=api_key,
                route=exc.alias,
                latency_ms=latency_ms,
                attempts=[a.__dict__ for a in exc.attempts],
            )
            return _error_response(
                502,
                str(exc),
                "upstream_error",
                "all_targets_failed",
            )

        latency_ms = round((time.monotonic() - started) * 1000, 1)
        response: ChatCompletionResponse = result.response
        cost_usd = prices.cost_usd(result.model, response.usage)
        store.record(api_key, response.usage, cost_usd)

        log_event(
            "request_completed",
            request_id=request_id,
            api_key=api_key,
            route=result.route,
            provider=result.provider,
            model=result.model,
            latency_ms=latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=cost_usd,
            attempts=result.attempts,
            fallback_hops=result.fallback_hops,
        )

        for evaluator in evaluators:
            try:
                eval_result = evaluator.score(payload, response)
                log_event(
                    "eval_scored",
                    request_id=request_id,
                    evaluator=eval_result.name,
                    score=eval_result.score,
                    detail=eval_result.detail,
                )
            except Exception as exc:  # noqa: BLE001 - evaluators must never fail a request
                log_event(
                    "eval_error",
                    level=logging.WARNING,
                    request_id=request_id,
                    evaluator=getattr(evaluator, "name", repr(evaluator)),
                    error=str(exc),
                )

        return response

    return app


def create_app() -> FastAPI:
    """App factory for uvicorn. Reads GATEWAY_CONFIG for the config path."""
    config_path = os.environ.get("GATEWAY_CONFIG", DEFAULT_CONFIG_PATH)
    return create_app_from_config(load_config(config_path))
