"""Route resolution, fallback, and retry.

A request names a route alias in its ``model`` field. The router looks
up the route, orders its targets by strategy, and works through them.
Each target gets up to ``retry.attempts`` tries, each bounded by
``retry.timeout_seconds``. Retryable errors and timeouts consume
attempts; non-retryable errors skip straight to the next target.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field

from .config import GatewayConfig, Route, RouteTarget
from .providers.base import Provider, ProviderError
from .schemas import ChatCompletionRequest, ChatCompletionResponse


class UnknownRouteError(Exception):
    def __init__(self, alias: str, known: list[str]) -> None:
        self.alias = alias
        self.known = known
        super().__init__(f"Unknown route {alias!r}. Known routes: {', '.join(sorted(known))}.")


@dataclass
class AttemptRecord:
    provider: str
    model: str
    error: str


class AllTargetsFailedError(Exception):
    def __init__(self, alias: str, attempts: list[AttemptRecord]) -> None:
        self.alias = alias
        self.attempts = attempts
        detail = "; ".join(f"{a.provider}/{a.model}: {a.error}" for a in attempts)
        super().__init__(f"All targets failed for route {alias!r}: {detail}")


@dataclass
class RouteResult:
    response: ChatCompletionResponse
    route: str
    provider: str
    model: str
    attempts: int
    fallback_hops: int
    failed_attempts: list[AttemptRecord] = field(default_factory=list)


class Router:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider]) -> None:
        self._config = config
        self._providers = providers
        self._rr_counters: dict[str, itertools.count] = {}
        self._validate()

    def _validate(self) -> None:
        for alias, route in self._config.routes.items():
            for target in route.targets:
                if target.provider not in self._providers:
                    raise ValueError(
                        f"Route {alias!r} references unknown provider {target.provider!r}. "
                        f"Available: {', '.join(sorted(self._providers))}."
                    )

    def route_names(self) -> list[str]:
        return list(self._config.routes)

    def get_route(self, alias: str) -> Route:
        try:
            return self._config.routes[alias]
        except KeyError:
            raise UnknownRouteError(alias, list(self._config.routes)) from None

    def ordered_targets(self, alias: str) -> list[RouteTarget]:
        """Targets in the order this request should try them."""
        route = self.get_route(alias)
        targets = list(route.targets)
        if route.strategy == "round_robin":
            counter = self._rr_counters.setdefault(alias, itertools.count())
            start = next(counter) % len(targets)
            targets = targets[start:] + targets[:start]
        return targets

    async def dispatch(self, request: ChatCompletionRequest) -> RouteResult:
        alias = request.model
        route = self.get_route(alias)
        targets = self.ordered_targets(alias)
        failed: list[AttemptRecord] = []
        total_attempts = 0

        for hop, target in enumerate(targets):
            provider = self._providers[target.provider]
            for attempt in range(route.retry.attempts):
                total_attempts += 1
                try:
                    response = await asyncio.wait_for(
                        provider.complete(request, target.model),
                        timeout=route.retry.timeout_seconds,
                    )
                    return RouteResult(
                        response=response,
                        route=alias,
                        provider=target.provider,
                        model=target.model,
                        attempts=total_attempts,
                        fallback_hops=hop,
                        failed_attempts=failed,
                    )
                except TimeoutError:
                    failed.append(
                        AttemptRecord(
                            provider=target.provider,
                            model=target.model,
                            error=f"timeout after {route.retry.timeout_seconds}s",
                        )
                    )
                except ProviderError as exc:
                    failed.append(
                        AttemptRecord(
                            provider=target.provider, model=target.model, error=str(exc)
                        )
                    )
                    if not exc.retryable:
                        break
                if route.retry.backoff_seconds and attempt < route.retry.attempts - 1:
                    await asyncio.sleep(route.retry.backoff_seconds)

        raise AllTargetsFailedError(alias, failed)
