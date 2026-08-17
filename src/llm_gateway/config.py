"""Gateway configuration loaded from YAML.

A config file defines named routes, budgets, and price overrides.
See ``examples/routes.yaml`` for a full annotated example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RetryPolicy(BaseModel):
    """Retry and timeout rules applied per route target."""

    attempts: int = Field(
        default=1, ge=1, le=10, description="Attempts per target before moving on."
    )
    timeout_seconds: float = Field(default=30.0, gt=0, description="Timeout per attempt.")
    backoff_seconds: float = Field(default=0.0, ge=0, description="Sleep between attempts.")

    model_config = ConfigDict(extra="forbid")


class RouteTarget(BaseModel):
    """One provider and model pair inside a route chain."""

    provider: str
    model: str

    model_config = ConfigDict(extra="forbid")


class Route(BaseModel):
    """A named route mapping an alias to a chain of targets.

    ``priority`` tries targets in listed order. ``round_robin`` rotates
    the starting target across requests; on failure it still falls
    through the remaining targets in rotated order.
    """

    strategy: Literal["priority", "round_robin"] = "priority"
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    targets: list[RouteTarget] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ModelPrice(BaseModel):
    """Price per 1K tokens. Illustrative prices, update for your providers."""

    input_per_1k: float = Field(ge=0)
    output_per_1k: float = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


class BudgetConfig(BaseModel):
    """Daily spend limits in USD, keyed by API key.

    ``default_daily_usd`` applies to any key without an explicit entry.
    ``None`` means no limit.
    """

    default_daily_usd: float | None = Field(default=None, ge=0)
    keys: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def limit_for(self, api_key: str) -> float | None:
        if api_key in self.keys:
            return self.keys[api_key]
        return self.default_daily_usd


class EvalsConfig(BaseModel):
    """Post-response evaluation hooks. Names refer to the registry in evals.py."""

    enabled: bool = False
    evaluators: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class GatewayConfig(BaseModel):
    routes: dict[str, Route] = Field(min_length=1)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    database_path: str = "usage.db"

    model_config = ConfigDict(extra="forbid")


def load_config(path: str | Path) -> GatewayConfig:
    """Load and validate a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return GatewayConfig.model_validate(raw)
