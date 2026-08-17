"""SQLite-backed usage accounting and daily budget enforcement.

One row per API key per day. The store is intentionally simple:
synchronous SQLite behind a lock. At gateway request rates this is
not a bottleneck, and it keeps the deployment to a single file.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import BudgetConfig
from .schemas import Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    api_key TEXT NOT NULL,
    day TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key, day)
)
"""


def today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


class BudgetExceededError(Exception):
    def __init__(self, api_key: str, spent_usd: float, limit_usd: float) -> None:
        self.api_key = api_key
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Daily budget exceeded for key {api_key!r}: "
            f"spent {spent_usd:.6f} USD of {limit_usd:.6f} USD."
        )


class UsageStore:
    """Records spend and answers budget questions."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(self, api_key: str, usage: Usage, cost_usd: float, day: str | None = None) -> None:
        day = day or today_utc()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO usage
                    (api_key, day, requests, prompt_tokens, completion_tokens, cost_usd)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT (api_key, day) DO UPDATE SET
                    requests = requests + 1,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    cost_usd = cost_usd + excluded.cost_usd
                """,
                (api_key, day, usage.prompt_tokens, usage.completion_tokens, cost_usd),
            )
            self._conn.commit()

    def spend_for_day(self, api_key: str, day: str | None = None) -> float:
        day = day or today_utc()
        with self._lock:
            row = self._conn.execute(
                "SELECT cost_usd FROM usage WHERE api_key = ? AND day = ?",
                (api_key, day),
            ).fetchone()
        return float(row["cost_usd"]) if row else 0.0

    def report(self, api_key: str | None = None) -> list[dict[str, Any]]:
        """Spend per key per day, newest first."""
        query = (
            "SELECT api_key, day, requests, prompt_tokens, completion_tokens, cost_usd "
            "FROM usage"
        )
        params: tuple[Any, ...] = ()
        if api_key is not None:
            query += " WHERE api_key = ?"
            params = (api_key,)
        query += " ORDER BY day DESC, api_key ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


class BudgetEnforcer:
    """Checks a key's spend for today against its configured limit."""

    def __init__(self, budgets: BudgetConfig, store: UsageStore) -> None:
        self._budgets = budgets
        self._store = store

    def check(self, api_key: str) -> None:
        limit = self._budgets.limit_for(api_key)
        if limit is None:
            return
        spent = self._store.spend_for_day(api_key)
        if spent >= limit:
            raise BudgetExceededError(api_key, spent, limit)
