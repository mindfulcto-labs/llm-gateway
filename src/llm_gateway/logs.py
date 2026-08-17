"""Structured JSON logging.

One JSON object per line on stdout. Request logs carry route,
provider, latency, token counts, cost, and fallback hops so a log
pipeline can aggregate spend and reliability without parsing text.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

LOGGER_NAME = "llm_gateway"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = repr(record.exc_info[1])
        return json.dumps(payload, default=str)


class StdoutHandler(logging.Handler):
    """Writes to whatever sys.stdout is at emit time.

    Resolving the stream late keeps logs working under process
    managers and test runners that swap stdout.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    get_logger().log(level, event, extra={"fields": fields})
