"""structlog configuration.

One initialization site so every module gets the same renderer, timestamper, and
log level. Renderer switches between human-readable console and JSON via config,
so scheduler output pipes cleanly to a log aggregator in prod.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

_INITIALIZED = False


def configure(level: str = "INFO", renderer: Literal["console", "json"] = "console") -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    final: structlog.typing.Processor
    final = (
        structlog.dev.ConsoleRenderer(colors=True)
        if renderer == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, final],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _INITIALIZED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _INITIALIZED:
        configure()
    return structlog.get_logger(name) if name else structlog.get_logger()
