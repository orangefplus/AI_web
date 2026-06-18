"""Centralised logging configuration for the multi-agent framework.

The default behavior is intentionally quiet: langchain / httpx / openai
loggers are downgraded to WARNING so that the terminal does not get
flooded with raw LLM messages. Framework-level key events are
emitted under dedicated child loggers that the supervisor uses to
print a single readable line per step.

A second guarantee: **logs are always plain text**. Any ANSI escape
sequence a third-party library (colorlog, rich, click, ...) tries to
inject is stripped before the line is emitted, so no level marker ever
shows up as red/green/blue in the terminal. This is enforced by a
:class:`_StripAnsiFilter` attached to the root handler.

Usage in application code::

    from tools._logging import setup_logging
    setup_logging()  # idempotent

    from logging import getLogger
    log = getLogger("agent.supervisor.step")
    log.info("STEP %d/%d [%s]: %s", 3, 5, "api", "OpenAlex search")

Environment knobs:

- ``AI_WEB_LOG_LEVEL`` (default ``INFO``): root log level.
- ``AI_WEB_LOG_NOISY_LEVEL`` (default ``WARNING``): cap noisy
  third-party loggers.
- ``AI_WEB_LOG_FORMAT`` (default includes timestamp + name + level).
- ``AI_WEB_LOG_NO_COLOR`` (default unset): even with this env var off
  the ANSI filter is still installed; set it to ``1`` to also strip
  the optional colour-filter inside the formatter (no-op today, kept
  for forward compatibility with colour-capable formatters).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Final

# ---------------------------------------------------------------------------
# Force stdout / stderr to UTF-8 on Windows so that Chinese characters
# survive the Python -> PowerShell -> file chain.  PowerShell on a
# Chinese-locale Windows box defaults to the OEM code page (GBK/CP936)
# for both the console and the Out-File / Tee-Object destinations; the
# downstream .log files would otherwise be unreadable in the IDE.
# ---------------------------------------------------------------------------
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:  # noqa: BLE001
            pass


ROOT_LOGGER_NAME: Final[str] = ""  # root
DEFAULT_FORMAT: Final[str] = "[%(asctime)s] [%(name)s] %(levelname)s | %(message)s"

# ANSI escape sequence matcher. Covers SGR (colours, styles) and the
# 2-/3-byte private-mode introducers. Using a generous pattern so
# colourised exception traces and progress bars are also stripped.
_ANSI_ESCAPE_RE: Final[re.Pattern[str]] = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

# Third-party loggers that the framework knows are noisy. Anything
# the user wants to debug can be re-enabled by setting the env var.
NOISY_LOGGERS: Final[dict[str, int]] = {
    "langchain": logging.WARNING,
    "langchain_core": logging.WARNING,
    "langchain_openai": logging.WARNING,
    "langgraph": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "openai": logging.WARNING,
}

# Key event loggers (children of the root logger) the supervisor uses.
KEY_EVENT_LOGGERS: Final[tuple[str, ...]] = (
    "agent.supervisor.intent",
    "agent.supervisor.plan",
    "agent.supervisor.step",
    "agent.supervisor.result",
    "agent.supervisor.error",
)


_initialised = False


class _StripAnsiFilter(logging.Filter):
    """Drop every ANSI escape sequence from log records.

    Even if a third-party logger (colorlog, rich, click, ...) tries to
    colourise WARNING/ERROR lines red, this filter removes those bytes
    before the handler writes to the terminal. The user explicitly
    asked for "no red in the logs", so we honour that across the stack
    without having to audit every dependency.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = _ANSI_ESCAPE_RE.sub("", record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (_ANSI_ESCAPE_RE.sub("", v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    _ANSI_ESCAPE_RE.sub("", a) if isinstance(a, str) else a
                    for a in record.args
                )
        if record.exc_info:
            # exc_text is generated lazily by logging; force it now
            # while we still control the text, then strip escapes.
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = _ANSI_ESCAPE_RE.sub("", record.exc_text)
        return True


def setup_logging(
    level: str | int | None = None,
    noisy_level: str | int | None = None,
    fmt: str | None = None,
    stream=None,
) -> None:
    """Configure root logging once. Safe to call multiple times.

    Args:
        level: Root log level. Defaults to ``AI_WEB_LOG_LEVEL`` env
            var, falling back to ``INFO``.
        noisy_level: Cap noisy third-party loggers. Defaults to
            ``AI_WEB_LOG_NOISY_LEVEL`` env var, falling back to
            ``WARNING``.
        fmt: Log record format. Defaults to ``AI_WEB_LOG_FORMAT``.
        stream: Output stream, defaults to ``sys.stderr``.
    """
    global _initialised
    if _initialised:
        return

    level = level if level is not None else os.getenv("AI_WEB_LOG_LEVEL", "INFO")
    noisy_level = (
        noisy_level
        if noisy_level is not None
        else os.getenv("AI_WEB_LOG_NOISY_LEVEL", "WARNING")
    )
    fmt = fmt if fmt is not None else os.getenv("AI_WEB_LOG_FORMAT", DEFAULT_FORMAT)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    # Always-on: strip ANSI colour codes so ERROR / WARNING never
    # render as red text, regardless of what library emitted them.
    handler.addFilter(_StripAnsiFilter())

    root = logging.getLogger(ROOT_LOGGER_NAME)
    # remove any pre-existing handlers (pytest, IDE debuggers, etc.)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(_coerce_level(level))

    for name, lvl in NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(_coerce_level(noisy_level))

    _initialised = True


def log_event(logger_name: str, message: str, **fields) -> None:
    """Emit a structured key event under ``agent.supervisor.<phase>``.

    Args:
        logger_name: One of the ``KEY_EVENT_LOGGERS`` entries.
        message: Human-readable message.
        **fields: Extra context rendered after the message.

    Multi-line values (e.g. JSON for an LLM payload) are emitted on
    their own lines, indented to line up with the first column of the
    main message so the operator can read the full content even when
    it is several KB long.
    """
    log = logging.getLogger(logger_name)
    if not log.isEnabledFor(logging.INFO):
        return
    if not fields:
        log.info(message)
        return
    # The format string is "[time] [logger] LEVEL | message". Indent
    # any multi-line field value to line up after the "| message" so
    # the column aligns with the first character after the pipe.
    head = f" | {message}"
    parts = [head]
    for k, v in fields.items():
        rendered = str(v) if v is not None else "None"
        if "\n" in rendered or len(rendered) > 120:
            # Multi-line / huge: emit on its own line, indented 2 spaces.
            body = "\n".join("  " + line for line in rendered.splitlines())
            parts.append(f"  {k}=\n{body}")
        else:
            parts.append(f"{k}={rendered}")
    log.info("\n".join(parts))


def _coerce_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    return logging.getLevelName(str(value).upper())


__all__ = [
    "setup_logging",
    "log_event",
    "NOISY_LOGGERS",
    "KEY_EVENT_LOGGERS",
    "DEFAULT_FORMAT",
]
