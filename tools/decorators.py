"""Reusable decorators for LangChain tools.

This module extracts cross-cutting concerns (verification, retries,
assertion reporting) out of individual tool functions so that
``tools/browser.py`` can stay focused on browser-specific logic.

Each decorator is intentionally framework-agnostic at its core: they
wrap the original callable and return a new callable with the same
signature, so they compose with ``langchain_core.tools.tool``.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def with_verification(
    check: Callable[[Any], bool],
    detail_extractor: Callable[[Any], dict] | None = None,
    max_retries: int = 1,
    retry_delay: float = 0.3,
    fail_marker: str = "success",
) -> Callable[[F], F]:
    """Wrap a tool so the caller can see whether it actually worked.

    The wrapped tool is invoked once. If ``check(result)`` is ``False``,
    the wrapped function is invoked again up to ``max_retries`` times
    with ``retry_delay`` seconds between attempts. The final return
    value is augmented with the following keys (only if missing):

    - ``success``: ``True`` iff ``check`` returned ``True``.
    - ``attempts``: how many times the underlying function ran.
    - ``detail``: extra info from ``detail_extractor`` (optional).

    Args:
        check: Predicate that inspects the tool's return value.
        detail_extractor: Optional callable producing a JSON-friendly
            dict with extra context (e.g. ``{"remaining": [...]}``).
        max_retries: Additional attempts after the first failure.
        retry_delay: Seconds to wait between retries.
        fail_marker: Name of the boolean flag injected into the result.
            Defaults to ``"success"`` for compatibility with
            ``browser_close_other_tabs``.

    Example:
        >>> @tool
        ... @with_verification(check=lambda r: r["ok"], max_retries=2)
        ... def my_tool() -> dict:
        ...     return {"ok": False}
    """

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            attempts = 0
            last_result: Any = None
            for attempt in range(1 + max_retries):
                attempts += 1
                last_result = func(*args, **kwargs)
                if check(last_result):
                    break
                if attempt < max_retries:
                    time.sleep(retry_delay)
            if not isinstance(last_result, dict):
                # Non-dict returns cannot be augmented; return as-is.
                return last_result
            last_result.setdefault(fail_marker, check(last_result))
            last_result.setdefault("attempts", attempts)
            if detail_extractor is not None:
                last_result.setdefault("detail", detail_extractor(last_result))
            return last_result

        # Preserve introspection-friendly attributes.
        wrapper.__name__ = getattr(func, "__name__", "wrapped_tool")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def describe(
    *,
    purpose: str,
    when_to_use: str,
    caveats: str = "",
) -> Callable[[F], F]:
    """Attach a structured description to a tool's docstring.

    LangChain reads the tool's docstring to expose the tool to the LLM.
    A clear, scenario-driven description dramatically improves
    function-calling accuracy. ``describe`` rewrites the function's
    docstring into a consistent three-paragraph format:

    1. ``Purpose`` — one-sentence summary.
    2. ``When to use`` — the kinds of user requests that should
       trigger this tool.
    3. ``Caveats`` — gotchas, common failure modes, and how to recover.

    Args:
        purpose: Single-sentence description of what the tool does.
        when_to_use: Bullet-list-ready sentence about which user
            intents should match.
        caveats: Optional string describing failure modes.
    """

    def decorator(func: F) -> F:
        caveat_block = f"\n\nCaveats:\n    {caveats.strip()}" if caveats else ""
        doc = (
            f"{purpose}\n\n"
            f"When to use:\n    {when_to_use.strip()}"
            f"{caveat_block}"
        )
        func.__doc__ = doc
        return func

    return decorator


__all__ = ["with_verification", "describe"]
