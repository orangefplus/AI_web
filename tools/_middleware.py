"""Observability middleware for the multi-agent system.

Every LLM call, LangGraph node, and tool invocation is logged through
this module so an operator can see exactly what the supervisor is
doing at any moment:

  agent.middleware.tool   -> tool_call / tool_result
  agent.middleware.llm    -> llm_call  / llm_reply
  agent.middleware.node   -> node_in   / node_out

The module is intentionally framework-agnostic at the call sites:
:func:`wrap_tool` returns a drop-in replacement for any LangChain
tool, :func:`wrap_llm` wraps any Runnable-style LLM, and
:func:`wrap_node` wraps a LangGraph node callable.

Usage in :mod:`agents.supervisor`::

    from tools._middleware import wrap_node, wrap_specialist_tools

    graph.add_node("refine", wrap_node("refine", refine_node))
    ctx["specialists"] = wrap_specialist_tools(build_specialists(llm))
"""
from __future__ import annotations

import functools
import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

from tools._logging import log_event, setup_logging


_NODE_LOG = "agent.middleware.node"
_TOOL_LOG = "agent.middleware.tool"
_LLM_LOG = "agent.middleware.llm"

# By default we DO NOT truncate LLM / tool payloads — the user wants to
# see the full content during debugging.  Set this env var to a positive
# integer to re-enable a character cap (e.g. for production logs).
import os as _os
_TRUNCATE_LIMIT: int = int(_os.environ.get("AGENT_LOG_TRUNCATE", "0") or "0")


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _msg_to_dict(m: Any) -> Any:
    """Best-effort serialiser for LangChain messages."""
    if hasattr(m, "model_dump"):
        try:
            return m.model_dump()
        except Exception:
            pass
    if hasattr(m, "to_json") and callable(m.to_json):
        try:
            return json.loads(m.to_json())
        except Exception:
            pass
    if hasattr(m, "content"):
        return {"type": type(m).__name__, "content": m.content}
    return repr(m)


def _short(value: Any, limit: int = 0) -> str:
    """Format ``value`` for log lines.

    ``limit`` <= 0 means "no truncation" (the default for this project —
    the operator wants to see the full payload).  When ``limit`` > 0 we
    truncate to that many characters and append a marker showing how
    many characters were dropped.

    Newlines are preserved so JSON / prompts stay readable; they are
    just indented with the log key.
    """
    if value is None:
        return "None"
    # LangChain messages / BaseMessage lists
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "content"):
        try:
            value = [_msg_to_dict(m) for m in value]
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        s = str(value)
    elif isinstance(value, (list, tuple)):
        try:
            s = json.dumps(list(value), ensure_ascii=False, default=str, indent=0)
        except Exception:
            s = repr(value)
    elif isinstance(value, dict):
        try:
            s = json.dumps(value, ensure_ascii=False, default=str, indent=0)
        except Exception:
            s = repr(value)
    else:
        # LangChain message / object with .content
        if hasattr(value, "content"):
            s = getattr(value, "content", str(value))
        else:
            s = repr(value)
    if limit and limit > 0 and len(s) > limit:
        s = s[:limit] + "...(+" + str(len(s) - limit) + ")"
    return s


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------

def wrap_node(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a LangGraph node so its entry/exit is logged.

    The wrapped function is otherwise identical — same args, same return
    type — so it can be passed straight to ``graph.add_node(name, ...)``.
    """

    @functools.wraps(fn)
    def _wrapped(state: dict) -> dict:
        setup_logging()
        node_id = uuid.uuid4().hex[:6]
        log_event(
            _NODE_LOG,
            f"NODE_IN  [{name}] (id={node_id})",
            state_keys=",".join(sorted(state.keys())) if isinstance(state, dict) else type(state).__name__,
            step_idx=state.get("current_step_idx") if isinstance(state, dict) else None,
        )
        start = time.monotonic()
        try:
            out = fn(state)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            summary = ""
            if isinstance(out, dict):
                # Surface the most relevant slices so a single log line
                # tells the operator *what changed*.
                if "error" in out and out["error"]:
                    summary = "error=" + _short(out["error"], _TRUNCATE_LIMIT)
                elif "pending_operation" in out and out["pending_operation"]:
                    summary = "op=" + _short(out["pending_operation"].get("assignee"))
                elif "final_answer" in out and out["final_answer"]:
                    summary = "answer=" + _short(out["final_answer"], 80 if _TRUNCATE_LIMIT > 80 else 0)
                else:
                    for k in ("refined", "intent", "plan", "subagent_history",
                              "direction_history", "operation_history",
                              "react_history"):
                        if k in out:
                            v = out[k]
                            if isinstance(v, list):
                                summary = f"{k}+={len(v)}"
                                break
            log_event(
                _NODE_LOG,
                f"NODE_OUT [{name}] (id={node_id}) {elapsed_ms}ms",
                delta=summary,
            )
            return out
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log_event(
                _NODE_LOG,
                f"NODE_ERR [{name}] (id={node_id}) {elapsed_ms}ms",
                error=_short(exc, _TRUNCATE_LIMIT),
            )
            raise

    _wrapped.__wrapped_node__ = name  # type: ignore[attr-defined]
    return _wrapped


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------

def _wrap_callable(label: str, description: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a wrapped version of ``fn`` that logs entry/exit."""
    call_counter = {"n": 0}

    def _wrapper(*args, **kwargs):
        setup_logging()
        call_counter["n"] += 1
        call_id = f"{uuid.uuid4().hex[:4]}{call_counter['n']:02d}"
        arg_repr = _short({"args": args, "kwargs": kwargs}, _TRUNCATE_LIMIT)
        log_event(
            _TOOL_LOG,
            f"TOOL_CALL [{label}] (id={call_id})",
            args=arg_repr,
            desc=description,
        )
        start = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log_event(
                _TOOL_LOG,
                f"TOOL_OK   [{label}] (id={call_id}) {elapsed_ms}ms",
                result=_short(result, _TRUNCATE_LIMIT),
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log_event(
                _TOOL_LOG,
                f"TOOL_ERR  [{label}] (id={call_id}) {elapsed_ms}ms",
                error=_short(exc, _TRUNCATE_LIMIT),
            )
            raise

    _wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    _wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return _wrapper


def wrap_tool(tool: Any) -> Any:
    """Wrap a LangChain tool so its calls/results are logged.

    The strategy is to wrap the *underlying* sync / async callable the
    tool dispatches to (``tool.func`` / ``tool.coroutine``). LangChain's
    ``invoke`` / ``run`` end up calling those, so this works for every
    flavour of tool (StructuredTool, BaseTool, function-decorated with
    ``@tool``) without needing to monkey-patch pydantic-protected
    attributes.
    """
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    description = (getattr(tool, "description", "") or "").splitlines()[0][:80]

    wrapped = False
    # 1) The structured / decorated path: tool.func is a regular Python
    #    function that LangChain invokes with kwargs.
    inner = getattr(tool, "func", None)
    if callable(inner) and not getattr(inner, "__is_middleware_wrapped__", False):
        try:
            new_func = _wrap_callable(name, description, inner)
            new_func.__is_middleware_wrapped__ = True  # type: ignore[attr-defined]
            object.__setattr__(tool, "func", new_func)
            wrapped = True
        except Exception:
            pass

    # 2) The async path: tool.coroutine.
    inner_coro = getattr(tool, "coroutine", None)
    if callable(inner_coro) and not getattr(inner_coro, "__is_middleware_wrapped__", False):
        try:
            new_coro = _wrap_callable(name, description, inner_coro)
            new_coro.__is_middleware_wrapped__ = True  # type: ignore[attr-defined]
            object.__setattr__(tool, "coroutine", new_coro)
            wrapped = True
        except Exception:
            pass

    # 3) Plain-function tool: it's a regular Python function — wrap in place.
    if not wrapped and callable(tool) and not getattr(tool, "__is_middleware_wrapped__", False):
        try:
            new_fn = _wrap_callable(name, description, tool)
            new_fn.__is_middleware_wrapped__ = True  # type: ignore[attr-defined]
            return new_fn
        except Exception:
            return tool

    return tool


def wrap_specialist_tools(specialists: dict) -> dict:
    """Wrap every tool exposed by every specialist in ``specialists``.

    Each specialist is a :class:`Subagent` with a ``tools`` list. This
    function in-place replaces each tool's ``invoke`` / ``run`` with
    the logging wrapper so the supervisor's sub-agent loop emits
    TOOL_CALL / TOOL_RESULT lines.
    """
    for sub in specialists.values():
        tools = getattr(sub, "tools", None) or []
        for t in tools:
            wrap_tool(t)
    return specialists


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

def wrap_llm(llm: Any, label: str = "llm") -> Any:
    """Wrap a Runnable-style LLM so its invocations are logged.

    The wrapper intercepts ``invoke`` / ``ainvoke`` (and ``stream`` if
    present). The logged content is the LLM's *input* (truncated) and
    the *output* type so the operator can verify the LLM was actually
    contacted and not short-circuited by a heuristic fallback.
    """
    label = label or getattr(llm, "model_name", "llm")

    if hasattr(llm, "invoke"):
        original_invoke = llm.invoke

        def _wrapped_invoke(input, *a, **kw):
            setup_logging()
            call_id = uuid.uuid4().hex[:6]
            log_event(
                _LLM_LOG,
                f"LLM_CALL  [{label}] (id={call_id})",
                input=_short(input, _TRUNCATE_LIMIT),
            )
            start = time.monotonic()
            try:
                result = original_invoke(input, *a, **kw)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                content = getattr(result, "content", str(result))
                log_event(
                    _LLM_LOG,
                    f"LLM_REPLY [{label}] (id={call_id}) {elapsed_ms}ms",
                    content=_short(content, _TRUNCATE_LIMIT),
                )
                return result
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                log_event(
                    _LLM_LOG,
                    f"LLM_ERR   [{label}] (id={call_id}) {elapsed_ms}ms",
                    error=_short(exc, _TRUNCATE_LIMIT),
                )
                raise

        llm.invoke = _wrapped_invoke

    if hasattr(llm, "ainvoke"):
        original_ainvoke = llm.ainvoke

        async def _wrapped_ainvoke(input, *a, **kw):
            setup_logging()
            call_id = uuid.uuid4().hex[:6]
            log_event(
                _LLM_LOG,
                f"LLM_CALL  [{label}] (id={call_id}) async",
                input=_short(input, _TRUNCATE_LIMIT),
            )
            start = time.monotonic()
            try:
                result = await original_ainvoke(input, *a, **kw)
                elapsed_ms = int((time.monotonic() - start) * 1000)
                content = getattr(result, "content", str(result))
                log_event(
                    _LLM_LOG,
                    f"LLM_REPLY [{label}] (id={call_id}) {elapsed_ms}ms",
                    content=_short(content, _TRUNCATE_LIMIT),
                )
                return result
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                log_event(
                    _LLM_LOG,
                    f"LLM_ERR   [{label}] (id={call_id}) {elapsed_ms}ms",
                    error=_short(exc, _TRUNCATE_LIMIT),
                )
                raise

        llm.ainvoke = _wrapped_ainvoke

    return llm


__all__ = ["wrap_node", "wrap_tool", "wrap_llm", "wrap_specialist_tools"]
