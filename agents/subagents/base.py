"""Base class shared by all sub-agents.

The supervisor hands each :class:`~agents.task_planner.Step` to the
matching :class:`Subagent`. The contract is intentionally minimal:

- :attr:`name` short identifier (e.g. ``"api"``).
- :attr:`description` one-line purpose used in error messages.
- :attr:`tools` list of LangChain tools the sub-agent can call.
- :meth:`build` returns a LangChain ``Runnable`` (typically a ReAct
  agent) given a scratchpad of state to read/write.
- :meth:`run` is a thin convenience over ``build(scratchpad).invoke``.

The shared scratchpad is a plain dict so sub-agents can pass
structured data between each other without a heavy schema. The
supervisor owns the dict; sub-agents only read the inputs they need
and write their own result under a reserved key.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from agents._error_diagnosis import diagnose, short_label
from agents.task_planner import Step
from tools._logging import log_event


class Subagent(ABC):
    """Abstract base for the 4 specialist sub-agents."""

    name: str = "subagent"
    description: str = ""

    def __init__(self, llm: BaseChatModel, tools: list) -> None:
        self.llm = llm
        self.tools = tools

    @abstractmethod
    def build(self, scratchpad: dict) -> Runnable:
        """Return a LangChain runnable bound to the right tools.

        Implementations should pass the scratchpad in via the runnable
        config so the LLM sees the prior sub-agent results in its
        system prompt.
        """

    def run(self, step: Step, scratchpad: dict) -> dict:
        """Execute ``step`` and return a structured result dict.

        Always returns a dict so the supervisor can merge it into
        the shared scratchpad uniformly. The dict always contains:

        - ``subagent``: this sub-agent's name.
        - ``step_id``: the step that was executed.
        - ``status``: ``"ok"`` / ``"partial"`` / ``"error"``.
        - ``elapsed_ms``: wall-clock execution time.
        - ``data``: arbitrary sub-agent specific output.

        On any uncaught exception the result is returned with
        ``status="error"`` and ``error`` set; the supervisor decides
        whether to retry, fall back, or abort.
        """
        log_event(
            "agent.supervisor.step",
            f"STEP {step.step_id} [{self.name}]: {step.description}",
            action=step.action,
        )
        start = time.monotonic()
        try:
            runnable = self.build(scratchpad)
            user_msg = self._compose_user_message(step, scratchpad)
            # Tenacity retry: 讯飞 one-api 偶发 "Engine Busy" (500) /
            # timeout / 502 — 必须重试多次,否则整个任务挂掉。
            from tenacity import (
                retry, stop_after_attempt, wait_exponential,
                retry_if_exception_type, before_sleep_log,
            )
            _log = logging.getLogger(f"agent.subagent.{self.name}.llm_retry")

            @retry(
                reraise=True,
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=2, min=2, max=20),
                retry=retry_if_exception_type(Exception),
                before_sleep=before_sleep_log(_log, logging.WARNING),
            )
            def _invoke():
                return runnable.invoke({
                    "messages": [{"role": "user", "content": user_msg}],
                })

            response = _invoke()
            data = self._extract_data(response)
            elapsed = int((time.monotonic() - start) * 1000)
            log_event(
                "agent.supervisor.result",
                f"STEP {step.step_id} OK in {elapsed}ms",
                subagent=self.name,
            )
            return {
                "subagent": self.name,
                "step_id": step.step_id,
                "status": "ok",
                "elapsed_ms": elapsed,
                "data": data,
            }
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            diag = diagnose(str(exc))
            logging.getLogger("agent.supervisor.error").info(
                "STEP %d ERR [%s/%s] conf=%.2f: %s",
                step.step_id, diag.category, short_label(diag.category),
                diag.confidence, exc,
            )
            return {
                "subagent": self.name,
                "step_id": step.step_id,
                "status": "error",
                "elapsed_ms": elapsed,
                "error": str(exc),
                "data": None,
                # NEW: structured diagnosis so the ReAct Master can
                # decide what to do next (retry, switch assignee, ask
                # the user, etc.) without re-parsing the raw exception.
                "error_category": diag.category,
                "error_short": diag.short,
                "error_detail": diag.detail,
                "error_recovery_hint": diag.recovery_hint,
                "error_can_retry": diag.can_retry,
                "error_confidence": diag.confidence,
            }

    # -- helpers ---------------------------------------------------------

    def _compose_user_message(self, step: Step, scratchpad: dict) -> str:
        """Build a single user message describing the task + inputs."""
        inputs = self._scratchpad_inputs(scratchpad)
        parts = [f"Task: {step.description}", f"Action: {step.action}"]
        if step.expected_output:
            parts.append(f"Expected output shape: {step.expected_output}")
        if inputs:
            parts.append("Inputs from prior steps: " + str(inputs))
        return "\n".join(parts)

    @abstractmethod
    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        """Return only the keys this sub-agent needs from the scratchpad."""

    @abstractmethod
    def _extract_data(self, response: Any) -> Any:
        """Pull the structured payload out of the LLM runnable response."""


def default_system_prompt(subagent_description: str) -> str:
    """Common system prompt preamble used by every sub-agent."""
    return (
        f"You are the {subagent_description} sub-agent of a multi-agent "
        "browser automation system. Stay focused on your specialty; the "
        "supervisor routes the next step to a different sub-agent when "
        "needed. Use the tools listed in your schema; do not invent "
        "tools that are not provided. When you finish, return a short "
        "JSON object summarizing the outcome."
    )


def extract_latest_tool_result(response: Any) -> Any:
    """Pull the most recent :class:`ToolMessage` payload from a ReAct response.

    ReAct agents return a dict like ``{"messages": [...]}`` where the
    final message is typically an ``AIMessage`` containing a summary
    of the agent's reasoning. The *structured* output we actually
    want is the JSON payload of the last ``ToolMessage``.

    Returns the parsed dict when the tool's content is JSON, the dict
    form when LangChain already parsed it, or the raw response as a
    last resort.
    """
    import json
    try:
        messages = response.get("messages", []) or []
    except AttributeError:
        return response

    for msg in reversed(messages):
        cls_name = type(msg).__name__
        if cls_name != "ToolMessage":
            continue
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                continue
    return response


__all__ = ["Subagent", "default_system_prompt", "extract_latest_tool_result"]
