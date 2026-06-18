"""VerifyAgent (Verify 智能体) — Layer-4 specialist for result verification.

The Operation Master dispatches check / verify operations here.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.runnables import Runnable

from agents.prompts import VERIFY_SPECIALIST_PROMPT
from agents.subagents.base import Subagent, extract_latest_tool_result
from agents.task_planner import Step
from tools import browser_get_page_info, browser_screenshot
from tools._tooling import tool as _tool_decorator


@_tool_decorator
def check_file_exists(path: str) -> dict:
    """Return ``{"exists": bool, "size": int}`` for a local file."""
    if not path:
        return {"exists": False, "size": 0}
    if not os.path.exists(path):
        return {"exists": False, "size": 0}
    return {"exists": True, "size": os.path.getsize(path)}


def check_step_result(step: Step, result: dict) -> dict:
    """Deterministic post-check, used before falling back to the LLM.

    The function inspects ``result['data']`` and ``expected_output``
    from the step. For each ``(key, spec)`` pair it makes a best
    effort to assert the spec.

    Returns a dict with:
        - ``ok``: bool, True iff every assertion passed.
        - ``issues``: list of human-readable failures.
    """
    issues: list[str] = []
    data = (result or {}).get("data") or {}
    expected = step.expected_output or {}
    for key, spec in expected.items():
        if key not in data:
            issues.append(f"missing key: {key}")
            continue
        value = data[key]
        if isinstance(spec, str) and spec.startswith("list"):
            if not isinstance(value, list):
                issues.append(f"{key} should be a list, got {type(value).__name__}")
                continue
            if "length=" in spec:
                try:
                    n = int(spec.split("length=")[1].rstrip("]"))
                    if len(value) < n:
                        issues.append(f"{key} has {len(value)} items, expected >= {n}")
                except (ValueError, IndexError):
                    pass
        if spec == "bool" and not isinstance(value, bool):
            issues.append(f"{key} should be bool, got {type(value).__name__}")
    hard_issues = [i for i in issues if "missing key" not in i]
    return {"ok": not hard_issues, "issues": issues}


class VerifyAgent(Subagent):
    """Specialist for validating prior step outputs."""

    name = "verify"
    description = "verification / sanity check specialist"

    def __init__(self, llm: Any) -> None:
        super().__init__(
            llm,
            tools=[check_file_exists, browser_screenshot, browser_get_page_info],
        )

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=VERIFY_SPECIALIST_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in ("last_result", "expected_output", "expected_signals", "pdf_paths")
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """See :func:`agents.subagents.base.extract_latest_tool_result`."""
        return extract_latest_tool_result(response)


__all__ = ["VerifyAgent", "check_step_result"]
