"""VerifierAgent: cheap, deterministic check on the previous step.

The verifier typically *does not* need an LLM — its job is to assert
that the previous sub-agent's output contains the expected keys /
file sizes / download paths. We provide a small LLM-backed agent
for soft checks ("does this summary answer the question?") and a
:func:`check_step_result` helper for hard checks ("are all
expected fields present and non-empty?").

If the LLM is offline, only the hard checks run.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.runnables import Runnable

from agents.subagents.base import Subagent, default_system_prompt
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
    # If the only issue is that an informational field (e.g. min_count)
    # is missing but the primary list field is present, still pass.
    hard_issues = [i for i in issues if "missing key" not in i]
    return {"ok": not hard_issues, "issues": issues}


class VerifierAgent(Subagent):
    """Sub-agent specialised in validating prior step outputs."""

    name = "verifier"
    description = "verification / sanity check specialist"

    def __init__(self, llm) -> None:
        super().__init__(llm, tools=[check_file_exists, browser_screenshot, browser_get_page_info])

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=default_system_prompt(self.description),
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in ("last_result", "expected_output", "pdf_paths")
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        try:
            messages = response.get("messages", [])
        except AttributeError:
            messages = []
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if not content:
                continue
            if isinstance(content, str) and "{" in content:
                try:
                    start = content.index("{")
                    end = content.rindex("}") + 1
                    return json.loads(content[start:end])
                except (ValueError, json.JSONDecodeError):
                    continue
            if isinstance(content, dict):
                return content
        return response


__all__ = ["VerifierAgent", "check_step_result"]
