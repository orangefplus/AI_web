"""Top-level entry point for the 4-layer multi-agent browser system.

Run from the project root::

    python agents/agent.py

This module is intentionally thin: it bootstraps the browser
daemon, captures the current browser state, and hands control to
:func:`agents.supervisor.run`, which drives the full
*Direction-Master → Prompt-Refiner → Operation-Master → Specialist*
pipeline.

The function :func:`build_simple_agent` is kept for callers that
still want a single ReAct agent bound to the full browser tool set
(no multi-agent coordination). It is used by some legacy tests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BROWSER_HARNESS_SRC = PROJECT_ROOT / "browser-harness" / "src"
if str(BROWSER_HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(BROWSER_HARNESS_SRC))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = [str(BROWSER_HARNESS_SRC)]
if existing_pythonpath:
    pythonpath_parts.append(existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

from config.config import chat_model_name, xf_api_key, xf_chat_base_url
from tools import BrowserSession, get_browser_tools

from .prompts import DEFAULT_AGENT_PROMPT


# ---------------------------------------------------------------------------
# Browser state
# ---------------------------------------------------------------------------

def read_current_browser_state() -> dict[str, Any]:
    """Read the current browser state before handing control to the agent."""
    with BrowserSession() as session:
        snapshot = session.snapshot()
    return snapshot.to_dict()


# ---------------------------------------------------------------------------
# Legacy single-agent entry (kept for back-compat with old tests/scripts)
# ---------------------------------------------------------------------------

def build_agent(prompt: str | None = None):
    """Build a single ReAct agent bound to the full browser tool set.

    This is the *old* entry point. The new 4-layer pipeline lives in
    :func:`agents.supervisor.run` and is what
    :func:`run_demo` invokes by default.
    """
    llm = ChatOpenAI(
        model=chat_model_name,
        api_key=xf_api_key,
        base_url=xf_chat_base_url,
        temperature=0,
    )
    return create_react_agent(
        llm,
        get_browser_tools(),
        prompt=prompt or DEFAULT_AGENT_PROMPT,
    )


# ---------------------------------------------------------------------------
# New 4-layer entry point
# ---------------------------------------------------------------------------

def run_demo(
    task: str,
    *,
    use_supervisor: bool = True,
    return_state: bool = False,
) -> dict[str, Any]:
    """Run a single user request through the multi-agent system.

    Args:
        task: Free-form user request (any language).
        use_supervisor: When ``True`` (default), drive the full
            4-layer supervisor. When ``False``, fall back to the
            single ReAct agent for quick local experiments.
        return_state: When ``True`` the full LangGraph state dict
            is returned (for debugging). When ``False`` only the
            final answer + scratchpad are returned.

    Returns:
        Dict containing at least ``final_answer`` and ``scratchpad``.
    """
    if use_supervisor:
        from .supervisor import run as supervisor_run
        final = supervisor_run(task)
    else:
        state = read_current_browser_state()
        agent = build_agent()
        prompt = (
            "I need you to continue from the browser's current state.\n\n"
            "Current browser snapshot:\n"
            f"{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Task: {task}"
        )
        final = agent.invoke({"messages": [("user", prompt)]})

    if not return_state and isinstance(final, dict):
        # Compress to the fields callers actually want.
        return {
            "final_answer": final.get("final_answer") or "",
            "scratchpad": final.get("scratchpad") or {},
            "subagent_history": final.get("subagent_history") or [],
            "direction_history": final.get("direction_history") or [],
            "operation_history": final.get("operation_history") or [],
            "refined": final.get("refined") or {},
            "user_input": task,
        }
    return final


if __name__ == "__main__":
    result = run_demo("仅保留当前活动窗口，删掉其余的窗口")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
