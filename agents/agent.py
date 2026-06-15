"""LangChain example that first inspects the already-open browser state.

Run from the project root:

    python agents/agent.py

Prerequisites:
    1. Install ``browser-harness`` and keep Chrome attached/already open.
    2. Install ``langchain-openai`` and ``langgraph``.
    3. Configure the model settings in ``config/config.py``.

This file is intentionally thin: the system prompt lives in
``agents.prompts`` and the browser-harness wiring lives in
``tools``. The agent's job is just to glue an LLM to a set of
browser tools and hand it the current browser state.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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


def read_current_browser_state() -> dict[str, Any]:
    """Read the current browser state before handing control to the agent."""
    with BrowserSession() as session:
        snapshot = session.snapshot()
    return snapshot.to_dict()


def build_agent(prompt: str | None = None):
    """Create a LangChain agent bound to the browser tools.

    Args:
        prompt: Optional override of the default system prompt. When
            ``None`` the value of ``prompts.DEFAULT_AGENT_PROMPT`` is
            used.
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


def run_demo(task: str) -> dict[str, Any]:
    """Inspect browser state first, then let the agent continue the task."""
    state = read_current_browser_state()
    agent = build_agent()

    prompt = (
        "I need you to continue from the browser's current state.\n\n"
        "Current browser snapshot:\n"
        f"{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        f"Task: {task}"
    )

    return agent.invoke({"messages": [("user", prompt)]})


if __name__ == "__main__":
    result = run_demo("仅保留当前活动窗口，删掉其余的窗口")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
