"""TabAgent (Tab 智能体) — Layer-4 specialist for tab management.

The Operation Master dispatches tab-class operations here.
"""
from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from agents.prompts import TAB_SPECIALIST_PROMPT
from agents.subagents.base import Subagent, extract_latest_tool_result
from tools import (
    browser_close_other_tabs,
    browser_close_tab,
    browser_ensure_real_tab,
    browser_list_tabs,
    browser_navigate,
    browser_navigate_to_link,
    browser_new_tab,
    browser_switch_tab,
)


class TabAgent(Subagent):
    """Specialist for tab-related operations only.

    Bound tools (no input / observe / extract):
        - browser_new_tab
        - browser_switch_tab
        - browser_close_tab
        - browser_close_other_tabs
        - browser_ensure_real_tab
        - browser_list_tabs
        - browser_navigate             (in-tab navigation counts as tab ops)
        - browser_navigate_to_link     (text/href match -> navigate; useful
            when the LLM can read the link title from a screenshot but
            cannot estimate exact click coordinates)
    """

    name = "tab"
    description = "tab management specialist (new / switch / close / list / navigate-by-link)"

    def __init__(self, llm: Any) -> None:
        super().__init__(
            llm,
            tools=[
                browser_new_tab,
                browser_switch_tab,
                browser_close_tab,
                browser_close_other_tabs,
                browser_ensure_real_tab,
                browser_list_tabs,
                browser_navigate,
                browser_navigate_to_link,
            ],
        )

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=TAB_SPECIALIST_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in ("target_url", "target_tab_id", "tab_action")
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """See :func:`agents.subagents.base.extract_latest_tool_result`."""
        return extract_latest_tool_result(response)


__all__ = ["TabAgent"]
