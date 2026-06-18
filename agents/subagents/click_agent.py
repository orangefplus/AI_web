"""ClickAgent (Click 智能体) — Layer-4 specialist for interactions.

The Operation Master dispatches click/input/scroll/keyboard operations here.
"""
from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from agents.prompts import CLICK_SPECIALIST_PROMPT
from agents.subagents.base import Subagent, extract_latest_tool_result
from tools import (
    browser_click_xy,
    browser_dispatch_key,
    browser_dismiss_overlay,
    browser_fill_input,
    browser_handle_dialog,
    browser_navigate_to_link,
    browser_press_key,
    browser_scroll,
    browser_type_text,
    browser_upload_file,
)


class ClickAgent(Subagent):
    """Specialist for click / type / scroll / keyboard / upload."""

    name = "click"
    description = "interaction specialist (click / type / scroll / keyboard)"

    def __init__(self, llm: Any) -> None:
        super().__init__(
            llm,
            tools=[
                browser_click_xy,
                # Navigate-by-link is the *click-by-text* alternative
                # for the vision LLM when it cannot estimate exact
                # coordinates from the screenshot.
                browser_navigate_to_link,
                browser_type_text,
                browser_fill_input,
                browser_press_key,
                browser_dispatch_key,
                browser_scroll,
                browser_upload_file,
                # Modals / dialogs are intrinsically interaction-time events
                browser_dismiss_overlay,
                browser_handle_dialog,
            ],
        )

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=CLICK_SPECIALIST_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in (
                "target_xy", "input_text", "selector", "key_combo",
                "scroll_target", "upload_path",
            )
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """See :func:`agents.subagents.base.extract_latest_tool_result`."""
        return extract_latest_tool_result(response)


__all__ = ["ClickAgent"]
