"""ObserveAgent (Observe 智能体) — Layer-4 specialist for read-only observation.

The Operation Master dispatches screenshot/read/wait operations here.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import Runnable

from agents.prompts import OBSERVE_SPECIALIST_PROMPT
from agents.subagents.base import Subagent, extract_latest_tool_result
from tools import (
    browser_extract_links,
    browser_find_field_by_label,
    browser_get_page_info,
    browser_list_form_fields,
    browser_read_page_text,
    browser_run_js,
    browser_screenshot,
    browser_wait_for_element,
    browser_wait_for_load,
    browser_wait_for_network_idle,
)


class ObserveAgent(Subagent):
    """Specialist for read-only observation.

    Bound tools (no click / no tab mutation / no extraction):
        - browser_screenshot
        - browser_get_page_info
        - browser_read_page_text
        - browser_run_js      (read-only usage enforced by the prompt)
        - browser_extract_links (read-only — returns link table, no click)
        - browser_wait_for_load
        - browser_wait_for_element
        - browser_wait_for_network_idle
    """

    name = "observe"
    description = "read-only observation specialist (screenshot / read / wait)"

    def __init__(self, llm: Any) -> None:
        super().__init__(
            llm,
            tools=[
                browser_screenshot,
                browser_get_page_info,
                browser_read_page_text,
                browser_run_js,
                browser_extract_links,
                browser_wait_for_load,
                browser_wait_for_element,
                browser_wait_for_network_idle,
                # form introspection (read-only — returns descriptors,
                # never mutates the page)
                browser_list_form_fields,
                browser_find_field_by_label,
            ],
        )

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=OBSERVE_SPECIALIST_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in (
                "expected_signals", "screenshot_path", "page_text",
                "url", "title", "scrollY", "viewport",
            )
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """Merge structured tool outputs from the ReAct history.

        Observe tools (screenshot / read_page_text / get_page_info)
        each return JSON-serializable dicts. We want *all* of them in
        one payload so the supervisor sees the full page state, not
        just the last one.

        Falls back to :func:`extract_latest_tool_result` for any
        subclass that only carries a single tool call.
        """
        try:
            messages = response.get("messages", []) or []
        except AttributeError:
            return extract_latest_tool_result(response)

        merged: dict = {}
        for msg in messages:
            if type(msg).__name__ != "ToolMessage":
                continue
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if not content:
                continue
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if v not in (None, "", [], {}) and k not in merged:
                            merged[k] = v
            elif isinstance(content, dict):
                for k, v in content.items():
                    if v not in (None, "", [], {}) and k not in merged:
                        merged[k] = v
        if merged:
            return merged
        return extract_latest_tool_result(response)


__all__ = ["ObserveAgent"]
