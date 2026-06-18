"""BrowserAgent: drives the real browser via CDP tools.

Gets a **restricted** subset of the 21-tool browser surface — only
the coordinate/observation/screenshot/download tools a human would
actually use — plus the PDF download helpers added in
:mod:`tools.download`. Selector-based tools (``browser_fill_input``,
``browser_dispatch_key``, ``browser_upload_file``, ``browser_http_get``,
``browser_wait_for_element``) are intentionally **excluded** from this
agent so the LLM cannot fall back to "DOM querying" instead of clicking
on real pixels.

A key change vs. the API-first design: this agent is now the
**primary** way of accomplishing paper-download tasks. The prompt
is tuned to behave like a human sitting in front of Chrome:

1. Screenshot first, then read coordinates from the image.
2. Click only on visible UI elements (search boxes, links, buttons).
3. When a PDF is shown in the viewer, find the toolbar download
   icon and click it (do not skip the viewer UI).
4. Fall back to ``browser_download_pdf(0, 0)`` -> printToPDF only
   when no visible click target exists.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import Runnable

from agents.subagents.base import Subagent, default_system_prompt
from agents.prompts import (
    BASE_RULES,
    CLICK_SPECIALIST_PROMPT,
    OBSERVE_SPECIALIST_PROMPT,
    TAB_SPECIALIST_PROMPT,
    build_system_prompt,
)
from tools import get_browser_tools
from tools.download import browser_download_pdf, browser_set_download_dir


# Tools explicitly hidden from BrowserAgent. These either bypass the
# viewport (selectors, raw HTTP) or feel like a programmatic shortcut
# the user does not want. Hiding them forces the agent down the
# "screenshot -> read coordinates -> click_xy" human-like path.
HIDDEN_BROWSER_TOOLS: frozenset[str] = frozenset({
    "browser_fill_input",     # CSS-selector fill
    "browser_dispatch_key",   # CSS-selector key event
    "browser_upload_file",    # CSS-selector file assignment
    "browser_http_get",       # pure-HTTP shortcut (no viewport)
    "browser_wait_for_element",  # CSS-selector wait
})


def _visible_browser_tools() -> list:
    """Return the browser tools the agent is allowed to call.

    Selectors and HTTP shortcuts are filtered out so the LLM cannot
    silently bypass the "real browsing" workflow the user asked for.
    """
    return [t for t in get_browser_tools() if t.name not in HIDDEN_BROWSER_TOOLS]


BROWSER_AGENT_PROMPT: str = build_system_prompt(
    "TAB_RULES",
    "INPUT_RULES",
    "OBSERVATION_RULES",
    "WAIT_RULES",
    "REPORTING_RULES",
) + (
    "\n\nYou are a person at a Chrome window. You see pixels and you have a mouse.\n"
    " - Look at the screenshot before each click; do not guess coordinates.\n"
    " - For text input, click_xy the field to focus, then type_text.\n"
    " - browser_run_js may only READ page text (titles, innerText); it must never write to the filesystem or call fetch() — no Node-style require('fs'), no node:fs, no postMessage tricks.\n"
    " - When a tool needs a file path (screenshot, download dir), pass an empty string / leave the default; the tool picks a writable location and returns the final path. Don't invent C:/... paths yourself.\n"
    " - **Prefer browser_read_page_text** over hand-written querySelectorAll scripts. The screenshot tells you what's on screen; the text dump tells you the same thing as a string. Hand-rolled CSS selectors tend to mis-quote and crash.\n"
    " - **Modal / overlay handling**: a lot of sites (NIH GDC, Elsevier, IEEE Xplore, WebVPN, library gateways, ...) drop a centered consent/Accept/I-agree modal on first load. If the screenshot shows such a modal, call browser_dismiss_overlay() right away — it scans the DOM for an accept-style button, clicks it, and tells you what it did. If that returns found=False, fall back to click_xy on the visible Accept / 同意 / OK button yourself. Do NOT try to click content behind the modal.\n"
    " - If browser_get_page_info returns a ``\"dialog\"`` field (a native alert/confirm), call browser_handle_dialog(accept=True) to dismiss it.\n"
    " - The user wants to watch you click around, not be teleported through a script. Stay on the screen."
)


class BrowserAgent(Subagent):
    """Sub-agent that operates the live browser, human-style."""

    name = "browser"
    description = "browser control specialist (CDP-backed, human-like)"

    def __init__(self, llm) -> None:
        tools = _visible_browser_tools() + [browser_download_pdf, browser_set_download_dir]
        super().__init__(llm, tools=tools)

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=BROWSER_AGENT_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in ("pdf_urls", "target_url", "save_dir", "search_query",
                      "candidate_papers", "scholar_home_shot", "result_shots")
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """Lift structured tool outputs out of the ReAct message history.

        The ReAct agent runs the LLM in a loop and accumulates
        ``HumanMessage / AIMessage / ToolMessage`` records. The
        final ``AIMessage`` is usually plain text (because the LLM is
        not a perfect JSON-emitting function) and the *real* data
        lives in the ``ToolMessage`` content blobs:
        ``{"screenshot_path": "...", "page_text": "...", ...}``.

        Without this scan, the supervisor only sees the LLM's last
        sentence and the verifier reports "page_text length 0" even
        though the tool actually returned 4000 characters.
        """
        try:
            messages = response.get("messages", [])
        except AttributeError:
            messages = []

        merged: dict = {}
        last_text: str = ""

        for msg in messages:
            cls_name = type(msg).__name__
            content = getattr(msg, "content", None)
            if not content:
                continue
            if cls_name == "ToolMessage" and isinstance(content, str):
                # Browser tools return a JSON string with a few
                # well-known fields. Merge the first parseable JSON
                # into the accumulator; non-JSON outputs are kept
                # under a "raw_tool_text" key for debugging.
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    merged.setdefault("raw_tool_text", content)
                    continue
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if v not in (None, "", [], {}) and k not in merged:
                            merged[k] = v
                else:
                    merged.setdefault("raw_tool_text", str(parsed))
            elif isinstance(content, str):
                last_text = content

        # If the LLM's final message was JSON, layer it on top of the
        # tool outputs (without overwriting real values).
        if last_text:
            try:
                parsed_final = json.loads(last_text)
                if isinstance(parsed_final, dict):
                    for k, v in parsed_final.items():
                        merged.setdefault(k, v)
            except json.JSONDecodeError:
                merged.setdefault("text", last_text)

        if merged:
            return merged
        return response


__all__ = ["BrowserAgent", "BROWSER_AGENT_PROMPT"]
