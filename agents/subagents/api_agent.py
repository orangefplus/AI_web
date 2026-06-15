"""APIAgent: prefers REST/Atom APIs over the browser.

The agent is intentionally given *only* ``browser_http_get`` so it
cannot accidentally open a browser tab. This is the design
guidance from every ``domain-skills/<site>/scraping.md`` document:
"Never use the browser for ArXiv / Crossref / OpenAlex / PubMed".

If the LLM is unavailable the agent still works because
``browser_http_get`` is a thin wrapper around the helper. The
agent runnable is created lazily by :meth:`build` so importing
this module does not require ``langgraph``.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.runnables import Runnable

from agents.subagents.base import Subagent, default_system_prompt
from tools import browser_http_get
from tools._tooling import tool as _tool_decorator


# Re-decorate with our @tool so schema is consistent. Cheap no-op
# when already a tool.
http_get_tool = _tool_decorator(
    browser_http_get.invoke  # underlying function not the StructuredTool
) if not hasattr(browser_http_get, "invoke") else browser_http_get


@_tool_decorator
def api_request(url: str, headers: Optional[dict] = None) -> str:
    """Issue an HTTP GET to a public API. Returns the raw body.

    Use this for any structured API (OpenAlex, arXiv Atom, Crossref,
    PubMed E-utilities, Semantic Scholar, etc.). Avoid the browser
    whenever the site has a JSON / XML API.
    """
    return browser_http_get.invoke({"url": url, "headers": headers or {}})


class APIAgent(Subagent):
    """Sub-agent specialised in REST/Atom API calls."""

    name = "api"
    description = "API / HTTP request specialist"

    def __init__(self, llm) -> None:
        super().__init__(llm, tools=[api_request])

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
            for k in ("topic", "count", "pdf_urls", "search_query")
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        # Try to find a JSON object in the assistant's last message.
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


__all__ = ["APIAgent"]
