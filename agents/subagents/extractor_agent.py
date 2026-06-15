"""ExtractorAgent: pulls structured data out of messy inputs.

Used after either an API call (raw JSON / XML) or a browser
operation (DOM / screenshot) to convert it into the
``expected_output`` shape the supervisor expects. The default
implementation is the same ReAct pattern; domain workflows can
subclass it and override :meth:`extract_from_path` to provide
deterministic extraction (e.g. pdfplumber for PDFs).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.runnables import Runnable

from agents.subagents.base import Subagent, default_system_prompt
from tools import browser_get_page_info, browser_run_js
from tools._tooling import tool as _tool_decorator


@_tool_decorator
def extract_pdf_text(pdf_path: str, max_pages: int = 3) -> str:
    """Read the first ``max_pages`` of a PDF file and return the text.

    Use this when you have a downloaded PDF on disk and need its
    content for summarisation. Returns concatenated page text.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:  # pragma: no cover
        return f"[pdfplumber not installed; cannot read {pdf_path}]"
    text_parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text_parts.append(page.extract_text() or "")
    except Exception as exc:  # pragma: no cover
        return f"[error reading {pdf_path}: {exc}]"
    return "\n\n".join(text_parts)


class ExtractorAgent(Subagent):
    """Sub-agent specialised in structured data extraction."""

    name = "extractor"
    description = "information extraction specialist (DOM / PDF / OCR)"

    def __init__(self, llm) -> None:
        super().__init__(llm, tools=[browser_get_page_info, browser_run_js, extract_pdf_text])

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
            for k in ("raw_api_response", "pdf_path", "pdf_paths", "raw_page_html", "expected_output")
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


__all__ = ["ExtractorAgent", "extract_pdf_text"]
