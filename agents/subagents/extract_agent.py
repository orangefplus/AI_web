"""ExtractAgent (Extract 智能体) — Layer-4 specialist for structured extraction.

The Operation Master dispatches PDF / HTML / JSON extraction here.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.runnables import Runnable

from agents.prompts import EXTRACT_SPECIALIST_PROMPT
from agents.subagents.base import Subagent, extract_latest_tool_result
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


class ExtractAgent(Subagent):
    """Specialist for pulling structured data from raw inputs."""

    name = "extract"
    description = "extraction specialist (PDF / HTML / JSON)"

    def __init__(self, llm: Any) -> None:
        super().__init__(
            llm,
            tools=[extract_pdf_text, browser_run_js, browser_get_page_info],
        )

    def build(self, scratchpad: dict) -> Runnable:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=EXTRACT_SPECIALIST_PROMPT,
        )

    def _scratchpad_inputs(self, scratchpad: dict) -> dict:
        return {
            k: scratchpad[k]
            for k in (
                "pdf_path", "pdf_paths", "raw_page_html",
                "raw_api_response", "expected_output",
            )
            if k in scratchpad
        }

    def _extract_data(self, response: Any) -> Any:
        """See :func:`agents.subagents.base.extract_latest_tool_result`.

        Extract tools return JSON dicts (PDF text, page JSON, etc.)
        and the latest call is the one the agent committed to, so the
        shared helper is sufficient.
        """
        return extract_latest_tool_result(response)


__all__ = ["ExtractAgent", "extract_pdf_text"]
