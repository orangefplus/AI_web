"""Sub-agent package — Layer 4 of the 4-layer multi-agent system.

The Operation Master (Layer 3) dispatches one of these specialists
per step. Each specialist holds a *narrow* tool set so the LLM
cannot drift into other categories.

Layer-4 specialists
-------------------
    TabAgent      - tab management
    ClickAgent    - click / type / scroll / keyboard
    ObserveAgent  - read-only observation
    ExtractAgent  - structured extraction
    VerifyAgent   - result verification

Legacy (backward-compat) exports
--------------------------------
    BrowserAgent, APIAgent, ExtractorAgent
    These remain importable for the older single-agent entry point
    in :mod:`agents.agent` and for any external caller that still
    builds an ad-hoc agent by hand.
"""
from __future__ import annotations

from typing import Any

from .api_agent import APIAgent
from .base import Subagent, default_system_prompt
from .browser_agent import BrowserAgent
from .click_agent import ClickAgent
from .extract_agent import ExtractAgent, extract_pdf_text
from .observe_agent import ObserveAgent
from .tab_agent import TabAgent
from .verifier_agent import VerifyAgent, check_step_result

# Legacy alias: old code (and the original ``ExtractorAgent`` name)
# still resolves to the new extract specialist.
ExtractorAgent = ExtractAgent


def build_specialists(llm: Any) -> dict[str, Subagent]:
    """Return the five Layer-4 specialists, keyed by assignee name."""
    return {
        "tab": TabAgent(llm),
        "click": ClickAgent(llm),
        "observe": ObserveAgent(llm),
        "extract": ExtractAgent(llm),
        "verify": VerifyAgent(llm),
    }


def build_subagents(llm: Any) -> dict[str, Subagent]:
    """Backwards-compatible alias for :func:`build_specialists`.

    Older workflow modules and tests still call
    ``build_subagents(llm)`` and expect the same dict. We now return
    the five Operation-Master-dispatched specialists.
    """
    return build_specialists(llm)


__all__ = [
    # Layer-4 specialists
    "TabAgent",
    "ClickAgent",
    "ObserveAgent",
    "ExtractAgent",
    "VerifyAgent",
    # Legacy
    "APIAgent",
    "BrowserAgent",
    "ExtractorAgent",
    # Base
    "Subagent",
    "default_system_prompt",
    # Builders
    "build_specialists",
    "build_subagents",
    # Helpers
    "check_step_result",
    "extract_pdf_text",
]
