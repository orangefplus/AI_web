"""Sub-agent package.

Each sub-agent implements :class:`agents.subagents.base.Subagent` and
contributes a small, focused set of LangChain tools. The supervisor
loads them by name to keep ``agents/subagents/`` easily extensible.
"""
from __future__ import annotations

from typing import Any

from .api_agent import APIAgent
from .base import Subagent, default_system_prompt
from .browser_agent import BrowserAgent
from .extractor_agent import ExtractorAgent
from .verifier_agent import VerifierAgent, check_step_result


def build_subagents(llm: Any) -> dict[str, Subagent]:
    """Construct one instance of every sub-agent, keyed by name."""
    return {
        "api": APIAgent(llm),
        "browser": BrowserAgent(llm),
        "extractor": ExtractorAgent(llm),
        "verifier": VerifierAgent(llm),
    }


__all__ = [
    "APIAgent",
    "BrowserAgent",
    "ExtractorAgent",
    "VerifierAgent",
    "Subagent",
    "default_system_prompt",
    "build_subagents",
    "check_step_result",
]
