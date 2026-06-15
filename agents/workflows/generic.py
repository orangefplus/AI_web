"""Generic multi-agent workflow.

Provides a single function :func:`run_generic` for any user input.
Useful for quick experimentation and for callers that just want
"give me the supervisor's result for this sentence".
"""
from __future__ import annotations

from typing import Any

from agents.supervisor import build_supervisor
from tools._logging import setup_logging


def run_generic(user_input: str) -> dict[str, Any]:
    """Run the supervisor on free-form ``user_input``."""
    setup_logging()
    app = build_supervisor()
    final = app.invoke({"user_input": user_input, "iteration_count": 0})
    final.setdefault("user_input", user_input)
    return final


__all__ = ["run_generic"]
