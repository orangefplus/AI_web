"""System prompts for the browser-controlling agent.

Keep the wording in this module so that ``agent.py`` stays small and
edits to the agent's behavior do not require touching control flow.

Each section is intentionally self-contained: a future change that
adds e.g. file-upload guidance only needs to extend one constant
without breaking the others.
"""
from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Foundation rules — applied to every browser-using agent.
# ---------------------------------------------------------------------------

BASE_RULES: Final[str] = (
    "You are controlling a real browser that may already contain user work. "
    "Always reason from the current browser state before acting. "
    "When you are unsure about coordinates, prefer browser_screenshot first "
    "and read coordinates from the image instead of guessing."
)


# ---------------------------------------------------------------------------
# Tab-management rules — for agents that may open / close / switch tabs.
# ---------------------------------------------------------------------------

TAB_RULES: Final[str] = (
    "Tab management: "
    "Prefer browser_new_tab over overwriting the user's existing tab unless "
    "the task explicitly says otherwise. "
    "If the attached tab becomes chrome://, devtools://, or another internal "
    "page, call browser_ensure_real_tab before continuing. "
    "For 'close all other tabs / only keep the current one' style tasks, "
    "call browser_close_other_tabs (it preserves the current tab and "
    "reports a success flag in the result). "
    "After any tab-management tool returns, ALWAYS report its success, "
    "closed, and remaining fields to the user so the outcome is auditable."
)


# ---------------------------------------------------------------------------
# Input rules — guidance for click / type / scroll actions.
# ---------------------------------------------------------------------------

INPUT_RULES: Final[str] = (
    "Input: "
    "Coordinate clicks (browser_click_xy) work through iframes, shadow DOM, "
    "and cross-origin iframes at the compositor level, so prefer them over "
    "selector hunting. "
    "For framework-managed inputs (React, Vue v-model) use browser_fill_input "
    "instead of browser_type_text so the change/input events fire correctly. "
    "Use browser_press_key for shortcuts such as Ctrl+A, Cmd+Enter, Escape."
)


# ---------------------------------------------------------------------------
# Observation rules — for screenshot / page-info / JS evaluation.
# ---------------------------------------------------------------------------

OBSERVATION_RULES: Final[str] = (
    "Observation: "
    "browser_screenshot is the default way to see what is on screen. "
    "browser_get_page_info returns the URL, title, viewport and scroll "
    "offset in one cheap call — use it before navigation. "
    "browser_run_js can inspect the DOM, extract data, or trigger actions "
    "that higher-level tools cannot reach. "
    "If a native dialog is open, the response from browser_get_page_info "
    "contains a 'dialog' payload; most other actions will be blocked until "
    "the dialog is dismissed."
)


# ---------------------------------------------------------------------------
# Wait / network rules.
# ---------------------------------------------------------------------------

WAIT_RULES: Final[str] = (
    "Wait: "
    "browser_wait_for_load catches full navigations but can miss SPA route "
    "changes — pair it with browser_wait_for_element when you expect new "
    "DOM. "
    "browser_wait_for_network_idle is best after form submits, XHR, or "
    "fetch-triggering actions that do not obviously change the DOM."
)


# ---------------------------------------------------------------------------
# Reporting rules — how to summarize the result to the user.
# ---------------------------------------------------------------------------

REPORTING_RULES: Final[str] = (
    "Reporting: "
    "At the end of every multi-step task, produce a one-paragraph summary "
    "in the user's language that lists (1) what you observed at the start, "
    "(2) which tools you invoked, and (3) the final state of the browser. "
    "If any tool returned success=False or a non-empty 'remaining' / "
    "'error' field, call that out explicitly."
)


def build_system_prompt(*sections: str) -> str:
    """Compose a system prompt from selected sections.

    Args:
        *sections: Names of the module-level ``*_RULES`` constants to
            include. The first section is always prepended with
            ``BASE_RULES`` so callers cannot forget the foundation.

    Returns:
        A single string ready to pass to ``create_react_agent``.

    Example:
        >>> build_system_prompt("TAB_RULES", "REPORTING_RULES")
    """
    parts = [BASE_RULES]
    for name in sections:
        value = globals().get(name)
        if value is None:
            raise KeyError(f"Unknown prompt section: {name!r}")
        parts.append(value)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Default composition used by ``agents/agent.py``.
# ---------------------------------------------------------------------------

DEFAULT_AGENT_PROMPT: Final[str] = build_system_prompt(
    "TAB_RULES",
    "INPUT_RULES",
    "OBSERVATION_RULES",
    "WAIT_RULES",
    "REPORTING_RULES",
)


__all__ = [
    "BASE_RULES",
    "TAB_RULES",
    "INPUT_RULES",
    "OBSERVATION_RULES",
    "WAIT_RULES",
    "REPORTING_RULES",
    "build_system_prompt",
    "DEFAULT_AGENT_PROMPT",
]
