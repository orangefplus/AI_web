"""LangChain tools wrapping browser-harness.

This package exposes the atomic browser-harness helpers as LangChain
``@tool``-decorated functions. Three flavors are available:

- :func:`get_browser_tools` — the 20 atomic browser tools (navigation,
  input, observation, wait, network). Bind this to every browser-using
  agent.
- :func:`get_admin_read_tools` — read-only lifecycle tools (doctor,
  profile discovery). Safe to expose to most agents.
- :func:`get_admin_write_tools` — stateful / billable tools (cloud
  browser start/stop, profile sync, daemon restart). Bind explicitly
  when you want the agent to manage its own browser lifecycle.
- :func:`get_all_tools` — everything above.

Quick start::

    from AI_web.tools import setup_browser, get_browser_tools
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    setup_browser()                                # start daemon once
    agent = create_react_agent(
        ChatOpenAI(model="gpt-4o"),
        get_browser_tools(),
        system_message=(
            "Use browser_screenshot first to see the page, then "
            "browser_click_xy. Avoid selector-hunting. Use "
            "browser_ensure_real_tab if the tab goes chrome://."
        ),
    )
    agent.invoke({"messages": [("user", "open https://example.com and tell me the title")]})
"""
from __future__ import annotations

from ._bootstrap import daemon_name, reset_for_tests, setup_browser
from .admin import (
    ADMIN_READ_TOOLS,
    ADMIN_TOOLS,
    ADMIN_WRITE_TOOLS,
    browser_doctor,
    browser_list_cloud_profiles,
    browser_list_local_profiles,
    browser_restart_daemon,
    browser_start_remote_session,
    browser_stop_remote_session,
    browser_sync_local_profile,
)
from .browser import (
    BROWSER_TOOLS,
    browser_click_xy,
    browser_close_other_tabs,
    browser_close_tab,
    browser_dispatch_key,
    browser_dismiss_overlay,
    browser_ensure_real_tab,
    browser_extract_links,
    browser_fill_input,
    browser_get_page_info,
    browser_handle_dialog,
    browser_http_get,
    browser_list_tabs,
    browser_navigate,
    browser_navigate_to_link,
    browser_new_tab,
    browser_press_key,
    browser_read_page_text,
    browser_run_js,
    browser_scroll,
    browser_screenshot,
    browser_switch_tab,
    browser_type_text,
    browser_upload_file,
    browser_wait_for_element,
    browser_wait_for_load,
    browser_wait_for_network_idle,
)
from .decorators import describe, with_verification
from .download import browser_download_pdf, browser_set_download_dir
from .session import BrowserSession, BrowserSnapshot


def get_browser_tools() -> list:
    """The 20 atomic browser tools. Bind these to any browser-using agent."""
    return list(BROWSER_TOOLS)


def get_admin_read_tools() -> list:
    """Read-only admin tools (doctor, list profiles)."""
    return list(ADMIN_READ_TOOLS)


def get_admin_write_tools() -> list:
    """Stateful / billable admin tools. Bind explicitly when needed."""
    return list(ADMIN_WRITE_TOOLS)


def get_admin_tools() -> list:
    """All admin tools (read + write)."""
    return list(ADMIN_TOOLS)


def get_all_tools() -> list:
    """All browser + admin tools. ~25 functions in total."""
    return get_browser_tools() + get_admin_tools()


__all__ = [
    # lifecycle
    "setup_browser",
    "daemon_name",
    "reset_for_tests",
    "BrowserSession",
    "BrowserSnapshot",
    # cross-cutting decorators
    "describe",
    "with_verification",
    # downloads
    "browser_download_pdf",
    "browser_set_download_dir",
    # aggregators
    "get_browser_tools",
    "get_admin_read_tools",
    "get_admin_write_tools",
    "get_admin_tools",
    "get_all_tools",
    # raw lists (advanced)
    "BROWSER_TOOLS",
    "ADMIN_TOOLS",
    "ADMIN_READ_TOOLS",
    "ADMIN_WRITE_TOOLS",
    # individual tools (re-exported for direct binding)
    "browser_navigate",
    "browser_new_tab",
    "browser_list_tabs",
    "browser_switch_tab",
    "browser_close_tab",
    "browser_close_other_tabs",
    "browser_ensure_real_tab",
    "browser_click_xy",
    "browser_type_text",
    "browser_fill_input",
    "browser_press_key",
    "browser_scroll",
    "browser_dispatch_key",
    "browser_upload_file",
    "browser_dismiss_overlay",
    "browser_handle_dialog",
    "browser_read_page_text",
    "browser_screenshot",
    "browser_get_page_info",
    "browser_run_js",
    "browser_wait_for_load",
    "browser_wait_for_element",
    "browser_wait_for_network_idle",
    "browser_http_get",
    "browser_doctor",
    "browser_list_cloud_profiles",
    "browser_list_local_profiles",
    "browser_start_remote_session",
    "browser_stop_remote_session",
    "browser_restart_daemon",
    "browser_sync_local_profile",
]
