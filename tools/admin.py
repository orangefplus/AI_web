"""LangChain tools for browser-harness admin / lifecycle / cloud
operations.

These are deliberately separated from the atomic browser tools in
``browser.py`` because they are stateful, side-effectful, and often
billable (cloud browsers charge per minute). They are useful for
agents that need to:

- Diagnose a broken install (``browser_doctor``).
- Spin up an isolated remote browser for a parallel sub-agent
  (``browser_start_remote_session`` / ``browser_stop_remote_session``).
- Carry local Chrome cookies into a cloud browser
  (``browser_sync_local_profile``).
- Discover which cloud profiles are already logged in
  (``browser_list_cloud_profiles`` / ``browser_list_local_profiles``).

Most agents should not bind all of these; the default ``get_tools()``
exposes only the read-only ones. ``get_all_tools()`` adds the
write/destructive ones.
"""
from __future__ import annotations

import os
from typing import Optional

from browser_harness import admin as _bh_admin

from ._bootstrap import setup_browser
from ._tooling import tool


# ---------------------------------------------------------------------------
# diagnostic
# ---------------------------------------------------------------------------

@tool
def browser_doctor() -> str:
    """Run ``browser-harness --doctor`` and return its text output.

    This is the first diagnostic to run when browser tools are failing
    or behaving unexpectedly. It checks install mode, daemon liveness,
    Chrome detection, active connections, cloud-key presence, and other
    environment issues.

    Returns:
        Plain-text diagnostic output prefixed with the doctor's exit
        code.
    """
    import io
    import contextlib

    setup_browser()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _bh_admin.run_doctor()
    return f"exit_code={rc}\n{buf.getvalue()}"


# ---------------------------------------------------------------------------
# cloud browser session
# ---------------------------------------------------------------------------

@tool
def browser_start_remote_session(
    name: str = "remote",
    profile_name: Optional[str] = None,
    profile_id: Optional[str] = None,
    proxy_country_code: Optional[str] = "us",
    timeout: int = 60,
    enable_recording: bool = False,
) -> dict:
    """Provision a Browser Use cloud browser and attach a daemon to it.

    Use this only when an agent needs an isolated remote browser or a
    logged-in cloud profile. This tool requires
    ``BROWSER_USE_API_KEY`` and can incur per-minute billing until
    ``browser_stop_remote_session`` is called or the timeout expires.
    The returned payload includes ``liveUrl`` so the user can watch the
    session.

    Args:
        name: Logical daemon name used to isolate multiple remote
            sessions in one process.
        profile_name: Cloud profile name to attach. Mutually exclusive
            with ``profile_id``.
        profile_id: Cloud profile UUID to attach. Mutually exclusive
            with ``profile_name``.
        proxy_country_code: ISO2 country code for Browser Use
            residential proxy routing. Pass ``None`` to disable the
            proxy.
        timeout: Auto-stop timeout in minutes.
        enable_recording: Whether Browser Use should record the remote
            session for playback later.

    Returns:
        Full Browser Use browser metadata dict, including ``id``,
        ``cdpUrl``, and ``liveUrl``.
    """
    if not os.environ.get("BROWSER_USE_API_KEY"):
        raise RuntimeError("BROWSER_USE_API_KEY not set — see .env.example")
    if profile_name and profile_id:
        raise ValueError("pass profile_name OR profile_id, not both")
    create_kwargs: dict = {"timeout": timeout, "enableRecording": enable_recording}
    if profile_id:
        create_kwargs["profileId"] = profile_id
    if proxy_country_code is not None:
        create_kwargs["proxyCountryCode"] = proxy_country_code
    setup_browser(name=name)
    browser = _bh_admin.start_remote_daemon(
        name=name, profileName=profile_name, **create_kwargs
    )
    return browser


@tool
def browser_stop_remote_session(name: str = "remote") -> str:
    """Stop a named remote daemon and its backing cloud browser.

    Use this to end billing promptly and persist any profile state from
    the remote browser.

    Args:
        name: Logical daemon name previously passed to
            ``browser_start_remote_session``.

    Returns:
        Short confirmation string naming the stopped remote session.
    """
    if not os.environ.get("BROWSER_USE_API_KEY"):
        raise RuntimeError("BROWSER_USE_API_KEY not set")
    _bh_admin.stop_remote_daemon(name=name)
    return f"stopped {name!r}"


@tool
def browser_restart_daemon(name: Optional[str] = None) -> str:
    """Stop the current or named daemon so the next tool call respawns it.

    Use this when the doctor's output shows a stale browser connection
    or after changing browser-harness helper code.

    Args:
        name: Optional daemon name. Leave empty to restart the default
            current daemon.

    Returns:
        Short confirmation string explaining that the next tool call
        will auto-start a fresh daemon.
    """
    setup_browser()  # make sure env is sane before we tear down
    _bh_admin.restart_daemon(name=name)
    return f"stopped {name or 'default'!r} — next call auto-respawns"


# ---------------------------------------------------------------------------
# profile discovery / sync
# ---------------------------------------------------------------------------

@tool
def browser_list_cloud_profiles() -> list:
    """List cloud profiles under the current ``BROWSER_USE_API_KEY``.

    Use this to discover reusable logged-in Browser Use profiles before
    starting a remote session.

    Returns:
        List of profile dicts. Each entry includes ``id``, ``name``,
        ``userId``, ``cookieDomains``, and ``lastUsedAt``.
    """
    if not os.environ.get("BROWSER_USE_API_KEY"):
        raise RuntimeError("BROWSER_USE_API_KEY not set")
    return _bh_admin.list_cloud_profiles()


@tool
def browser_list_local_profiles() -> list:
    """Detect local browser profiles on this machine.

    This shells out to ``profile-use list --json`` and is mainly useful
    before syncing cookies to a cloud profile.

    Returns:
        List of local profile dicts with fields such as
        ``BrowserName``, ``ProfileName``, ``ProfilePath``, and
        ``DisplayName``.
    """
    return _bh_admin.list_local_profiles()


@tool
def browser_sync_local_profile(
    profile_name: str,
    browser: Optional[str] = None,
    cloud_profile_id: Optional[str] = None,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
) -> str:
    """Sync cookies from a local browser profile into a cloud profile.

    Only cookies are synced, not local storage, extensions, or history.
    This requires both ``profile-use`` and ``BROWSER_USE_API_KEY``.
    Re-run the tool whenever you want to refresh cloud cookies from the
    local profile.

    Args:
        profile_name: Local profile name from
            ``browser_list_local_profiles``.
        browser: Optional browser name to disambiguate profiles with the
            same label across multiple installed browsers.
        cloud_profile_id: Existing cloud profile UUID to update. Leave
            empty to create a new cloud profile.
        include_domains: Optional allowlist of cookie domains to sync.
        exclude_domains: Optional denylist of cookie domains to skip.

    Returns:
        Cloud profile UUID that received the synced cookies.
    """
    if not os.environ.get("BROWSER_USE_API_KEY"):
        raise RuntimeError("BROWSER_USE_API_KEY not set")
    return _bh_admin.sync_local_profile(
        profile_name=profile_name,
        browser=browser,
        cloud_profile_id=cloud_profile_id,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )


# ---------------------------------------------------------------------------
# public lists
# ---------------------------------------------------------------------------

# Read-only — safe to bind to most agents.
ADMIN_READ_TOOLS = [
    browser_doctor,
    browser_list_cloud_profiles,
    browser_list_local_profiles,
]

# Write / billable / stateful — opt in explicitly.
ADMIN_WRITE_TOOLS = [
    browser_start_remote_session,
    browser_stop_remote_session,
    browser_restart_daemon,
    browser_sync_local_profile,
]

ADMIN_TOOLS = ADMIN_READ_TOOLS + ADMIN_WRITE_TOOLS
