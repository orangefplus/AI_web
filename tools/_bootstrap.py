"""Browser-harness daemon bootstrap helpers.

The browser-harness package is lazy: helpers assume a daemon is already
running, and they will raise a connection error on the first call if it
isn't. These helpers guarantee a single, idempotent start so LangChain
agents can call browser tools without worrying about lifecycle.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

try:
    from browser_harness.admin import ensure_daemon as _bh_ensure_daemon
    from browser_harness.admin import NAME as _BH_NAME
except ImportError as e:  # pragma: no cover - import error path
    raise ImportError(
        "browser-harness is not installed. Run:\n"
        "  uv tool install -e c:/Users/wangxin/Documents/trae_projects/Rag_heima/AI_web/browser-harness\n"
        "or\n"
        "  pip install -e c:/Users/wangxin/Documents/trae_projects/Rag_heima/AI_web/browser-harness"
    ) from e

_lock = threading.Lock()
_started: bool = False


def setup_browser(name: Optional[str] = None, wait: float = 60.0, env: Optional[dict] = None) -> None:
    """Ensure a browser-harness daemon is running.

    Call once before invoking any browser tool. Subsequent calls are
    cheap no-ops. The daemon endpoint is namespaced by the BU_NAME env
    var (default "default"); pass ``name`` to override per-call, or set
    BU_NAME in the process environment to isolate parallel agents.

    Args:
        name:  Daemon name (sets BU_NAME for this call only).
        wait:  Seconds to wait for the daemon to come up.
        env:   Extra env vars merged into the daemon's environment
               (e.g. {"BU_CDP_WS": "wss://..."} for a cloud browser).
    """
    global _started
    with _lock:
        if _started and not name and not env:
            return
        merged_env = dict(env or {})
        if name:
            merged_env["BU_NAME"] = name
        _bh_ensure_daemon(wait=wait, name=name, env=merged_env or None)
        if not name and not env:
            _started = True


def daemon_name() -> str:
    """Current daemon's BU_NAME (the env var is read each call so re-assignments stick)."""
    return os.environ.get("BU_NAME", _BH_NAME)


def reset_for_tests() -> None:  # pragma: no cover - test helper
    global _started
    with _lock:
        _started = False
