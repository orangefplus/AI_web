"""High-level browser session that hides daemon details from agents.

Most agent tools only need three things from the harness:

1. Make sure the daemon is up (``setup_browser``).
2. Read the current browser state (``current_tab`` + ``page_info``).
3. Optionally take a screenshot for visual reasoning.

``BrowserSession`` packages all three behind a small, typed API so
that tools can stay focused on their own logic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from browser_harness import helpers as _bh

from ._bootstrap import setup_browser
from .browser import (
    browser_ensure_real_tab,
    browser_get_page_info,
    browser_list_tabs,
    browser_screenshot,
)


@dataclass
class BrowserSnapshot:
    """Point-in-time view of the browser used as the agent's context."""

    real_tab: Optional[dict]
    tabs: list[dict]
    page_info: dict
    screenshot_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowserSession:
    """Context manager that prepares a browser for tool invocations.

    Usage:
        >>> with BrowserSession() as session:
        ...     snapshot = session.snapshot()
        ...     print(snapshot.page_info["title"])

    The session ensures ``setup_browser`` has been called, then exposes
    ``snapshot()`` and ``current_tab()`` helpers. ``close()`` is
    idempotent and safe to call from a ``finally`` block.
    """

    def __init__(self, *, screenshot_dim: int = 1400) -> None:
        self.screenshot_dim = screenshot_dim
        self._snapshot: Optional[BrowserSnapshot] = None

    # -- context-manager protocol ---------------------------------------

    def __enter__(self) -> "BrowserSession":
        setup_browser()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Nothing to release; daemon is process-global.
        return None

    # -- public API ------------------------------------------------------

    def snapshot(self, *, refresh: bool = True) -> BrowserSnapshot:
        """Return a fresh or cached snapshot of the browser state."""
        if self._snapshot is not None and not refresh:
            return self._snapshot

        real_tab = browser_ensure_real_tab.invoke({})
        tabs = browser_list_tabs.invoke({"include_chrome": False})
        page_info = browser_get_page_info.invoke({})
        screenshot_path = browser_screenshot.invoke(
            {"max_dim": self.screenshot_dim}
        )
        self._snapshot = BrowserSnapshot(
            real_tab=real_tab,
            tabs=tabs,
            page_info=page_info,
            screenshot_path=screenshot_path,
        )
        return self._snapshot

    def current_tab(self) -> dict:
        """Return just the currently attached tab (cheap call)."""
        setup_browser()
        return _bh.current_tab()

    def list_tabs(self, *, include_chrome: bool = False) -> list[dict]:
        """List tabs, optionally including internal pages."""
        setup_browser()
        return _bh.list_tabs(include_chrome=include_chrome)

    def close(self) -> None:  # pragma: no cover - trivial
        """No-op; kept for symmetry with future stateful sessions."""
        return None


__all__ = ["BrowserSession", "BrowserSnapshot"]
