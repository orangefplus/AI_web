"""Browser-only download tool (no HTTP fallback).

This module intentionally avoids ``urllib`` / ``requests`` so the
download goes through the actual browser. We expose two tools:

- :func:`browser_set_download_dir` — configures Chrome's download
  directory via ``Browser.setDownloadBehavior``. Call this first
  before any download attempt.
- :func:`browser_download_pdf` — drives the browser to a paper
  page, screenshots the viewer, then either:

  1. Clicks the viewer's toolbar download icon at the given
     coordinates (if the BrowserAgent inspected the screenshot
     first), or
  2. Falls back to ``Page.printToPDF`` (also a browser-internal
     operation) when no click coordinates are known yet.

The first form mimics a human: open page, look at UI, click
download. The second is a graceful degradation that still does
not bypass the browser.
"""
from __future__ import annotations

import os
from pathlib import Path

from browser_harness import helpers as _bh

from ._bootstrap import setup_browser
from ._tooling import tool


DEFAULT_TIMEOUT = 30.0


def _ensure_save_dir(save_dir: str) -> Path:
    """Return ``save_dir`` if writable, else fall back to ``./downloads``."""
    if save_dir:
        path = Path(save_dir).expanduser()
    else:
        path = Path("./downloads")
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    except OSError:
        fallback = Path.cwd() / "downloads"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


@tool
def browser_set_download_dir(save_dir: str) -> dict:
    """Configure the browser-wide download directory.

    This MUST be called before ``browser_download_pdf``. Without it,
    Chrome will route the file to its default download folder and
    the wait/download detection logic will not see the file.

    Args:
        save_dir: Directory to land downloads in. Created if missing.

    Returns:
        Dict with ``ok`` (bool) and ``path`` (str) keys.
    """
    setup_browser()
    target = _ensure_save_dir(save_dir)
    _bh.cdp("Browser.setDownloadBehavior",
            behavior="allow", downloadPath=str(target), eventsEnabled=True)
    return {"ok": True, "path": str(target)}


@tool
def browser_download_pdf(
    url: str = "",
    save_dir: str = "",
    click_x: int = 0,
    click_y: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Drive the browser to download a PDF, mirroring human behavior.

    The function is *strictly browser-driven*: no HTTP shortcuts, no
    URL hits bypassing the viewer UI. Workflow:

    1. Configure download directory (Browser.setDownloadBehavior).
    2. Navigate the browser to ``url`` (PDF renders in viewer).
    3. If click_x and click_y are non-zero, click those
       coordinates to hit the viewer's download icon.
    4. Wait for a new file to appear in ``save_dir``.
    5. If step 4 fails, fall back to ``Page.printToPDF`` — still
       a browser-internal operation, never a network shortcut.

    Args:
        url: Absolute URL of the PDF. If empty, assumes the
            browser is already on the PDF page (e.g. after
            calling ``browser_navigate``) and the caller only
            wants to click the viewer toolbar or re-print.
        save_dir: Where to save the file.
        click_x: Pixel X coordinate of the viewer's download
            icon (typically top-right of the toolbar). Use
            ``browser_screenshot`` first to read the
            coordinates from the image. Pass 0 to skip the
            click and go straight to ``printToPDF``.
        click_y: Pixel Y coordinate of the viewer's download
            icon.
        timeout: Wait timeout in seconds.

    Returns:
        Dict with ``ok``, ``path``, ``size``, ``via`` (one of
        ``"click"``, ``"event"``, ``"polling"``, ``"print"``).
    """
    setup_browser()
    target = _ensure_save_dir(save_dir)
    _bh.cdp("Browser.setDownloadBehavior",
            behavior="allow", downloadPath=str(target), eventsEnabled=True)

    if url:
        _bh.cdp("Page.navigate", url=url)
        # Give the viewer time to render its toolbar.
        import time
        time.sleep(2.0)

    # Snapshot files before any action so we only treat new arrivals.
    initial = {p.name: p.stat().st_mtime for p in target.iterdir() if p.is_file()}

    via = "print"
    final_path = ""
    final_size = 0

    if click_x > 0 and click_y > 0:
        # Click the viewer toolbar to trigger Chrome's native
        # download flow. The mouse events go through the same
        # compositor as a real click, so any site-specific
        # listeners still fire.
        _bh.cdp("Input.dispatchMouseEvent",
                type="mousePressed", x=click_x, y=click_y,
                button="left", clickCount=1)
        _bh.cdp("Input.dispatchMouseEvent",
                type="mouseReleased", x=click_x, y=click_y,
                button="left", clickCount=1)
        via = "click"

        # Wait for the file to land in the download directory.
        import time
        deadline = time.monotonic() + timeout
        stable_count = 0
        last_size = -1
        while time.monotonic() < deadline:
            try:
                for ev in _bh.drain_events():
                    if ev.get("method") == "Page.downloadProgress":
                        state = ev.get("params", {}).get("state")
                        if state == "completed":
                            via = "event"
            except Exception:
                pass

            for p in target.iterdir():
                if not p.is_file() or p.name in initial:
                    continue
                size = p.stat().st_size
                if size > final_size:
                    final_path = str(p)
                    final_size = size
                if size == last_size:
                    stable_count += 1
                    if stable_count >= 2 and size > 0:
                        return {
                            "ok": True, "path": final_path, "size": final_size,
                            "via": via,
                        }
                else:
                    stable_count = 0
                    last_size = size
            time.sleep(0.3)

    # Viewer click did not produce a file: ask Chrome to print the
    # current page to PDF. This is still a browser-internal action
    # — we are *not* hitting the URL again or downloading via HTTP.
    print_result = _bh.cdp("Page.printToPDF",
                           printBackground=False, landscape=False,
                           paperWidth=8.5, paperHeight=11,
                           marginTop=0.4, marginBottom=0.4,
                           marginLeft=0.4, marginRight=0.4)
    if "data" not in print_result:
        return {
            "ok": False,
            "error": "viewer click did not download and printToPDF returned no data",
            "via": "print",
        }
    import base64, time as _t
    out_path = target / f"printout-{int(_t.time() * 1000)}.pdf"
    out_path.write_bytes(base64.b64decode(print_result["data"]))
    return {
        "ok": True,
        "path": str(out_path),
        "size": out_path.stat().st_size,
        "via": "print",
    }


__all__ = ["browser_set_download_dir", "browser_download_pdf"]
