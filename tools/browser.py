"""LangChain tools wrapping browser-harness atomic browser operations.

All tools are thin @tool-decorated wrappers around helpers in
``browser_harness.helpers``. The daemon is bootstrapped lazily on first
use (or eagerly via :func:`setup_browser`). Every tool returns a
JSON-serializable value so the agent can re-prompt cleanly.

Naming convention: ``browser_*`` to avoid clashing with other tool
sets in a RAG agent (e.g. retriever tools). Docstrings are written in
Google style so LangChain can expose parameter descriptions in the tool
schema for function-calling models.

Each tool uses :func:`tools.decorators.describe` to attach a
structured, scenario-driven description and
:func:`tools.decorators.with_verification` where the operation needs
a success/failure report.
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from browser_harness import helpers as _bh

from ._bootstrap import setup_browser
from ._tooling import tool
from .decorators import describe, with_verification


_BH_ENV_SKILLS = "BH_DOMAIN_SKILLS"  # when "1", goto_url also returns matching domain skills


# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------

@describe(
    purpose=(
        "Open a URL in the tab the agent is currently attached to, "
        "replacing whatever page was there."
    ),
    when_to_use=(
        "User wants to revisit the same tab, refresh navigation, or "
        "explicitly says 'go to ... in this tab'. For most 'open a "
        "website' requests prefer browser_new_tab to leave the user's "
        "existing pages intact."
    ),
    caveats=(
        "Overwrites the current tab. If a user has unsaved work, this "
        "is destructive — switch to browser_new_tab in that case. "
        "When BH_DOMAIN_SKILLS=1 is set, the response may include a "
        "'domain_skills' list with site-specific playbooks."
    ),
)
@tool
def browser_navigate(url: str) -> dict:
    """Navigate the current tab to a URL.

    Args:
        url: Absolute URL to open in the currently attached tab.

    Returns:
        The CDP Page.navigate response as a dict, optionally extended
        with 'domain_skills'.
    """
    setup_browser()
    return _bh.goto_url(url)


@describe(
    purpose=(
        "Open a brand-new browser tab and attach the agent's session "
        "to it, leaving every existing tab untouched."
    ),
    when_to_use=(
        "Default choice for 'open this website', 'go search for ...', "
        "'log into X', 'navigate to ...' requests. Also use when the "
        "user's existing tab has content the agent must not disturb."
    ),
    caveats=(
        "If url is 'about:blank' the new tab is left empty and the "
        "tool returns immediately. For any other URL the tool waits "
        "for the new page to finish loading before returning, so the "
        "next call sees a fully rendered page."
    ),
)
@tool
def browser_new_tab(url: str = "about:blank") -> dict:
    """Open a new tab, attach the session to it, and optionally navigate.

    Args:
        url: URL to open in the new tab. Leave as 'about:blank' to
            create an empty working tab.

    Returns:
        Dict with the new tab's targetId, url, and title.
    """
    setup_browser()
    _bh.new_tab(url)
    if url != "about:blank":
        _bh.wait_for_load()
    return _bh.current_tab()


@tool
def browser_list_tabs(include_chrome: bool = True) -> list:
    """List open browser tabs that the harness can attach to.

    Use this before switching or closing tabs. Set
    ``include_chrome=False`` to hide internal pages such as
    ``chrome://``, ``devtools://``, and extension tabs.

    Args:
        include_chrome: Whether to include internal Chrome and DevTools
            tabs in the results.

    Returns:
        List of ``{targetId, title, url}`` dicts.
    """
    setup_browser()
    return _bh.list_tabs(include_chrome=include_chrome)


@tool
def browser_switch_tab(target_id: str) -> dict:
    """Switch the attached session to another tab by ``targetId``.

    Call ``browser_list_tabs`` first to discover valid target IDs.

    Args:
        target_id: Tab ``targetId`` returned by ``browser_list_tabs``.

    Returns:
        Dict describing the newly attached tab.
    """
    setup_browser()
    _bh.switch_tab(target_id)
    return _bh.current_tab()


@tool
def browser_close_tab(target_id: str = "") -> str:
    """Close a tab by ``targetId`` or close the current tab by default.

    Args:
        target_id: Tab ``targetId`` to close. Leave empty to close the
            currently attached tab.

    Returns:
        The ``targetId`` of the tab that was closed.
    """
    setup_browser()
    closed = target_id or (_bh.current_tab() or {}).get("targetId")
    _bh.close_tab(target_id or None)
    return closed or ""


@describe(
    purpose=(
        "Close every browser tab except the one the user is currently "
        "looking at, then verify the result."
    ),
    when_to_use=(
        "User says 'close the other tabs', 'only keep this window', "
        "'clean up the tabs', '关掉其他窗口', '只保留这个标签页', "
        "'只保留当前页', or any other phrasing that means "
        "'remove every other tab and keep the active one'."
    ),
    caveats=(
        "The currently attached tab is detected automatically and is "
        "never closed. about:blank tabs are treated as user tabs and "
        "are closed. Internal pages (chrome://, devtools://, extension "
        "pages) are closed by default because Chrome's new-tab page is "
        "the most common source of 'leftover' tabs; pass "
        "include_chrome=False to keep them. CDP closes tabs "
        "asynchronously, so the tool polls and retries once before "
        "reporting success. The returned dict always has 'success', "
        "'kept', 'closed', and 'remaining' fields — always report "
        "them to the user."
    ),
)
@tool
def browser_close_other_tabs(include_chrome: bool = True, verify: bool = True) -> dict:
    """Close every browser tab except the active one and verify.

    Use this for "close all other tabs / only keep the current one"
    style tasks. Triggered by phrasing like 'close the other tabs',
    'only keep this window', 'clean up the tabs', or Chinese
    equivalents like '关掉其他窗口' / '只保留这个标签页'. The
    currently attached tab is detected automatically and never closed;
    about:blank tabs are treated as user tabs and closed; internal
    pages (chrome://, devtools://, extensions) are closed by default
    because Chrome's new-tab page is the most common source of
    'leftover' tabs. CDP closes tabs asynchronously, so the tool
    polls and retries once. The returned dict always has 'success',
    'kept', 'closed', and 'remaining' fields — always report them to
    the user.

    Args:
        include_chrome: Whether to also close internal Chrome pages,
            DevTools, and extension pages. Defaults to True so the
            new-tab page does not survive.
        verify: Re-list tabs after closing and assert only the kept
            tab survives. Defaults to True so callers can rely on
            the success flag.

    Returns:
        Dict with:
            - kept: the tab that survived (targetId, url, title)
            - closed: list of targetIds that were closed
            - success: True iff verification confirmed only the kept
              tab remains
            - remaining: tabs still open after the operation
            - verified: whether a verification pass was actually run
    """
    setup_browser()
    kept = _bh.current_tab()
    candidates = _list_user_tabs(include_chrome=include_chrome)
    closed: list[str] = []
    for tab in candidates:
        target_id = tab.get("targetId")
        if not target_id or target_id == kept.get("targetId"):
            continue
        try:
            _bh.close_tab(target_id)
            closed.append(target_id)
        except Exception as exc:  # pragma: no cover - defensive
            # One bad tab should not abort the rest of the cleanup.
            closed.append(f"{target_id} (error: {exc})")

    result: dict = {
        "kept": kept,
        "closed": closed,
        "verified": False,
        "success": True,
        "remaining": [],
    }
    if not verify:
        return result

    # Verification pass: only the kept tab should be left.
    # Target.closeTarget is asynchronous, so poll briefly and retry
    # stubborn tabs once before declaring success.
    kept_id = kept.get("targetId")
    deadline = time.monotonic() + 2.0
    leftover: list[dict] = []
    while time.monotonic() < deadline:
        remaining = _list_user_tabs(include_chrome=include_chrome)
        leftover = [t for t in remaining if t.get("targetId") != kept_id]
        if not leftover:
            break
        time.sleep(0.1)

    if leftover:
        # Retry once for tabs that haven't gone away yet (e.g.
        # chrome://newtab, which Chrome sometimes delays closing).
        for tab in leftover:
            try:
                _bh.close_tab(tab.get("targetId"))
            except Exception:
                pass
        time.sleep(0.3)
        remaining = _list_user_tabs(include_chrome=include_chrome)
        leftover = [t for t in remaining if t.get("targetId") != kept_id]

    result["remaining"] = leftover
    result["verified"] = True
    result["success"] = not leftover
    return result


# Internal helper: same shape as helpers.list_tabs but treats
# ``about:blank`` as a user tab (the default ``list_tabs`` filters
# every ``about:`` URL out as "internal", which would silently leave
# stray empty tabs behind when the user asked to "close all others").
_HARD_INTERNAL_PREFIXES = ("chrome://", "chrome-untrusted://", "devtools://", "chrome-extension://")


def _list_user_tabs(include_chrome: bool = False) -> list[dict]:
    """Like ``helpers.list_tabs`` but keeps ``about:blank`` tabs.

    ``include_chrome=False`` still hides ``chrome://`` /
    ``devtools://`` / extension pages, but ``about:blank`` is treated
    as a real user tab. When ``include_chrome=True`` everything is
    returned (mirroring ``list_tabs(include_chrome=True)``).
    """
    all_tabs = _bh.list_tabs(include_chrome=True)
    if include_chrome:
        return all_tabs
    return [
        t for t in all_tabs
        if not t.get("url", "").startswith(_HARD_INTERNAL_PREFIXES)
    ]


@tool
def browser_ensure_real_tab() -> dict | None:
    """Recover from an internal or stale tab by switching to a real page.

    Use this after unexpected redirects to ``chrome://`` or when the
    daemon reconnects to a stale target.

    Returns:
        The newly attached real tab, or ``None`` if no real page tabs
        are open.
    """
    setup_browser()
    return _bh.ensure_real_tab()


# ---------------------------------------------------------------------------
# input — coordinate clicks default, since they pass through
# iframes / shadow DOM / cross-origin at the compositor level.
# ---------------------------------------------------------------------------

@tool
def browser_click_xy(
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
    clicks: int = 1,
) -> str:
    """Click at viewport coordinates (x, y) in CSS pixels.

    Coordinate clicks work through iframes, shadow DOM, and cross-origin
    iframes at the compositor level, so this is usually more reliable
    than selector hunting. Use ``browser_screenshot`` first to read the
    coordinates from the current page image.

    Args:
        x: Horizontal viewport coordinate in CSS pixels.
        y: Vertical viewport coordinate in CSS pixels.
        button: Mouse button to press: ``left``, ``right``, or
            ``middle``.
        clicks: Number of clicks to send. Use ``2`` for a double-click.

    Returns:
        Short confirmation string describing the click that was sent.
    """
    setup_browser()
    _bh.click_at_xy(x, y, button=button, clicks=clicks)
    return f"clicked ({x},{y}) {button} x{clicks}"


@tool
def browser_type_text(text: str) -> str:
    """Type text into the currently focused element.

    This sends text through ``Input.insertText``. For framework-managed
    inputs such as React controlled fields or Vue ``v-model`` fields,
    prefer ``browser_fill_input`` so the framework receives the expected
    focus, key, input, and change signals.

    Args:
        text: Text to insert into the currently focused element.

    Returns:
        Short confirmation string with the character count typed.
    """
    setup_browser()
    _bh.type_text(text)
    return f"typed {len(text)} chars"


@tool
def browser_fill_input(
    selector: str,
    text: str,
    clear_first: bool = True,
    timeout: float = 0.0,
) -> str:
    """Fill a framework-managed input by CSS selector.

    This focuses the element, optionally clears it, types via real key
    events, and fires ``input`` plus ``change`` so frameworks notice the
    update. Use this whenever plain typing leaves a save or submit
    button disabled.

    Args:
        selector: CSS selector for the input, textarea, or editable
            element to fill.
        text: Text to enter into the matched element.
        clear_first: Whether to select-all and delete existing content
            before typing.
        timeout: Seconds to wait for the element to appear before
            failing. Useful after SPA route changes or delayed renders.

    Returns:
        Short confirmation string describing what was filled.
    """
    setup_browser()
    _bh.fill_input(selector, text, clear_first=clear_first, timeout=timeout)
    return f"filled {selector!r} with {len(text)} chars"


@tool
def browser_press_key(key: str, modifiers: int = 0) -> str:
    """Press a single key.

    Use this for shortcuts such as Ctrl+A, Cmd+Enter, Escape, or arrow
    navigation. ``modifiers`` is a bitfield where ``1=Alt``,
    ``2=Ctrl``, ``4=Meta/Cmd``, and ``8=Shift``.

    Args:
        key: Named key such as ``Enter``, ``Tab``, ``Escape``,
            ``ArrowDown``, or a single character such as ``a``.
        modifiers: Modifier bitfield. Example: ``2`` for Ctrl,
            ``4`` for Cmd, ``10`` for Ctrl+Shift.

    Returns:
        Short confirmation string describing the keypress.
    """
    setup_browser()
    _bh.press_key(key, modifiers=modifiers)
    return f"pressed {key!r} (mod={modifiers})"


@tool
def browser_scroll(x: int, y: int, dy: int = -300, dx: int = 0) -> str:
    """Dispatch a mouse-wheel event at the given viewport coordinates.

    Args:
        x: Horizontal viewport coordinate in CSS pixels where the wheel
            event should be targeted.
        y: Vertical viewport coordinate in CSS pixels where the wheel
            event should be targeted.
        dy: Vertical wheel delta. Positive scrolls down, negative
            scrolls up.
        dx: Horizontal wheel delta. Positive scrolls right, negative
            scrolls left.

    Returns:
        Short confirmation string describing the scroll event.
    """
    setup_browser()
    _bh.scroll(x, y, dy=dy, dx=dx)
    return f"scrolled dx={dx} dy={dy} at ({x},{y})"


@tool
def browser_dispatch_key(selector: str, key: str, event: str = "keypress") -> str:
    """Dispatch a synthetic DOM ``KeyboardEvent`` on a matched element.

    Use this when a page listens for DOM events on the element itself
    and ignores lower-level CDP key input.

    Args:
        selector: CSS selector for the element that should receive the
            keyboard event.
        key: Keyboard key name such as ``Enter`` or ``Escape``.
        event: DOM keyboard event type, usually ``keydown``,
            ``keypress``, or ``keyup``.

    Returns:
        Short confirmation string describing the dispatched event.
    """
    setup_browser()
    _bh.dispatch_key(selector, key=key, event=event)
    return f"dispatched {event} {key!r} on {selector!r}"


@tool
def browser_upload_file(selector: str, path: str) -> str:
    """Assign a file to an ``<input type=file>`` element.

    This works even when the file input is hidden. The path must point
    to a real local file.

    Args:
        selector: CSS selector for the target file input element.
        path: Absolute path to the file that should be uploaded.

    Returns:
        Short confirmation string describing the uploaded file.
    """
    setup_browser()
    _bh.upload_file(selector, path)
    return f"uploaded {path} -> {selector!r}"


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------

@describe(
    purpose=(
        "Take a PNG screenshot of the current tab so the agent can "
        "reason about pixels and pick click targets from the image."
    ),
    when_to_use=(
        "Default 'what is on screen right now?' call. Take it before "
        "any browser_click_xy call, after navigation, after form "
        "submits, or whenever text-based page_info is not enough. "
        "Also useful for summarizing the page in the user's response."
    ),
    caveats=(
        "Returns a file path, not the image itself. The max_dim "
        "parameter caps the longest side to keep the image within "
        "common multimodal-model limits. Use full_page=True only when "
        "the user explicitly asks for everything; otherwise prefer the "
        "viewport so screenshots stay small."
    ),
)
@tool
def browser_screenshot(
    path: str = "",
    full_page: bool = False,
    max_dim: int = 1800,
) -> str:
    """Take a PNG screenshot of the current tab and return the file path.

    Args:
        path: Output path for the PNG. Leave empty to use a temp file.
        full_page: Whether to capture the full scroll height instead of
            only the visible viewport.
        max_dim: Resize the longest side to this many pixels.

    Returns:
        Absolute path to the saved PNG file.
    """
    setup_browser()
    return _bh.capture_screenshot(path or None, full=full_page, max_dim=max_dim)


@describe(
    purpose=(
        "Return cheap text metadata about the current tab and viewport: "
        "URL, title, dimensions, scroll offset, and any open dialog."
    ),
    when_to_use=(
        "Call this at the start of a task to understand which page the "
        "agent is on. Use it whenever the user asks 'what page am I "
        "on', 'what is the title', or 'is this dialog blocking me'. "
        "Cheaper than browser_screenshot when only text context is "
        "needed."
    ),
    caveats=(
        "If a native dialog (alert, confirm, prompt, beforeunload) is "
        "open the response contains a 'dialog' payload instead of the "
        "usual fields — the page's JavaScript thread is paused and "
        "most other actions will not work until the dialog is handled."
    ),
)
@tool
def browser_get_page_info() -> dict:
    """Return metadata about the current tab and viewport.

    Returns:
        Dict containing page URL, title, viewport size, scroll offset,
        and page dimensions, or a pending dialog payload.
    """
    setup_browser()
    return _bh.page_info()


@tool
def browser_handle_dialog(accept: bool = True, prompt_text: str = "") -> dict:
    """Dismiss a blocking native dialog (``alert``/``confirm``/``prompt``).

    Some sites (e.g. a campus WebVPN gateway on first load) pop a
    native dialog. While that dialog is open, the page's JavaScript
    thread is frozen and every other tool that talks to the page
    (screenshot, click_xy, run_js, ...) will hang or return a dialog
    payload. Call this tool whenever ``browser_get_page_info`` returns
    a ``"dialog"`` field, or proactively right after navigating to a
    new site.

    Args:
        accept: ``True`` to click OK / Yes, ``False`` to click Cancel.
        prompt_text: If the dialog is a ``prompt()``, the value to type
            in before accepting. Ignored for alerts / confirms.

    Returns:
        Dict ``{"ok": bool, "dialog": {...} | None, "accepted": bool}``.
    """
    setup_browser()
    return _bh.handle_dialog(accept=accept, prompt_text=prompt_text)


@tool
def browser_dismiss_overlay(accept_keywords: list[str] | None = None) -> dict:
    """Look for a visible modal / overlay / consent dialog and dismiss it.

    Many sites (NIH GDC Data Portal, Elsevier, IEEE Xplore, university
    WebVPN gateways, ...) drop a centered "Accept" / "I agree" / "Got
    it" / "Continue" modal on first load. The modal usually blocks the
    search box, the navigation menu, or any click behind it, so the
    agent must close it before doing anything else on the page.
    search box, the navigation menu, or any click behind it, so the
    agent must close it before doing anything else on the page.

    This tool finds the topmost visible modal/overlay element and tries
    to click its primary "accept"-style button. If it cannot find a
    modal it returns ``found=False`` so the caller knows nothing was
    dismissed.

    Args:
        accept_keywords: Optional override of the button text we will
            accept. Default covers the most common variants in
            English / Simplified Chinese / Traditional Chinese.

    Returns:
        Dict ``{"found": bool, "clicked": bool, "text": str,
        "button": {"tag", "text", "x", "y"}, "note": str}``.
    """
    setup_browser()
    if not accept_keywords:
        accept_keywords = [
            "accept", "i agree", "agree", "got it", "ok", "okay",
            "continue", "proceed", "close", "dismiss", "allow",
            "yes", "confirm",
            "同意", "我同意", "接受", "确定", "好", "确认", "知道了",
            "继续", "关闭", "允许",
        ]
    # We do this in pure DOM JS — no selector tools, no remote fetch.
    # Returns a small payload describing what (if anything) was found.
    payload = _bh.js(
        """(function() {
            const kws = (arguments[0] || []).map(s => String(s).toLowerCase());
            const isVisible = el => {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                if (r.width < 4 || r.height < 4) return false;
                const cs = getComputedStyle(el);
                if (cs.visibility === 'hidden' || cs.display === 'none') return false;
                if (parseFloat(cs.opacity || '1') < 0.05) return false;
                return r.top < innerHeight && r.bottom > 0
                    && r.left < innerWidth && r.right > 0;
            };
            // Heuristics for a "modal": a centered element that covers
            // most of the viewport, with a role=button or button-ish
            // child. We look at the topmost element under the cursor's
            // path first (elementFromPoint), then fall back to a
            // walking scan.
            const candidates = [];
            const visit = el => {
                if (!el || !isVisible(el)) return;
                const tag = el.tagName;
                const role = (el.getAttribute('role') || '').toLowerCase();
                if (tag === 'BUTTON' || tag === 'A' || tag === 'INPUT'
                    || role === 'button' || role === 'link') {
                    const txt = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
                    if (txt) {
                        const r = el.getBoundingClientRect();
                        candidates.push({el, text: txt, x: r.left + r.width/2, y: r.top + r.height/2});
                    }
                }
                for (const c of el.children) visit(c);
            };
            // 1) Try the topmost element at the visual center.
            const cx = innerWidth/2, cy = innerHeight/2;
            const top = document.elementFromPoint(cx, cy);
            if (top) {
                // Walk up to find a modal-like container, then within
                // it find any clickable button that matches our
                // keyword list.
                let modal = top;
                for (let i = 0; i < 6 && modal; i++) {
                    const r = modal.getBoundingClientRect();
                    if (r.width > innerWidth*0.3 && r.height > innerHeight*0.3) break;
                    modal = modal.parentElement;
                }
                if (modal) {
                    const buttons = modal.querySelectorAll('button, a[role="button"], input[type="button"], input[type="submit"], [role="button"]');
                    for (const b of buttons) {
                        if (!isVisible(b)) continue;
                        const t = (b.innerText || b.value || b.getAttribute('aria-label') || '').trim();
                        if (!t) continue;
                        const r = b.getBoundingClientRect();
                        candidates.push({el: b, text: t, x: r.left + r.width/2, y: r.top + r.height/2});
                    }
                }
            }
            // 2) Fall back: walk the whole tree for any visible button
            //    whose text matches a keyword.
            visit(document.body);
            const ranked = candidates
                .filter(c => kws.some(k => c.text.toLowerCase().includes(k)))
                .sort((a, b) => {
                    // Prefer the visually largest (most-likely primary) button.
                    const ra = a.el.getBoundingClientRect();
                    const rb = b.el.getBoundingClientRect();
                    return (rb.width*rb.height) - (ra.width*ra.height);
                });
            if (ranked.length === 0) {
                return {found: false, candidates: candidates.length};
            }
            const top_btn = ranked[0];
            return {
                found: true,
                text: top_btn.text,
                x: Math.round(top_btn.x),
                y: Math.round(top_btn.y),
                tag: top_btn.el.tagName,
            };
        })(arguments[0])""",
        accept_keywords,
    )
    if not isinstance(payload, dict) or not payload.get("found"):
        return {
            "found": False,
            "clicked": False,
            "note": "no accept-style button found",
        }
    x = int(payload.get("x") or 0)
    y = int(payload.get("y") or 0)
    if x <= 0 or y <= 0:
        return {
            "found": True,
            "clicked": False,
            "text": payload.get("text", ""),
            "note": "button found but coordinates invalid",
        }
    _bh.click_at_xy(x, y)
    return {
        "found": True,
        "clicked": True,
        "text": payload.get("text", ""),
        "button": {
            "tag": payload.get("tag"),
            "text": payload.get("text"),
            "x": x,
            "y": y,
        },
    }


@tool
def browser_read_page_text(max_chars: int = 8000) -> str:
    """Return the visible text of the current page as one big string.

    A safer, less error-prone alternative to writing your own
    ``document.querySelectorAll(...)`` snippet. The harness collects
    the visible text of ``document.body`` (skipping script/style and
    hidden subtrees) and returns the first ``max_chars`` characters.

    Use this whenever you need to:
      - extract structured data (titles, authors, snippets) from a
        results page;
      - confirm which page you are on;
      - check whether a click actually had an effect.

    Args:
        max_chars: Hard cap on the returned string length. Defaults
            to 8000 which is plenty for one search-results page.
            Pass a smaller number if you only need the top of the
            page (e.g. the first 1500 chars for a navigation menu).

    Returns:
        Plain text — already-trimmed, no HTML, no emoji, no scripts.
        Empty string if the page has no body text yet.
    """
    setup_browser()
    if max_chars <= 0 or max_chars > 100000:
        max_chars = 8000
    expression = (
        "(function(){"
        "  function walk(node, out){"
        "    if (!node) return;"
        "    if (node.nodeType === 3) { out.push(node.nodeValue); return; }"
        "    if (node.nodeType !== 1) return;"
        "    var tag = node.tagName;"
        "    if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') return;"
        "    var cs = getComputedStyle(node);"
        "    if (cs.display === 'none' || cs.visibility === 'hidden') return;"
        "    for (var c = node.firstChild; c; c = c.nextSibling) walk(c, out);"
        "  }"
        "  var out = [];"
        "  walk(document.body, out);"
        "  var s = out.join('').replace(/\\s+/g, ' ').trim();"
        "  return s.length > arguments[0] ? s.slice(0, arguments[0]) : s;"
        "})()"
    )
    return _bh.js(expression, max_chars) or ""


@tool
def browser_run_js(expression: str, target_id: str = "") -> object:
    """Evaluate JavaScript in the current tab or a specific iframe target.

    Promises are awaited. Expressions with a top-level ``return`` are
    auto-wrapped in an IIFE, so both ``document.title`` and
    ``const x = 1; return x`` are valid. Use this for DOM inspection,
    extraction, and advanced interaction when higher-level browser tools
    are not enough.

    Args:
        expression: JavaScript expression or snippet to evaluate.
        target_id: Optional iframe target ID. Leave empty to run in the
            current top-level page. **Must be a string** — non-string
            values trigger a clear error so the LLM does not silently
            pass a dict/number that Chrome DevTools Protocol rejects.

    Returns:
        The raw deserialized JavaScript result.
    """
    setup_browser()
    # Validate target_id: Chrome DevTools Protocol will return
    # "Invalid parameters" if we hand it anything other than a string
    # (or None).  LLM-generated callables occasionally pass dicts or
    # numbers that *look* like IDs; we want a clean TypeError that the
    # multi-agent error classifier can map to cdp-bad-params.
    if target_id is not None and not isinstance(target_id, str):
        raise TypeError(
            f"browser_run_js: target_id must be a string or None, "
            f"got {type(target_id).__name__}={target_id!r}. "
            f"Either pass a real iframe targetId (a string from "
            f"Page.getFrameTree) or leave it empty to run in the "
            f"current top-level page."
        )
    return _bh.js(expression, target_id=target_id or None)


# ---------------------------------------------------------------------------
# wait helpers
# ---------------------------------------------------------------------------

@tool
def browser_wait_for_load(timeout: float = 15.0) -> bool:
    """Block until ``document.readyState == 'complete'`` or timeout.

    This works well for full navigations, but it can miss SPA route
    changes because the document may already be complete before the
    framework finishes rendering.

    Args:
        timeout: Maximum number of seconds to wait.

    Returns:
        ``True`` if the page reached ``readyState='complete'`` before
        timeout, otherwise ``False``.
    """
    setup_browser()
    return _bh.wait_for_load(timeout=timeout)


@tool
def browser_wait_for_element(
    selector: str,
    timeout: float = 10.0,
    visible: bool = False,
) -> bool:
    """Block until ``querySelector(selector)`` exists in the DOM.

    Use this after actions that trigger async rendering such as route
    changes, lazy components, or data fetches.

    Args:
        selector: CSS selector that must appear in the DOM.
        timeout: Maximum number of seconds to wait.
        visible: When ``True``, also require the element to be visible
            and present in layout.

    Returns:
        ``True`` if the element appeared in time, otherwise ``False``.
    """
    setup_browser()
    return _bh.wait_for_element(selector, timeout=timeout, visible=visible)


@tool
def browser_wait_for_network_idle(timeout: float = 10.0, idle_ms: int = 500) -> bool:
    """Block until no in-flight network requests for ``idle_ms`` ms.

    Use this after form submits, SPA route transitions, and actions
    that trigger XHR or fetch without an obvious DOM change.

    Args:
        timeout: Maximum number of seconds to wait before giving up.
        idle_ms: How long the network must stay quiet to count as idle.

    Returns:
        ``True`` if the active session became idle before timeout,
        otherwise ``False``.
    """
    setup_browser()
    return _bh.wait_for_network_idle(timeout=timeout, idle_ms=idle_ms)


# ---------------------------------------------------------------------------
# network — pure HTTP, no browser
# ---------------------------------------------------------------------------

@tool
def browser_http_get(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 20.0,
) -> str:
    """Fetch a URL over plain HTTP without using the browser session.

    Use this for static pages, APIs, robots-friendly scraping, or bulk
    fetches. This does not share cookies, local storage, or page state
    with the live browser tab.

    Args:
        url: Absolute URL to request.
        headers: Optional HTTP headers to include with the request.
        timeout: Request timeout in seconds.

    Returns:
        Response body as text. When ``BROWSER_USE_API_KEY`` is set, the
        request may route through Browser Use proxy infrastructure.
    """
    setup_browser()  # daemon may not be needed but keeps lifecycle consistent
    return _bh.http_get(url, headers=headers, timeout=timeout)


@describe(
    purpose=(
        "Find a link on the current page by text or URL pattern and "
        "navigate to it — *without* requiring pixel-accurate clicking. "
        "Useful for hot search / nav bars / result lists where the LLM "
        "can read the link text in a screenshot but cannot estimate "
        "exact coordinates."
    ),
    when_to_use=(
        "Default choice when the page shows a list of links, the user "
        "asks to open a particular one, and you don't need the click "
        "to trigger custom JavaScript. Also great for back-and-forth "
        "workflows: open the target link in a NEW tab, read the result, "
        "close that tab, repeat."
    ),
    caveats=(
        "Only matches the FIRST link whose text or href contains the "
        "given substring (case-insensitive). For lists, pass the topic "
        "title visible in the screenshot. If the page uses JS-only "
        "navigation (no href), fall back to browser_click_xy."
    ),
)
@tool
def browser_navigate_to_link(
    text_or_href_match: str,
    open_new_tab: bool = False,
    wait_for_load: bool = True,
    timeout: float = 8.0,
) -> dict:
    """Find a link whose text or href contains the substring and navigate to it.

    Args:
        text_or_href_match: Case-insensitive substring to look for in
            the link's visible text or href. Empty string means "first
            link on the page".
        open_new_tab: If True, open the link in a fresh tab and attach
            the session to it; otherwise navigate the current tab.
        wait_for_load: Wait for the new page to fully load before returning.
        timeout: Seconds to wait for load.

    Returns:
        Dict with the matched link's {text, href, index} and the page
        the browser is now on. If no match, returns ``{ok: false,
        error: "no link matches ..."}``.
    """
    setup_browser()
    page_info = _bh.page_info() or {}
    url = page_info.get("url", "")
    base = _bh.domain_root(url) if hasattr(_bh, "domain_root") else None
    # Extract every <a> on the page; use a small inline JS expression
    # so we don't need a new tool just for the query.
    js = r"""
    (() => {
        const out = [];
        document.querySelectorAll('a[href]').forEach((a, i) => {
            const rect = a.getBoundingClientRect();
            const text = (a.innerText || a.textContent || '').trim();
            const href = a.getAttribute('href') || '';
            if (!href || href === '#') return;
            if (rect.width === 0 && rect.height === 0) return;
            out.push({i, text: text.slice(0, 200), href, x: rect.left + rect.width/2, y: rect.top + rect.height/2});
        });
        return out;
    })()
    """
    raw = _bh.js(js)
    if isinstance(raw, dict) and "result" in raw:
        links = raw.get("result") or []
    else:
        links = raw if isinstance(raw, list) else []
    if not links:
        return {"ok": False, "error": "no anchor tags on this page", "matched": None, "total_links": 0}

    needle = (text_or_href_match or "").strip().lower()
    matched = None
    if needle:
        for li in links:
            if needle in (li.get("text", "").lower()) or needle in (li.get("href", "").lower()):
                matched = li
                break
    else:
        matched = links[0]
    if not matched:
        return {
            "ok": False,
            "error": f"no link matches {text_or_href_match!r}",
            "matched": None,
            "total_links": len(links),
            "first_few": links[:5],
        }

    href = matched.get("href", "")
    # Resolve relative URLs against the current page URL.
    target_url = href
    if href.startswith("/") and url:
        from urllib.parse import urlparse, urljoin
        target_url = urljoin(url, href)
    elif not href.startswith(("http://", "https://", "about:", "data:")) and url:
        from urllib.parse import urljoin
        target_url = urljoin(url, href)

    if open_new_tab:
        _bh.new_tab(target_url)
        if wait_for_load:
            _bh.wait_for_load(timeout=timeout)
    else:
        _bh.goto_url(target_url)
        if wait_for_load:
            _bh.wait_for_load(timeout=timeout)

    return {
        "ok": True,
        "matched": {
            "text": matched.get("text"),
            "href": href,
            "index": matched.get("i"),
            "target_url": target_url,
        },
        "total_links": len(links),
        "now_on": _bh.page_info(),
    }


@describe(
    purpose=(
        "Return every <a href> link visible on the current page, with "
        "their visible text and absolute target URL.  Use this when the "
        "agent can read the link text in a screenshot but cannot guess "
        "the exact click coordinates — it can pick a link by text and "
        "then call browser_navigate_to_link (or browser_navigate) with "
        "the resolved URL."
    ),
    when_to_use=(
        "When the page shows a list of items (hot search / nav bar / "
        "search results / product list) and the agent needs to inspect "
        "the full link table before deciding where to go."
    ),
    caveats=(
        "Only returns links with a non-empty href and a non-zero size. "
        "If the list is huge, pass a ``text_filter`` substring to "
        "narrow it down.  ``limit`` caps the response size. Links whose "
        "href is just ``#`` or starts with the JS pseudo-protocol are "
        "dropped unless ``include_hash`` is True."
    ),
)
@tool
def browser_extract_links(
    text_filter: str = "",
    limit: int = 50,
    include_hash: bool = False,
) -> dict:
    """Return the list of visible links on the current page.

    Args:
        text_filter: Optional case-insensitive substring. Only links
            whose text OR href contains it are returned.
        limit: Maximum number of links to return (default 50).
        include_hash: If False, drop links whose href is just ``#`` or
            uses a non-navigating pseudo-protocol.

    Returns:
        Dict ``{ok, count, links: [{text, href, target_url}], ...}``.
    """
    setup_browser()
    page_info = _bh.page_info() or {}
    url = page_info.get("url", "")
    js = r"""
    (() => {
        const out = [];
        document.querySelectorAll('a[href]').forEach((a, i) => {
            const rect = a.getBoundingClientRect();
            const text = (a.innerText || a.textContent || '').trim();
            const href = a.getAttribute('href') || '';
            if (!href) return;
            if (rect.width === 0 && rect.height === 0) return;
            out.push({i, text: text.slice(0, 240), href});
        });
        return out;
    })()
    """
    raw = _bh.js(js)
    if isinstance(raw, dict) and "result" in raw:
        links = raw.get("result") or []
    else:
        links = raw if isinstance(raw, list) else []

    from urllib.parse import urljoin
    needle = (text_filter or "").strip().lower()
    out: list[dict] = []
    for li in links:
        href = li.get("href", "")
        text = li.get("text", "")
        if not include_hash and (href in ("#", "") or href.startswith("javascript:")):
            continue
        if needle and needle not in (text or "").lower() and needle not in (href or "").lower():
            continue
        target_url = href
        if href.startswith("/"):
            target_url = urljoin(url, href)
        elif not href.startswith(("http://", "https://", "about:", "data:", "mailto:")) and url:
            target_url = urljoin(url, href)
        out.append({
            "index": li.get("i"),
            "text": text,
            "href": href,
            "target_url": target_url,
        })
        if len(out) >= limit:
            break

    return {"ok": True, "count": len(out), "current_url": url, "links": out}


# ---------------------------------------------------------------------------
# public list
# ---------------------------------------------------------------------------

BROWSER_TOOLS = [
    # navigation
    browser_navigate,
    browser_navigate_to_link,
    browser_new_tab,
    browser_list_tabs,
    browser_switch_tab,
    browser_close_tab,
    browser_close_other_tabs,
    browser_ensure_real_tab,
    # input
    browser_click_xy,
    browser_type_text,
    browser_fill_input,
    browser_press_key,
    browser_scroll,
    browser_dispatch_key,
    browser_upload_file,
    browser_dismiss_overlay,
    browser_handle_dialog,
    # observation
    browser_screenshot,
    browser_get_page_info,
    browser_extract_links,
    browser_run_js,
    # wait
    browser_wait_for_load,
    browser_wait_for_element,
    browser_wait_for_network_idle,
    # network
    browser_http_get,
]
