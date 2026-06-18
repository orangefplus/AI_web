"""Multimodal payload helpers.

The iFlytek MaaS API uses an OpenAI-compatible message schema, so we
can attach images to a HumanMessage by passing a list of content
blocks:

    content = [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]

The Xunfei gateway accepts:

  - public http(s) URL;
  - ``data:<mime>;base64,<...>`` URI (most reliable);
  - ``file://`` URI (only when the LLM server can read the path,
    generally not the case in a hosted environment — we still try
    file:// first because it skips the base64 round-trip and keeps
    the request body small).

This module centralises the read+encode logic so the masters don't
all reinvent it.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Optional, Union


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# iFlytek has a hard size cap (about 4 MB per image in our tests);
# bigger files are downscaled via the browser harness so we rarely
# hit it, but we surface a clear error if we do.
MAX_INLINE_BYTES = 3_500_000  # 3.5 MB safe margin


def is_image_path(value: Any) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    p = Path(str(value))
    return p.suffix.lower() in SUPPORTED_IMAGE_EXTS and p.is_file()


def _detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def _read_data_url(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_INLINE_BYTES:
        # 1) Try to downscale with pillow if installed.
        try:
            from PIL import Image  # type: ignore
            import io
            with Image.open(path) as im:
                im = im.convert("RGB")
                q, w, h = 85, im.width, im.height
                # Iteratively compress to fit under the cap.
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                while buf.tell() > MAX_INLINE_BYTES and q > 40:
                    q -= 10
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=q, optimize=True)
                if buf.tell() <= MAX_INLINE_BYTES:
                    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            pass
        raise ValueError(
            f"screenshot {path} is {len(raw)} bytes, "
            f"larger than {MAX_INLINE_BYTES} (Pillow not installed to downscale)"
        )
    mime = _detect_mime(path)
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def make_image_content_block(
    path: Union[str, os.PathLike],
    prefer_file_url: bool = False,
) -> dict:
    """Return a single content block for a screenshot.

    Args:
        path: local file path to the screenshot.
        prefer_file_url: if True, return a ``file://`` URL block; if the
            LLM gateway cannot resolve it, the user will see a clean
            error.  Default: encode to a base64 data URL.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if prefer_file_url:
        return {
            "type": "image_url",
            "image_url": {"url": "file://" + str(p.resolve())},
        }
    return {
        "type": "image_url",
        "image_url": {"url": _read_data_url(p)},
    }


def build_multimodal_human_content(
    text: str,
    screenshot_paths: Optional[list[Union[str, os.PathLike]]] = None,
    prefer_file_url: bool = False,
) -> list[dict]:
    """Compose the ``content`` list for a HumanMessage.

    Args:
        text: the textual instruction.
        screenshot_paths: list of screenshot file paths to attach.
        prefer_file_url: if True, use ``file://`` URLs (no base64).
    """
    blocks: list[dict] = [{"type": "text", "text": text}]
    for p in screenshot_paths or []:
        if is_image_path(p):
            blocks.append(make_image_content_block(p, prefer_file_url=prefer_file_url))
        else:
            # Allow plain strings to be passed for debugging.
            blocks.append({"type": "text", "text": f"[missing screenshot: {p}]"})
    return blocks


def extract_screenshot_paths(state_dict: dict) -> list[str]:
    """Pull screenshot paths out of the supervisor state.

    Looks at the most recent subagent result, the scratchpad, and a
    top-level ``last_screenshot`` key (set by the browser harness).
    """
    out: list[str] = []
    candidates: list[Any] = [
        state_dict.get("last_screenshot"),
        (state_dict.get("scratchpad") or {}).get("last_screenshot"),
    ]
    last_obs = (state_dict.get("current_state") or {}).get("last_observation") or {}
    last_data = last_obs.get("data") or {}
    candidates.append(last_data.get("screenshot_path"))
    candidates.append(last_data.get("screenshot"))

    last_history = (state_dict.get("history") or [])
    if last_history:
        h = last_history[-1] or {}
        candidates.append((h.get("data") or {}).get("screenshot_path"))

    for c in candidates:
        if is_image_path(c):
            out.append(str(c))
    # Deduplicate, preserve order.
    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


__all__ = [
    "build_multimodal_human_content",
    "extract_screenshot_paths",
    "is_image_path",
    "make_image_content_block",
]
