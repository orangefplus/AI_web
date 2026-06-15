"""Compatibility helpers for LangChain tool decorators.

These wrappers make the exported tool schemas friendlier for LLM tool
calling by enabling Google-style docstring parsing when the installed
LangChain version supports it. Older versions fall back to the plain
``@tool`` behavior.
"""
from __future__ import annotations

from inspect import signature

try:
    from langchain_core.tools import tool as _base_tool
except ImportError:  # pragma: no cover
    from langchain.tools import tool as _base_tool  # type: ignore


_SUPPORTS_PARSE_DOCSTRING = "parse_docstring" in signature(_base_tool).parameters


def tool(*args, **kwargs):
    """Return a LangChain tool with docstring parsing enabled when possible."""
    if _SUPPORTS_PARSE_DOCSTRING:
        kwargs.setdefault("parse_docstring", True)
        kwargs.setdefault("error_on_invalid_docstring", False)
    return _base_tool(*args, **kwargs)
