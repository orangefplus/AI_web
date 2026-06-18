"""Intent recognition for the multi-agent supervisor.

The router turns free-form user input into a typed :class:`Intent`
object so the planner and sub-agents can reason about it without
re-parsing the original sentence.

A small LLM call drives the classification. If the LLM call fails
(offline, malformed response, etc.) we fall back to keyword rules.
This is the same fallback philosophy the browser-harness agent
workspace uses, just at the intent layer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tools._logging import log_event


# ---------------------------------------------------------------------------
# Intent schema
# ---------------------------------------------------------------------------

Domain = Literal[
    "research_papers",
    "browse_summary",
    "shopping",
    "form_filling",
    "data_scraping",
    "general_browser_task",
    "unknown",
]


class Intent(BaseModel):
    """Structured representation of the user's request.

    Attributes:
        domain: Which high-level workflow the request belongs to.
        confidence: 0-1 score from the classifier.
        params: Domain-specific parameters (topic, count, etc.).
        needs_browser: True if the workflow requires the live
            browser session (vs. pure API calls).
        needs_api: True if the workflow can be served by an API.
        requires_summary: True if the final answer should be
            summarized/compared/insight'd (advanced feature).
    """

    domain: Domain = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    needs_browser: bool = True
    needs_api: bool = True
    requires_summary: bool = False

    def short(self) -> str:
        """Single-line summary used in key-event logs."""
        return f"INTENT {self.domain} (conf={self.confidence:.2f})"


# ---------------------------------------------------------------------------
# LLM-driven classifier
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """You are an intent classifier for a browser automation agent.

Return ONLY a JSON object with this schema:
{
  "domain": one of ["research_papers","browse_summary","shopping","form_filling","data_scraping","general_browser_task","unknown"],
  "confidence": float between 0 and 1,
  "params": object with domain-specific fields, see below,
  "needs_browser": bool,
  "needs_api": bool,
  "requires_summary": bool
}

Domain parameter hints:
- research_papers: {"topic": "<string>", "count": int, "must_be_published": bool, "download": bool, "language": "zh" | "en"}
- browse_summary: {"url": "<string>", "description": "<string>"} — the user wants to open a URL (or just "arxiv" / "the site") and read/summarise its landing page, NOT to search for papers.
- shopping: {"item": "<string>", "site": "<string>", "max_price": float, "currency": "CNY" | "USD"}
- form_filling: {"url": "<string>", "fields": {"name": "value"}}
- data_scraping: {"url": "<string>", "fields": ["title","price"], "max_pages": int}
- general_browser_task: {"description": "<string>"}
- unknown: {}

Rules:
- "download" or "下载" in the user request => download=true for research_papers.
- "summary"/"compare"/"insight"/"总结"/"比较"/"推论" => requires_summary=true.
- If the request mentions "published"/"已发表"/"期刊"/"会议" => must_be_published=true.
- Set needs_api=true for domains with known APIs (research_papers, data_scraping).
- Default needs_browser=true unless the request is clearly API-only.
- IMPORTANT: If the request says "打开 <URL or 站点>" and ALSO says "总结"/"介绍"/"看看"/"首页"/"看", and does NOT mention "论文/paper/找/搜/下载", classify as browse_summary (not research_papers). The user is asking to view one page, not to run a multi-step paper search.
- If the user only says "打开 <URL>" with no further intent, classify as browse_summary with requires_summary=false.

Respond with JSON only. No commentary, no markdown fences."""


def _parse_llm_json(raw: str) -> dict:
    """Robustly extract a JSON object from an LLM response."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty LLM response")
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # otherwise expect the first { ... } block
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"No JSON object found in LLM response: {raw[:120]!r}")


# ---------------------------------------------------------------------------
# Keyword fallback (used when LLM is unavailable)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[re.Pattern, dict]] = [
    # High-priority: pure "open URL + look at it" requests. The presence
    # of an "open + 总结/看/介绍" pattern with no "找/搜/论文" beats
    # every other rule, because it expresses a fundamentally different
    # workflow (one page snapshot vs. multi-step search).
    (re.compile(r"打开\s*\S+.*?(?:总结|介绍|看看|首页|看一下|看看)|(?:总结|介绍)\s*一下\s*\S+", re.I),
     {"domain": "browse_summary", "needs_api": False, "needs_browser": True}),
    (re.compile(r"论文|paper|article|research", re.I),
     {"domain": "research_papers", "needs_api": True, "needs_browser": True}),
    (re.compile(r"买|shop|buy|商品|产品", re.I),
     {"domain": "shopping", "needs_api": False, "needs_browser": True}),
    (re.compile(r"填表|填|form|submit", re.I),
     {"domain": "form_filling", "needs_api": False, "needs_browser": True}),
    (re.compile(r"爬|抓|scrape|extract", re.I),
     {"domain": "data_scraping", "needs_api": True, "needs_browser": True}),
]


def _keyword_intent(user_input: str) -> Intent:
    """Crude regex fallback when the LLM is offline / errors out."""
    params: dict = {}
    for pattern, base in _KEYWORD_RULES:
        if pattern.search(user_input):
            base = dict(base)
            params = _extract_inline_params(user_input, base["domain"])
            base["params"] = params
            return Intent(
                domain=base["domain"],
                confidence=0.4,
                params=base["params"],
                needs_browser=base.get("needs_browser", True),
                needs_api=base.get("needs_api", True),
                requires_summary=_wants_summary(user_input),
            )
    return Intent(domain="general_browser_task", confidence=0.3,
                  params={"description": user_input})


def _wants_summary(text: str) -> bool:
    return bool(re.search(r"总结|summary|比较|compare|对比|推论|insight|介绍", text, re.I))


def _extract_inline_params(text: str, domain: str) -> dict:
    """Pull cheap regex params out of the user input."""
    params: dict = {}
    # Always try to extract a URL — the user may have given us a
    # specific portal (e.g. a WebVPN gateway) and the planner must
    # honour that instead of falling back to a hard-coded entry.
    url = _extract_url(text)
    if url:
        params["url"] = url
    if domain == "research_papers":
        # topic: multiple patterns, first hit wins.
        #   关于 X 的论文
        #   about X papers/articles
        #   关键词 'X' / 关键词 X
        #   X 相关论文 / 找 X 论文
        patterns = [
            r"关于\s*(.+?)\s*(?:的|论文|paper)",
            r"about\s+(.+?)\s+(?:papers?|articles?)",
            r"关键词\s*[\"']?(.+?)[\"']?\s*(?:检索|搜索|搜|查询)",
            r"关键词\s+is\s+[\"']?(.+?)[\"']?",
            r"关键词\s*[:：]\s*[\"']?(.+?)[\"']?",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                params["topic"] = m.group(1).strip()
                break
        # count: "3 篇" / "3 papers" / "top 5"
        m = re.search(r"(?:top\s*)?(\d+)\s*(?:篇|papers?|articles?)", text, re.I)
        if m:
            params["count"] = int(m.group(1))
        params["must_be_published"] = bool(re.search(r"发表|published|期刊|会议|journal|conference", text, re.I))
        # Negative look-around for "不要下载" / "不下载" / "无需下载" so
        # the user can opt out of PDF downloads explicitly.
        if re.search(r"(不|不要|不用|无需|不用再)\s*(?:要\s*)?下载|不要\s*pdf|no\s+download|do\s+not\s+download", text, re.I):
            params["download"] = False
        else:
            params["download"] = bool(re.search(r"下载|download|pdf", text, re.I))
        params.setdefault("language", "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en")
    elif domain == "shopping":
        m = re.search(r"买\s*(.+?)(?:\s|$)", text)
        if m:
            params["item"] = m.group(1).strip()
    elif domain == "browse_summary":
        # The whole instruction is the description; the URL is already
        # extracted by the generic ``_extract_url`` above. We don't try
        # to do topic / count / download inference here — those are
        # research_papers concerns and would confuse the planner.
        params.setdefault("description", text.strip())
        # Detect "滚动 / 看看内容" hints so the planner can plan a
        # short scroll-and-read step instead of a hard stop on the
        # first viewport.
        if re.search(r"内容|看仔细|看完整|整页|滚动", text, re.I):
            params["scroll"] = True
    return params


# Match http(s) URLs while leaving Chinese punctuation alone. The
# trailing group stops at common delimiters so trailing punctuation
# like "。" or "," does not get swept into the URL.
_URL_RE = re.compile(
    r"https?://[^\s\u4e00-\u9fff，。；,;]+",
    re.I,
)


def _extract_url(text: str) -> str:
    """Return the first http(s) URL in ``text`` or ``""`` if none."""
    if not text:
        return ""
    m = _URL_RE.search(text)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class IntentRouter:
    """Classify user input into an :class:`Intent`.

    The router is intentionally cheap: one LLM call (or zero on
    fallback) per request. State is kept across calls in case
    follow-ups want to inherit domain context.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._last: Optional[Intent] = None

    def classify(self, user_input: str) -> Intent:
        """Classify the request; returns an :class:`Intent`.

        Order of operations:
        1. Try the LLM with a strict JSON prompt.
        2. On any error, fall back to keyword rules.
        3. Always run :func:`_extract_inline_params` on the result so
           cheap params (topic, count, must_be_published, download)
           are populated even when the LLM is unreachable.
        """
        if self._llm is not None:
            try:
                intent = self._llm_classify(user_input)
            except Exception as exc:  # pragma: no cover - LLM outage
                logging.getLogger("agent.supervisor.error").info(
                    "INTENT-LLM fallback (reason=%s)", exc
                )
                intent = _keyword_intent(user_input)
        else:
            intent = _keyword_intent(user_input)

        # Always try to enrich params via regex, even on the LLM path.
        inline = _extract_inline_params(user_input, intent.domain)
        for k, v in inline.items():
            intent.params.setdefault(k, v)
        intent.requires_summary = intent.requires_summary or _wants_summary(user_input)

        self._last = intent
        log_event(
            "agent.supervisor.intent",
            intent.short(),
            params=json.dumps(intent.params, ensure_ascii=False),
        )
        return intent

    def _llm_classify(self, user_input: str) -> Intent:
        """Single LLM call expecting strict JSON output."""
        llm = self._llm
        # Lazy imports so that unit tests can stub the LLM via ctor.
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        # Try the structured-output path first.
        if hasattr(llm, "with_structured_output"):
            try:
                structured = llm.with_structured_output(Intent)
                return structured.invoke(messages)
            except Exception:
                pass
        raw = llm.invoke(messages)
        text = getattr(raw, "content", str(raw))
        return Intent(**_parse_llm_json(text))


__all__ = ["Intent", "Domain", "IntentRouter"]
