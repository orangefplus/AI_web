"""Prompt Refiner (提示词智能体) — Layer 2 of the 4-layer system.

Role
----
The Prompt Refiner takes the user's raw (often vague, colloquial,
typo-laden) input and turns it into a *precise* spec that the
Operation Master can dispatch. It does **not** know about browser
internals — it only shapes language.

Output is a :class:`RefinedPrompt` with the fields:
    refined_goal, acceptance_criteria, constraints, assumptions,
    ambiguities, priority, domain_hint.

A small LLM call drives the refinement. If the LLM is offline, the
:func:`_keyword_refine` fallback uses cheap regex heuristics
borrowed from :mod:`agents.intent_router`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.prompts import PROMPT_REFINER_PROMPT
from tools._logging import log_event


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

Domain = str  # free-form, matched against the intent router's domains


class RefinedPrompt(BaseModel):
    """Structured, precise version of the user's request."""

    refined_goal: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    priority: str = "normal"
    domain_hint: str = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty LLM response")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"no JSON in: {raw[:120]!r}")


def _coerce_nulls_to_defaults(data: dict, model_cls) -> dict:
    """Replace ``None`` values in ``data`` with the model's defaults.

    LLMs frequently emit ``"next_directive": null`` for optional string
    fields. Pydantic v2 treats ``None`` as a *value* (not "missing") and
    fails the ``str`` field. Pre-processing to empty-string / empty-list
    keeps the LLM in charge of the structure while sidestepping the
    strict-mode rejection.
    """
    out = dict(data)
    for field_name, field_info in model_cls.model_fields.items():
        if field_name not in out or out[field_name] is None:
            default = field_info.default
            if default is None and field_info.default_factory is not None:
                default = field_info.default_factory()
            out[field_name] = default
    return out


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _keyword_refine(raw: str) -> RefinedPrompt:
    """Fallback refinement when the LLM is unavailable.

    Recognises the same domain signals as the old ``intent_router``
    fallback so behavior is at least consistent.
    """
    text = (raw or "").strip()
    domain = "general_browser_task"
    constraints: list[str] = []
    assumptions: list[str] = []

    # Domain detection
    if re.search(r"打开\s*\S+.*?(?:总结|介绍|看看|首页|看一下)", text, re.I):
        domain = "browse_summary"
    elif re.search(r"论文|paper|article|research", text, re.I):
        domain = "research_papers"
    elif re.search(r"买|shop|buy|商品|产品", text, re.I):
        domain = "shopping"
    elif re.search(r"填表|填|form|submit", text, re.I):
        domain = "form_filling"
    elif re.search(r"爬|抓|scrape|extract", text, re.I):
        domain = "data_scraping"

    # Default count / language
    m = re.search(r"(\d+)\s*(?:篇|papers?|articles?|个|条)", text, re.I)
    if m:
        assumptions.append(f"quantity default: {m.group(1)}")
    if _is_chinese(text):
        assumptions.append("output language: zh")

    # Acceptance criteria: at minimum, something should be returned.
    criteria = [
        "任务执行有可观察的中间结果(截图/页面文本/文件路径)。",
        "完成后有结构化 summary。",
    ]

    # Ambiguities
    ambiguities: list[str] = []
    if domain == "shopping" and not re.search(r"价格|价|price|\$|¥|元", text, re.I):
        ambiguities.append("未指定价格上限,默认不限。")
    if domain == "research_papers" and not re.search(r"下载|下载|download|pdf", text, re.I):
        ambiguities.append("未指定是否下载,默认仅列出详情页。")

    return RefinedPrompt(
        refined_goal=text or "(empty user input)",
        acceptance_criteria=criteria,
        constraints=constraints,
        assumptions=assumptions,
        ambiguities=ambiguities,
        priority="normal",
        domain_hint=domain,
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PromptRefiner:
    """Layer 2: 把用户原话打磨成精确可执行的目标。"""

    name = "refiner"

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def refine(self, raw_user_input: str, context: Optional[dict] = None) -> RefinedPrompt:
        """Return a :class:`RefinedPrompt` for ``raw_user_input``.

        Args:
            raw_user_input: The user's free-form sentence.
            context: Optional dict with browser state / history.
        """
        ctx = context or {}
        if self._llm is not None:
            try:
                return self._llm_refine(raw_user_input, ctx)
            except Exception as exc:  # pragma: no cover
                logging.getLogger("agent.refiner.error").info(
                    "REFINER-LLM fallback (reason=%s)", exc
                )
        refined = _keyword_refine(raw_user_input)
        log_event(
            "agent.refiner.result",
            f"REFINED domain={refined.domain_hint}",
            goal=refined.refined_goal[:80],
            ambiguities=len(refined.ambiguities),
        )
        return refined

    def _llm_refine(self, raw: str, ctx: dict) -> RefinedPrompt:
        from langchain_core.messages import HumanMessage, SystemMessage

        payload = {"raw_user_input": raw, "context": ctx}
        messages = [
            SystemMessage(content=PROMPT_REFINER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        text = self._llm.invoke(messages)
        text = getattr(text, "content", str(text))
        data = _parse_json(text)
        # LLM often returns null for optional string fields; coerce to "" so
        # Pydantic v2 (which is strict about str/None) does not reject.
        data = _coerce_nulls_to_defaults(data, RefinedPrompt)
        # Filter to the model's known fields; pydantic fills defaults for the rest.
        allowed = set(RefinedPrompt.model_fields)
        refined = RefinedPrompt(**{k: v for k, v in data.items() if k in allowed})

        # ------------------------------------------------------------------
        # Guardrail: prevent the LLM refiner from silently dropping
        # structured data (paper lists, JSON arrays, URLs, file paths)
        # that the downstream ReAct Master needs verbatim.
        #
        # Heuristic: if the raw input contains "data-shaped" content
        # (URLs, numbered lists, fenced code, JSON-looking blocks) and
        # the LLM's refined_goal is much shorter than the raw, we
        # suspect the LLM summarised away the data.  In that case
        # we append the raw input back to refined_goal so nothing
        # important is lost.
        # ------------------------------------------------------------------
        if raw and len(raw) > 200 and len(refined.refined_goal) < 0.6 * len(raw):
            raw_indicators = (
                "arxiv.org",
                "http://",
                "https://",
                "/abs/",
                ".json",
                ".pdf",
            )
            numbered_list = bool(re.search(r"^\s*\d{1,3}\.\s", raw, re.M))
            if any(s in raw for s in raw_indicators) or numbered_list:
                log_event(
                    "agent.refiner.guardrail",
                    "REFINER appended raw data to refined_goal (LLM truncated)",
                    raw_len=len(raw),
                    refined_len=len(refined.refined_goal),
                    sample=raw[:60],
                )
                refined = refined.model_copy(update={
                    "refined_goal": (
                        refined.refined_goal.rstrip()
                        + "\n\n【原始数据(必须保留)】\n" + raw
                    )
                })

        log_event(
            "agent.refiner.result",
            f"REFINED domain={refined.domain_hint}",
            goal=refined.refined_goal[:80],
            ambiguities=len(refined.ambiguities),
        )
        return refined


__all__ = ["PromptRefiner", "RefinedPrompt", "Domain"]
