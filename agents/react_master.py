"""ReAct Master — 真正的 Observe → Reason → Act 循环的思考者。

与 ``OperationMaster`` 不同,ReAct Master **不依赖预先制定的计划**:
它每一轮循环都会:

    1. OBSERVE: 读取当前浏览器状态(URL / 标题 / 可见文本 / tabs / 截图)
    2. REASON:  借助 LLM 综合 refined_goal + history + current_state,判断:
                   - 目标是否已经达成(action=stop)
                   - 需要用户决策(action=ask_user)
                   - 下一步应派给哪个 specialist,具体任务是什么(action=dispatch)
    3. ACT:     输出 :class:`ReactDecision`,supervisor 据此路由

设计目标:让 LLM 自己决定"下一步做什么",而不是执行固定计划,
更接近经典 ReAct 论文 (Yao et al. 2022) 的 thought/action/observation
循环,并为浏览器自动化场景做了工程化落地。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agents.prompts import REACT_MASTER_PROMPT
from tools._logging import log_event


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

Assignee = Literal["tab", "click", "observe", "extract", "verify"]
ReactAction = Literal["dispatch", "stop", "ask_user"]


class ReactDecision(BaseModel):
    """ReAct 思考者在每轮循环产出的下一步决策。

    Attributes:
        action: dispatch=派给某个 specialist 执行;
                stop=目标已达成,准备收尾;
                ask_user=需要人工输入(登录墙/验证码/歧义).
        assignee: 当 action=dispatch 时,选定的细分执行智能体.
        task: 给 specialist 的具体任务描述.
        rationale: 这一步决策的推理(对应 ReAct 中的 Thought).
        progress_estimate: 估算任务完成度 0-100.
        expected_signals: 期望观察到的成功信号(供 verifier 用).
        fallback_on_fail: 失败时的回退目标.
        question_for_user: 当 action=ask_user 时,向用户提出的问题.
    """

    action: ReactAction
    assignee: Optional[Assignee] = None
    task: Optional[str] = None
    rationale: str = ""
    progress_estimate: int = 0
    expected_signals: list[str] = Field(default_factory=list)
    fallback_on_fail: Optional[Assignee] = None
    question_for_user: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON parsing helpers
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
    """Normalise LLM JSON before pydantic validation.

    The LLM sometimes returns list-typed fields as a bare string, or
    uses ``"null"`` for fields that have a real default, or omits
    fields altogether.  This function:

      * converts ``None`` / missing fields to the model's default;
      * wraps a single string into a one-element list when the
        field annotation is ``List[...]``;
      * leaves properly-typed values alone.

    Without this, the supervisor falls back to the heuristic
    "keyword match" decision and throws away the LLM's actual
    reasoning whenever it gets a type slightly wrong.
    """
    import typing
    out = dict(data)
    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        is_list = (
            annotation is not None
            and (
                annotation is list
                or typing.get_origin(annotation) in (list, typing.List)
            )
        )
        val = out.get(field_name)
        # Coerce None / missing -> default.
        if val is None or field_name not in out:
            default = field_info.default
            if default is None and field_info.default_factory is not None:
                default = field_info.default_factory()
            out[field_name] = default
            continue
        # Coerce bare string into a 1-element list if the field is List.
        if is_list and isinstance(val, str):
            out[field_name] = [val]
        # Coerce dict-as-list: a single dict object the LLM meant as
        # a 1-element list of objects.
        elif is_list and isinstance(val, dict):
            out[field_name] = [val]
    return out


def _history_tail(history: list[dict], n: int = 6) -> list[dict]:
    """Return the last ``n`` history items in a compact form.

    The history item shape is::

        {
            "subagent": "click",
            "step_id": 2,
            "status": "error" | "ok" | "partial",
            "task": "click the search button",
            "data": {...} | None,           # may also be a list/str
            "error": "raw exception text",  # present iff status == error
            "error_category": "selector_null",   # NEW: from diagnose()
            "error_short": "DOM 元素未找到",    # NEW
            "error_recovery_hint": "use extract_links first",  # NEW
        }
    """
    compact = []
    for h in history[-n:]:
        d = h.get("data")
        if isinstance(d, dict):
            data_keys = list(d.keys())[:8]
            result_summary = d.get("summary") or d.get("ok") or d.get("result")
        else:
            data_keys = []
            result_summary = d
        compact.append({
            "step_id": h.get("step_id"),
            "subagent": h.get("subagent"),
            "status": h.get("status"),
            "task": h.get("task") or h.get("description"),
            "data_keys": data_keys,
            "result_summary": result_summary,
            "error": h.get("error"),
            "error_category": h.get("error_category"),
            "error_short": h.get("error_short"),
            "error_recovery_hint": h.get("error_recovery_hint"),
        })
    return compact


# ---------------------------------------------------------------------------
# Heuristic fallback (offline / LLM 异常时)
# ---------------------------------------------------------------------------

def _heuristic_react(
    refined_goal: str,
    history: list[dict],
    current_state: dict,
) -> ReactDecision:
    """Cheap rule-based ReAct when the LLM is offline.

    Strategy:
      - 如果历史上已经做过 observe -> 跟着上一个 specialist 走(避免死循环)
      - 否则从 refined_goal 关键词映射到 assignee
      - 永远 action=dispatch,除非连续 3 次相同失败
    """
    text = (refined_goal or "").strip()
    if not text:
        return ReactDecision(
            action="ask_user",
            rationale="heuristic: empty refined goal",
            question_for_user="请提供更具体的目标",
        )

    if history:
        tail = history[-3:]
        if len(tail) >= 3 and all(h.get("status") == "error" for h in tail):
            return ReactDecision(
                action="ask_user",
                rationale="heuristic: 连续 3 次失败,需要人工介入",
                question_for_user="自动执行连续失败,请提供更具体的指示或确认目标。",
            )

    keyword_map = [
        (re.compile(r"\b(打开|切到|关闭|保留|列表|列出来|new[_ ]?tab|switch|close|list)\b", re.I), "tab"),
        (re.compile(r"\b(点|click|输入|typing|fill|按键|按|滚|scroll|上[传传]|upload|press|key)\b", re.I), "click"),
        (re.compile(r"\b(看|截图|观察|observe|screenshot|读|read|等待|wait|加载|load|信息|info)\b", re.I), "observe"),
        (re.compile(r"\b(提取|抽取|解析|extract|parse|pdf|ocr|json)\b", re.I), "extract"),
        (re.compile(r"\b(校验|检查|验证|verify|check|ok\?|是否达成)\b", re.I), "verify"),
    ]
    assignee: Assignee = "observe"
    for pattern, who in keyword_map:
        if pattern.search(text):
            assignee = who
            break

    return ReactDecision(
        action="dispatch",
        assignee=assignee,
        task=text,
        rationale=f"heuristic: keyword match -> {assignee}",
        progress_estimate=min(len(history) * 15, 90),
        fallback_on_fail="observe" if assignee != "observe" else None,
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ReactMaster:
    """Layer 3.5 — Observe / Reason / Act 循环思考者。

    每一轮调用 ``think()`` 都必须:
      1. 观察 current_state(由调用方注入,可以是 browser snapshot)
      2. 思考 goal 是否达成、该派给谁
      3. 输出 ReactDecision
    """

    name = "react"

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def think(
        self,
        *,
        refined_goal: str,
        current_state: Optional[dict] = None,
        history: Optional[list[dict]] = None,
        acceptance_criteria: Optional[list[str]] = None,
        ambiguities: Optional[list[str]] = None,
    ) -> ReactDecision:
        """Return the next :class:`ReactDecision`.

        Args:
            refined_goal: The Prompt Refiner's polished goal.
            current_state: Browser snapshot dict.
            history: All previous step records.
            acceptance_criteria: From RefinedPrompt.
            ambiguities: From RefinedPrompt (if any).
        """
        history = history or []
        current_state = current_state or {}

        if self._llm is not None:
            try:
                return self._llm_think(
                    refined_goal=refined_goal,
                    current_state=current_state,
                    history=history,
                    acceptance_criteria=acceptance_criteria or [],
                    ambiguities=ambiguities or [],
                )
            except Exception as exc:  # pragma: no cover
                logging.getLogger("agent.react.error").info(
                    "REACT-LLM fallback (reason=%s)", exc
                )

        decision = _heuristic_react(refined_goal, history, current_state)
        log_event(
            "agent.react.think",
            f"REACT -> {decision.action} {decision.assignee or ''}".rstrip(),
            rationale=decision.rationale,
            progress=decision.progress_estimate,
        )
        return decision

    def _llm_think(
        self,
        *,
        refined_goal: str,
        current_state: dict,
        history: list[dict],
        acceptance_criteria: list[str],
        ambiguities: list[str],
    ) -> ReactDecision:
        from langchain_core.messages import HumanMessage, SystemMessage
        from tools._multimodal import (
            build_multimodal_human_content,
            extract_screenshot_paths,
            is_image_path,
        )

        # Build a rich, structured payload so the LLM can truly reason.
        payload = {
            "refined_goal": refined_goal,
            "acceptance_criteria": acceptance_criteria,
            "ambiguities": ambiguities,
            "current_state": {
                "url": current_state.get("url"),
                "title": current_state.get("title"),
                "scrollY": current_state.get("scrollY"),
                "tabs": [
                    {"active": t.get("active"), "url": t.get("url"), "title": t.get("title")}
                    for t in (current_state.get("tabs") or [])[:6]
                ],
                "text_excerpt": (current_state.get("text_excerpt") or "")[:600],
                "screenshot_path": current_state.get("screenshot_path"),
            },
            "history": _history_tail(history),
        }
        # Collect every screenshot the LLM should be allowed to see.
        screenshots: list[str] = []
        if is_image_path(current_state.get("screenshot_path")):
            screenshots.append(current_state.get("screenshot_path"))
        last_obs = current_state.get("last_observation") or {}
        last_data = last_obs.get("data") or {}
        for key in ("screenshot_path", "screenshot"):
            if is_image_path(last_data.get(key)):
                screenshots.append(last_data.get(key))
        # Also scan the very last step's data, in case supervisor stashed
        # the path elsewhere.
        screenshots.extend(extract_screenshot_paths({"current_state": current_state}))
        # Deduplicate, keep order.
        seen: set[str] = set()
        deduped: list[str] = []
        for s in screenshots:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        screenshots = deduped

        if screenshots:
            # The Xunfei MaaS model is multimodal — send the screenshot
            # alongside the structured JSON payload.  The LLM can then
            # *see* the page, not only read the text excerpt.
            text = (
                "【多模态输入】\n"
                "你既会收到一份结构化 JSON(包含 refined_goal、current_state、"
                "history 等),也会收到最多 3 张最新的浏览器截图(依次为最近一次"
                "截图 -> 上一轮 observe 截图 -> 当前 current_state 截图)。\n"
                "请**先用眼睛看截图**,再参考 JSON 中的文字。\n\n"
                "JSON payload:\n" + json.dumps(payload, ensure_ascii=False)
            )
            content_blocks = build_multimodal_human_content(text, screenshots)
            human_msg = HumanMessage(content=content_blocks)
        else:
            human_msg = HumanMessage(
                content=json.dumps(payload, ensure_ascii=False)
            )
        messages = [
            SystemMessage(content=REACT_MASTER_PROMPT),
            human_msg,
        ]
        text = self._llm.invoke(messages)
        text = getattr(text, "content", str(text))
        data = _parse_json(text)
        data = _coerce_nulls_to_defaults(data, ReactDecision)
        allowed = set(ReactDecision.model_fields)
        decision = ReactDecision(**{k: v for k, v in data.items() if k in allowed})

        log_event(
            "agent.react.think",
            f"REACT -> {decision.action} {decision.assignee or ''}".rstrip(),
            rationale=decision.rationale,
            progress=decision.progress_estimate,
        )
        return decision


__all__ = ["ReactMaster", "ReactDecision", "ReactAction", "Assignee"]
