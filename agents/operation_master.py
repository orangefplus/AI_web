"""Operation Master (操作总智能体) — Layer 3 of the 4-layer system.

Role
----
The Operation Master turns the Direction Master's high-level directive
*plus* the Prompt Refiner's refined goal into a concrete
:class:`Operation`. An operation is a single, executable unit that
will be dispatched to **one** specialist sub-agent:

    tab      — tab management (new_tab / switch / close / etc.)
    click    — click / type / scroll / keyboard / upload
    observe  — screenshot / read text / wait / get page info
    extract  — pull structured data out of raw HTML/PDF/JSON
    verify   — check that the previous step's output is acceptable

The master picks exactly one assignee per call; it never bundles
multiple operations together. Bundling is the supervisor's job
(``dispatcher`` in the graph).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agents.prompts import OPERATION_MASTER_PROMPT
from tools._logging import log_event


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

Assignee = Literal["tab", "click", "observe", "extract", "verify"]


class Operation(BaseModel):
    """One executable unit of work for a single specialist sub-agent.

    Attributes:
        assignee: Which specialist sub-agent should run it.
        rationale: One-sentence explanation of the dispatch choice.
        task: Concrete task description handed to the specialist.
        expected_signals: Success signals the verifier should look for.
        fallback_on_fail: Where to route the work if it fails.
    """

    assignee: Assignee
    rationale: str = ""
    task: str
    expected_signals: list[str] = Field(default_factory=list)
    fallback_on_fail: Optional[Assignee] = None


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Regex -> assignee. First match wins. Order matters.
_DISPATCH_RULES: list[tuple[re.Pattern, Assignee]] = [
    (re.compile(r"\b(打开|切到|关闭|保留|列表|列出来|new[_ ]?tab|switch|close|list)\b", re.I), "tab"),
    (re.compile(r"\b(点|click|输入|typing|fill|按键|按|滚|scroll|上[传传]|upload|press|key)\b", re.I), "click"),
    (re.compile(r"\b(看|截图|观察|observe|screenshot|读|read|等待|wait|加载|load|信息|info)\b", re.I), "observe"),
    (re.compile(r"\b(提取|抽取|解析|extract|parse|pdf|ocr|json)\b", re.I), "extract"),
    (re.compile(r"\b(校验|检查|验证|verify|check|ok\?|是否达成)\b", re.I), "verify"),
]


def _heuristic_dispatch(
    directive: str,
    refined_goal: str,
    history_tail: list[dict],
) -> Operation:
    """Cheap rule-based dispatch when the LLM is offline.

    The rules intentionally re-use the *operation* wording (点/看/打开)
    rather than the user goal wording so the LLM can refine the goal
    language without confusing this layer.
    """
    text = (directive or refined_goal or "").strip()
    assignee: Assignee = "observe"  # safest default: look first
    for pattern, who in _DISPATCH_RULES:
        if pattern.search(text):
            assignee = who
            break

    # If the previous step was a click and the goal still mentions an
    # observation signal, follow up with observe.
    if history_tail and assignee == "click":
        last = history_tail[-1]
        if last.get("subagent") == "click" and last.get("status") == "ok":
            assignee = "observe"

    fallback: Optional[Assignee] = None
    if assignee == "click":
        fallback = "observe"
    elif assignee == "observe":
        fallback = "click"
    elif assignee == "tab":
        fallback = "observe"
    elif assignee == "extract":
        fallback = "observe"
    elif assignee == "verify":
        fallback = None  # verify failure -> surface to Direction Master

    return Operation(
        assignee=assignee,
        rationale=f"heuristic dispatch picked {assignee} from directive keywords",
        task=text or "(empty directive)",
        expected_signals=[],
        fallback_on_fail=fallback,
    )


# ---------------------------------------------------------------------------
# LLM helpers
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
    out = dict(data)
    for field_name, field_info in model_cls.model_fields.items():
        if field_name not in out or out[field_name] is None:
            default = field_info.default
            if default is None and field_info.default_factory is not None:
                default = field_info.default_factory()
            out[field_name] = default
    return out


def _history_tail(history: list[dict], n: int = 3) -> list[dict]:
    return [
        {
            "step_id": h.get("step_id"),
            "subagent": h.get("subagent"),
            "status": h.get("status"),
            "data_keys": list((h.get("data") or {}).keys()),
        }
        for h in history[-n:]
    ]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class OperationMaster:
    """Layer 3: 决定下一步要做什么浏览器操作并分派给单一细分智能体。"""

    name = "operation"

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def dispatch(
        self,
        *,
        directive: str,
        refined_goal: str,
        current_state: Optional[dict] = None,
        history: Optional[list[dict]] = None,
    ) -> Operation:
        """Return the next :class:`Operation` to execute.

        Args:
            directive: High-level instruction from the Direction Master.
            refined_goal: The Prompt Refiner's output goal.
            current_state: Optional browser snapshot dict.
            history: All previously executed steps.
        """
        history = history or []
        current_state = current_state or {}

        if self._llm is not None:
            try:
                return self._llm_dispatch(
                    directive=directive,
                    refined_goal=refined_goal,
                    current_state=current_state,
                    history=history,
                )
            except Exception as exc:  # pragma: no cover
                logging.getLogger("agent.operation.error").info(
                    "OPERATION-LLM fallback (reason=%s)", exc
                )

        op = _heuristic_dispatch(directive, refined_goal, _history_tail(history))
        log_event(
            "agent.operation.dispatch",
            f"OP -> {op.assignee}",
            rationale=op.rationale,
        )
        return op

    def _llm_dispatch(
        self,
        *,
        directive: str,
        refined_goal: str,
        current_state: dict,
        history: list[dict],
    ) -> Operation:
        from langchain_core.messages import HumanMessage, SystemMessage

        payload = {
            "directive": directive,
            "refined_goal": refined_goal,
            "current_state": {
                k: current_state.get(k)
                for k in ("url", "title", "scrollY", "viewport")
                if k in current_state
            },
            "history": _history_tail(history),
        }
        messages = [
            SystemMessage(content=OPERATION_MASTER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        text = self._llm.invoke(messages)
        text = getattr(text, "content", str(text))
        data = _parse_json(text)
        data = _coerce_nulls_to_defaults(data, Operation)
        allowed = set(Operation.model_fields)
        op = Operation(**{k: v for k, v in data.items() if k in allowed})
        log_event(
            "agent.operation.dispatch",
            f"OP -> {op.assignee}",
            rationale=op.rationale,
        )
        return op


__all__ = ["OperationMaster", "Operation", "Assignee"]
