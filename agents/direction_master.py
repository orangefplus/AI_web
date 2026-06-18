"""Direction Master (方向总智能体) — Layer 1 of the 4-layer system.

Role
----
The Direction Master is the *brain* of the multi-agent system. It is
invoked **before every single step** and **after every single step**:

* Before the step:  "given the current state and the Operation
  Master's pending action, should we proceed?"
* After the step:   "did the result move us closer to the user goal?"

It produces a structured :class:`DirectionVerdict` that the
supervisor uses to decide whether to continue, adjust, escalate to
the user, or terminate.

The Direction Master is intentionally *LLM-driven* so it can
interpret soft signals (LLM rambling, off-topic pages, dead loops).
For deterministic fall-back (LLM offline) it uses the simple
heuristics in :func:`_heuristic_verdict`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.prompts import DIRECTION_MASTER_PROMPT
from tools._logging import log_event


# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------

class DirectionVerdict(BaseModel):
    """Structured output of the Direction Master.

    Attributes:
        verdict: continue / adjust / stop / need_user.
        reason: One-sentence justification.
        direction_ok: Is the current execution on the user's goal?
        progress_pct: Estimated completion (0-100).
        adjustments: When ``verdict == 'adjust'``, list of fixes.
        next_directive: High-level instruction passed to the
            Operation Master on the next step.
    """

    verdict: str = "continue"
    reason: str = ""
    direction_ok: bool = True
    progress_pct: int = 0
    adjustments: list[str] = Field(default_factory=list)
    next_directive: Optional[str] = ""

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """Robustly extract a JSON object from an LLM response."""
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

    See :func:`agents.prompt_refiner._coerce_nulls_to_defaults` for the
    full rationale; duplicated here so each Master is self-contained.
    """
    out = dict(data)
    for field_name, field_info in model_cls.model_fields.items():
        if field_name not in out or out[field_name] is None:
            default = field_info.default
            if default is None and field_info.default_factory is not None:
                default = field_info.default_factory()
            out[field_name] = default
    return out


def _summarise_history(history: list[dict]) -> str:
    """Compact, LLM-friendly text dump of the last few steps."""
    if not history:
        return "(no prior steps)"
    lines: list[str] = []
    for h in history[-5:]:
        sub = h.get("subagent", "?")
        sid = h.get("step_id", "?")
        st = h.get("status", "?")
        data = h.get("data") or {}
        sig = data.get("url") or data.get("screenshot_path") or data.get("text") or data.get("summary") or ""
        sig = str(sig)[:80]
        lines.append(f"#{sid} [{sub}] {st} {sig}".rstrip())
    return "\n".join(lines)


def _heuristic_verdict(
    user_goal: str,
    history: list[dict],
    pending_action: Optional[dict],
) -> DirectionVerdict:
    """Cheap rule-based verdict used when the LLM is offline.

    The rules are deliberately conservative — when in doubt, continue.
    """
    if not history:
        return DirectionVerdict(
            verdict="continue", reason="first step, no history",
            direction_ok=True, progress_pct=0,
            next_directive=pending_action.get("task", "") if pending_action else "",
        )

    # Detect a long streak of failures.
    last3 = [h.get("status") for h in history[-3:]]
    if last3 == ["error", "error", "error"]:
        return DirectionVerdict(
            verdict="stop", reason="3 consecutive failures",
            direction_ok=False, progress_pct=min(50, len(history) * 5),
        )

    # Detect stalled progress (no successful data with data field).
    if len(history) >= 6 and all(h.get("status") == "ok" and not (h.get("data") or {}) for h in history[-3:]):
        return DirectionVerdict(
            verdict="adjust",
            reason="last 3 successful steps produced no data; switch to observe first",
            direction_ok=True, progress_pct=min(70, len(history) * 5),
            adjustments=["insert an observe step before the next click"],
        )

    return DirectionVerdict(
        verdict="continue", reason="heuristic continue",
        direction_ok=True, progress_pct=min(95, len(history) * 5),
        next_directive=pending_action.get("task", "") if pending_action else "",
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DirectionMaster:
    """Layer 1: 任务方向把控 / 监视.

    The :class:`DirectionMaster` is cheap to instantiate; pass ``llm=None``
    to fall back to the heuristic path.
    """

    name = "direction"

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def evaluate(
        self,
        *,
        user_goal: str,
        history: list[dict],
        current_state: Optional[dict] = None,
        pending_action: Optional[dict] = None,
        phase: str = "after",  # "before" or "after" a step
    ) -> DirectionVerdict:
        """Return a :class:`DirectionVerdict`.

        Args:
            user_goal: The (refined) user goal.
            history: Already-executed steps (most recent last).
            current_state: Browser snapshot dict, may be empty.
            pending_action: The Operation Master's proposal, if any.
            phase: "before" (about to act) or "after" (just acted).
        """
        if self._llm is not None:
            try:
                return self._llm_evaluate(
                    user_goal=user_goal,
                    history=history,
                    current_state=current_state or {},
                    pending_action=pending_action,
                    phase=phase,
                )
            except Exception as exc:  # pragma: no cover - LLM outage
                logging.getLogger("agent.direction.error").info(
                    "DIRECTION-LLM fallback (reason=%s)", exc
                )

        verdict = _heuristic_verdict(user_goal, history, pending_action)
        log_event(
            "agent.direction.verdict",
            f"DIRECTION {verdict.verdict} ({phase})",
            reason=verdict.reason,
            pct=verdict.progress_pct,
        )
        return verdict

    # -- LLM path ---------------------------------------------------------

    def _llm_evaluate(
        self,
        *,
        user_goal: str,
        history: list[dict],
        current_state: dict,
        pending_action: Optional[dict],
        phase: str,
    ) -> DirectionVerdict:
        from langchain_core.messages import HumanMessage, SystemMessage

        payload = {
            "user_goal": user_goal,
            "phase": phase,
            "current_state": {
                k: current_state.get(k)
                for k in ("url", "title", "scrollY", "viewport", "text_excerpt",
                         "tab_count", "active_tab_idx", "tabs")
                if k in current_state
            },
            "history": [
                {
                    "step_id": h.get("step_id"),
                    "subagent": h.get("subagent"),
                    "status": h.get("status"),
                    "data_keys": list((h.get("data") or {}).keys()),
                    "error": h.get("error"),
                }
                for h in history[-10:]
            ],
            "pending_action": pending_action,
        }
        messages = [
            SystemMessage(content=DIRECTION_MASTER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        raw = self._llm.invoke(messages)
        text = getattr(raw, "content", str(raw))
        data = _parse_json(text)
        data = _coerce_nulls_to_defaults(data, DirectionVerdict)
        allowed = set(DirectionVerdict.model_fields)
        verdict = DirectionVerdict(**{k: v for k, v in data.items() if k in allowed})
        log_event(
            "agent.direction.verdict",
            f"DIRECTION {verdict.verdict} ({phase})",
            reason=verdict.reason,
            pct=verdict.progress_pct,
        )
        return verdict


__all__ = ["DirectionMaster", "DirectionVerdict"]
