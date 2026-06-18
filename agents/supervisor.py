"""Supervisor — the new 4-layer multi-agent orchestrator.

The graph is::

    entry
      └─► refine              (Prompt Refiner,   Layer 2)
            └─► classify      (intent classification, kept from old flow)
                  └─► plan    (TaskPlanner, template → LLM → generic)
                        └─► direction_before  (Direction Master, Layer 1, before)
                              └─► operation   (Operation Master, Layer 3)
                                    └─► execute  (Layer-4 specialist)
                                          └─► direction_after   (Direction Master, after)
                                                └─► (loop or finalize)
finalize  (compose final answer)
                                                   └─► END

Key behavioural changes vs. the old supervisor:

* Every single step is gated by a *before* and *after* call to the
  Direction Master, so a stuck plan / dead-loop is detected as soon
  as it happens.
* The Operation Master decides which specialist (tab / click /
  observe / extract / verify) should run the next unit of work —
  the plan no longer hard-codes a subagent.
* Fallback dispatch is automatic: when an operation fails, the
  Operation Master's ``fallback_on_fail`` is consulted and the
  step is retried with the new assignee before surfacing to the
  user.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.direction_master import DirectionMaster, DirectionVerdict
from agents.intent_router import Intent, IntentRouter
from agents.operation_master import Operation, OperationMaster
from agents.prompt_refiner import PromptRefiner, RefinedPrompt
from agents.react_master import ReactDecision, ReactMaster
from agents.subagents import build_specialists, check_step_result
from agents.task_planner import Step, TaskPlanner
from config.config import chat_model_name, xf_api_key, xf_chat_base_url
from tools._logging import log_event, setup_logging
from tools._middleware import (
    wrap_llm,
    wrap_node,
    wrap_specialist_tools,
)


MAX_ITERATIONS = 50  # safety net


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Inputs
    user_input: str
    # Mode selector: "plan" (default) or "react" (ReAct loop, no fixed plan)
    mode: str
    # Layer-2 output
    refined: Optional[dict]
    # Layer-0/1 (kept for backward compat with the old supervisor API)
    intent: Optional[dict]
    # Plan (steps; ``subagent`` is now an *initial hint* and may be overridden)
    plan: list[dict]
    current_step_idx: int
    # Per-step direction & operation decisions
    direction_history: list[dict]
    operation_history: list[dict]
    pending_operation: Optional[dict]   # set by operation_node, consumed by execute_node
    # History
    subagent_history: list[dict]
    # ReAct state (mode == "react")
    react_decision: Optional[dict]
    react_history: list[dict]            # every ReactDecision seen so far
    react_streak: int                    # count of consecutive dispatches w/ no progress
    # Shared scratchpad
    scratchpad: dict
    # Final answer
    final_answer: Optional[str]
    error: Optional[str]
    iteration_count: int
    # Fallback tracking
    step_retries: dict
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# LLM factory (kept from old supervisor: tenacious retry on flaky proxy)
# ---------------------------------------------------------------------------

def default_llm():
    """Return the configured LLM, lazily imported.

    Wraps the base ``ChatOpenAI`` with a tenacity retry that handles
    the intermittent 502s / connection refusals / 60s timeouts we
    get from the 讯飞 one-api proxy.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log,
    )
    import logging

    base = ChatOpenAI(
        model=chat_model_name,
        api_key=xf_api_key,
        base_url=xf_chat_base_url,
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    log = logging.getLogger("agent.supervisor.llm_retry")

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def _invoke_with_retry(target, payload):
        return target(payload)

    class _RetryingLLM(Runnable):
        def __init__(self, target=None) -> None:
            # ``target`` is the underlying callable used by ``invoke``.
            # The root instance uses the bare ``base.invoke`` so the
            # retry wrapper is a drop-in replacement for ChatOpenAI.
            self._target = target or base.invoke

        def invoke(self, input, config=None, **kw):
            return _invoke_with_retry(self._target, input)

        async def ainvoke(self, input, config=None, **kw):
            return _invoke_with_retry(self._target, input)

        def bind_tools(self, tools, **kw):
            return _RetryingLLM(target=base.bind_tools(tools, **kw).invoke)

        def with_structured_output(self, schema, **kw):
            structured = base.with_structured_output(schema, **kw)
            return _RetryingLLM(target=structured.invoke)

        def __getattr__(self, name):
            return getattr(base, name)

    return _RetryingLLM()


# ---------------------------------------------------------------------------
# Browser-state helper (used by every node that needs "current state")
# ---------------------------------------------------------------------------

def _read_browser_state() -> dict:
    """Read a cheap browser snapshot; returns empty dict on failure."""
    try:
        from tools import BrowserSession  # local import: avoid daemon boot in tests
        with BrowserSession(screenshot_dim=1024) as session:
            snap = session.snapshot()
        return {
            "url": (snap.page_info or {}).get("url", ""),
            "title": (snap.page_info or {}).get("title", ""),
            "scrollY": (snap.page_info or {}).get("scrollY", 0),
            "viewport": (snap.page_info or {}).get("viewport", {}),
            "tabs": snap.tabs or [],
            "screenshot_path": snap.screenshot_path or "",
        }
    except Exception as exc:  # pragma: no cover
        logging.getLogger("agent.supervisor.error").info(
            "browser snapshot failed: %s", exc
        )
        return {}


def _state_step(state: AgentState) -> Optional[Step]:
    plan = state.get("plan") or []
    idx = state.get("current_step_idx", 0)
    if idx >= len(plan):
        return None
    return Step(**plan[idx])


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

# Each node reads shared resources (LLM, specialists, master agents) from
# a module-level context that ``build_supervisor`` populates before
# invoking the graph. This keeps the AgentState schema small and the
# nodes themselves pure ``(state) -> dict`` functions.
_RUNTIME_CTX: dict = {}


def _ctx() -> dict:
    """Return the active build_supervisor runtime context.

    Raises if no context is active so misuse is loud rather than silent.
    """
    if not _RUNTIME_CTX:
        raise RuntimeError(
            "supervisor runtime context is empty; "
            "call build_supervisor() before invoking any node."
        )
    return _RUNTIME_CTX


def refine_node(state: AgentState) -> dict:
    """Layer 2: 把用户原话打磨成 RefinedPrompt."""
    ctx = _ctx()
    refiner = ctx["refiner"]
    refined = refiner.refine(state["user_input"], context=state.get("scratchpad") or {})
    return {
        "refined": refined.model_dump(),
        "scratchpad": {
            **(state.get("scratchpad") or {}),
            "refined_goal": refined.refined_goal,
            "domain_hint": refined.domain_hint,
            "acceptance_criteria": refined.acceptance_criteria,
            "constraints": refined.constraints,
            "assumptions": refined.assumptions,
            "ambiguities": refined.ambiguities,
            "priority": refined.priority,
        },
    }


def classify_node(state: AgentState) -> dict:
    """Classify the refined intent (kept from the old flow)."""
    ctx = _ctx()
    router = ctx["router"]
    refined = RefinedPrompt(**state["refined"]) if state.get("refined") else None
    text = (refined.refined_goal if refined else state["user_input"])
    intent = router.classify(text)
    # Merge refined hints into intent.params.
    if refined:
        for k in ("topic", "count", "download", "must_be_published", "language"):
            if k in (refined.assumptions or []) or k in (refined.constraints or []):
                continue
        # Surface refined.domain_hint as the router's hint.
        if refined.domain_hint and intent.domain == "unknown":
            intent.domain = refined.domain_hint
    return {
        "intent": intent.model_dump(),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def plan_node(state: AgentState) -> dict:
    ctx = _ctx()
    intent = Intent(**state["intent"]) if state.get("intent") else Intent()
    planner = ctx["planner"]
    plan = planner.plan(intent)
    return {
        "plan": [s.model_dump() for s in plan],
        "current_step_idx": 0,
        "scratchpad": {
            **(state.get("scratchpad") or {}),
            "intent_params": intent.params,
        },
    }


def direction_before_node(state: AgentState) -> dict:
    """Layer 1: pre-step verdict. May terminate the loop."""
    ctx = _ctx()
    dm = ctx["direction"]
    step = _state_step(state)
    refined = RefinedPrompt(**state["refined"]) if state.get("refined") else None
    goal = refined.refined_goal if refined else state["user_input"]
    directive = (step.description if step else "")
    history = list(state.get("subagent_history") or [])

    if not step:
        # No plan left — Direction Master decides final fate.
        verdict = dm.evaluate(
            user_goal=goal, history=history, current_state={},
            pending_action=None, phase="after",
        )
    else:
        verdict = dm.evaluate(
            user_goal=goal, history=history, current_state=_read_browser_state(),
            pending_action={"task": directive, "step_id": step.step_id},
            phase="before",
        )

    dh = list(state.get("direction_history") or [])
    dh.append(verdict.model_dump())
    update = {"direction_history": dh}

    if verdict.verdict in ("stop", "need_user"):
        update["error"] = verdict.reason or verdict.verdict
    if verdict.verdict == "adjust" and verdict.adjustments:
        # Record the adjustment as a directive hint for the next operation.
        scratch = dict(state.get("scratchpad") or {})
        scratch["adjustments"] = list(scratch.get("adjustments") or []) + verdict.adjustments
        update["scratchpad"] = scratch
    return update


def operation_node(state: AgentState) -> dict:
    """Layer 3: produce an Operation for the current step."""
    ctx = _ctx()
    om = ctx["operation"]
    step = _state_step(state)
    refined = RefinedPrompt(**state["refined"]) if state.get("refined") else None
    goal = refined.refined_goal if refined else state["user_input"]
    directive = ""
    if step:
        directive = step.description
        # Incorporate Direction Master's adjustment hint.
        scratch = state.get("scratchpad") or {}
        if scratch.get("adjustments"):
            directive = directive + " 调整建议:" + ";".join(scratch["adjustments"][-2:])

    op = om.dispatch(
        directive=directive,
        refined_goal=goal,
        current_state=_read_browser_state(),
        history=list(state.get("subagent_history") or []),
    )
    # Plan-level hint: the planner wrote ``step.subagent`` (e.g. "browser",
    # "extractor") for the old 4-subagent world. The Operation Master's
    # LLM is the *authority* on which specialist to use — the old hint
    # is only consulted when the LLM had no opinion and picked
    # ``observe`` as a generic default. In that case we lift the plan's
    # ``subagent`` into the new vocabulary so the user still gets the
    # ``extractor``/``verifier`` specialists the planner expected.
    if step and step.subagent and op.assignee == "observe":
        hint_map = {
            "api": "extract",
            "extractor": "extract",
            "verifier": "verify",
            # NB: "browser" is intentionally NOT mapped to "click" —
            # the Operation Master can pick tab/click/observe as needed
            # based on the directive.  Forcing "click" here was the
            # root cause of the ReAct dead-loop on the "close other
            # tabs" task (the click agent has no tab-close tool).
        }
        if step.subagent in hint_map:
            op.assignee = hint_map[step.subagent]  # type: ignore[assignment]

    oh = list(state.get("operation_history") or [])
    oh.append(op.model_dump())
    return {
        "operation_history": oh,
        "pending_operation": op.model_dump(),
    }


def execute_node(state: AgentState) -> dict:
    """Layer 4: run the dispatched specialist."""
    ctx = _ctx()
    op_data = state.get("pending_operation")
    if not op_data:
        return {"error": "no pending operation"}
    op = Operation(**op_data)
    step = _state_step(state)
    is_react = state.get("mode") == "react"

    specialists = ctx["specialists"]
    if op.assignee not in specialists:
        # In ReAct mode we don't have a plan; just append an error step.
        step_id = (step.step_id if step else 0) or state.get("current_step_idx", 0)
        return {
            "subagent_history": list(state.get("subagent_history") or []) + [{
                "subagent": op.assignee, "step_id": step_id,
                "status": "error", "error": f"unknown assignee: {op.assignee}",
                "elapsed_ms": 0, "data": None,
            }],
            "error": f"unknown assignee: {op.assignee}",
            "current_step_idx": state.get("current_step_idx", 0) + (0 if is_react else 1),
        }

    sub = specialists[op.assignee]
    # Build a synthetic Step whose ``action`` carries the Operation payload.
    synth = Step(
        step_id=(step.step_id if step else 0) or state.get("current_step_idx", 0),
        description=op.task or (step.description if step else op.rationale),
        subagent=op.assignee,  # type: ignore[arg-type]
        action=op.rationale,
        expected_output={},
    )
    scratchpad = dict(state.get("scratchpad") or {})
    # Provide Operation hints as scratchpad inputs the specialist may read.
    if op.expected_signals:
        scratchpad["expected_signals"] = op.expected_signals
    scratchpad["refined_goal"] = (state.get("refined") or {}).get("refined_goal", "")

    result = sub.run(synth, scratchpad)
    history = list(state.get("subagent_history") or [])
    history.append(result)

    update: dict = {
        "subagent_history": history,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    # In plan mode, advance the step counter. In ReAct mode, the counter
    # is already incremented by the react node and used for observation
    # indexing; do not bump it again here.
    if not is_react:
        update["current_step_idx"] = state.get("current_step_idx", 0) + 1

    # Lift sub-agent ``data`` into the top-level scratchpad.
    scratchpad = _flatten_data(result, scratchpad)
    update["scratchpad"] = scratchpad

    # On error, consult fallback_on_fail and retry ONCE on the same step.
    if result.get("status") == "error" and op.fallback_on_fail:
        # In ReAct mode the fallback is consumed immediately; we just
        # rewrite pending_operation and let the next loop iteration
        # in react_node pick it up.
        if is_react:
            update["pending_operation"] = Operation(
                assignee=op.fallback_on_fail,
                rationale=f"react fallback from {op.assignee}",
                task=op.task,
                expected_signals=op.expected_signals,
                fallback_on_fail=None,
            ).model_dump()
            update.pop("error", None)
            return update
        retry_step = step
        if retry_step is None:
            # We are past the end of the plan; surface the error.
            update["error"] = result.get("error", "specialist failed")
            return update
        idx = state.get("current_step_idx", 0)
        retries = dict(state.get("step_retries") or {})
        if retries.get(str(idx), 0) < 1:
            retries[str(idx)] = retries.get(str(idx), 0) + 1
            update["step_retries"] = retries
            # Decrement the step index so the next loop iteration re-runs it.
            update["current_step_idx"] = idx
            # Override the pending operation with the fallback assignee.
            fallback_op = Operation(
                assignee=op.fallback_on_fail,
                rationale=f"fallback from {op.assignee}",
                task=op.task,
                expected_signals=op.expected_signals,
                fallback_on_fail=None,
            )
            update["pending_operation"] = fallback_op.model_dump()
            update.pop("subagent_history", None)  # discard the failed attempt
            update.pop("error", None)
    elif result.get("status") == "error":
        update["error"] = result.get("error", "specialist failed")

    return update


def direction_after_node(state: AgentState) -> dict:
    """Layer 1: post-step verdict."""
    ctx = _ctx()
    dm = ctx["direction"]
    refined = RefinedPrompt(**state["refined"]) if state.get("refined") else None
    goal = refined.refined_goal if refined else state["user_input"]
    history = list(state.get("subagent_history") or [])

    verdict: DirectionVerdict = dm.evaluate(
        user_goal=goal,
        history=history,
        current_state=_read_browser_state(),
        pending_action=None,
        phase="after",
    )

    dh = list(state.get("direction_history") or [])
    dh.append(verdict.model_dump())

    update: dict = {"direction_history": dh}
    if verdict.verdict in ("stop", "need_user") and not state.get("error"):
        update["error"] = verdict.reason or verdict.verdict
    return update


# ---------------------------------------------------------------------------
# ReAct nodes (mode == "react" only)
# ---------------------------------------------------------------------------

def react_node(state: AgentState) -> dict:
    """Layer 3.5 — single iteration of the ReAct loop.

    Inputs (read from state):
      * refined_goal         — the polished target
      * acceptance_criteria  — to know when the goal is met
      * subagent_history     — the latest observation (just executed)
      * iteration_count      — to enforce the safety cap

    Outputs:
      * react_decision       — the new ReactDecision (always set)
      * react_history        — appended with this decision
      * pending_operation    — when action == "dispatch"
      * error                — when action == "stop" or "ask_user"
    """
    ctx = _ctx()
    rm: ReactMaster = ctx["react"]
    refined = RefinedPrompt(**state["refined"]) if state.get("refined") else None
    goal = refined.refined_goal if refined else state["user_input"]
    history = list(state.get("subagent_history") or [])
    last_decision_dict = state.get("react_decision")
    last_decision = ReactDecision(**last_decision_dict) if last_decision_dict else None

    # Safety cap.
    if state.get("iteration_count", 0) > MAX_ITERATIONS:
        return {
            "error": f"ReAct loop exceeded {MAX_ITERATIONS} iterations",
            "react_decision": None,
        }

    # Build a rich observation: latest browser snapshot + last subagent
    # result + last decision. The ReAct master consumes this.
    current_state = _read_browser_state()
    last_observation = history[-1] if history else None
    if last_observation:
        # Surface the subagent's data slice so the LLM can use it.
        current_state["last_observation"] = {
            "subagent": last_observation.get("subagent"),
            "status": last_observation.get("status"),
            "task": last_observation.get("task"),
            "data": last_observation.get("data") or {},
            "error": last_observation.get("error"),
        }

    decision: ReactDecision = rm.think(
        refined_goal=goal,
        current_state=current_state,
        history=history,
        acceptance_criteria=(refined.acceptance_criteria if refined else []),
        ambiguities=(refined.ambiguities if refined else []),
    )

    rh = list(state.get("react_history") or [])
    rh.append(decision.model_dump())

    update: dict = {
        "react_decision": decision.model_dump(),
        "react_history": rh,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

    if decision.action == "dispatch":
        if not decision.assignee or not decision.task:
            update["error"] = "react decision 'dispatch' missing assignee/task"
            return update
        op = Operation(
            assignee=decision.assignee,
            rationale=decision.rationale,
            task=decision.task,
            expected_signals=decision.expected_signals or [],
            fallback_on_fail=decision.fallback_on_fail,
        )
        update["pending_operation"] = op.model_dump()
        # In ReAct mode we don't advance a fixed step counter — we just
        # keep a numeric scratch. Bump it so middleware can show progress.
        update["current_step_idx"] = state.get("current_step_idx", 0) + 1
        return update

    if decision.action == "stop":
        summary = decision.rationale or "react master says goal achieved"
        update["final_answer"] = summary
        update["react_streak"] = 0
        return update

    if decision.action == "ask_user":
        update["error"] = decision.question_for_user or "react master needs user input"
        update["react_streak"] = 0
        return update

    return update


def _route_after_react(state: AgentState) -> str:
    """ReAct mode routing — return next node name."""
    if state.get("error") or state.get("final_answer"):
        return "finalize"
    if state.get("pending_operation"):
        return "execute"
    if state.get("iteration_count", 0) > MAX_ITERATIONS:
        return "finalize"
    return "react"  # safety net; should not normally happen


def _route_after_execute_react(state: AgentState) -> str:
    """After execute in ReAct mode, loop back to react (no plan)."""
    if state.get("error") and not state.get("pending_operation"):
        return "finalize"
    return "react"


def finalizer_node(state: AgentState) -> dict:
    """Compose a user-facing answer from the scratchpad."""
    history = list(state.get("subagent_history") or [])
    scratchpad = state.get("scratchpad") or {}
    refined = state.get("refined") or {}
    direction_history = state.get("direction_history") or []
    react_history = list(state.get("react_history") or [])

    if state.get("error"):
        summary = (
            f"Task ended: {state['error']}. "
            f"Ran {len(history)} step(s). See subagent_history for detail."
        )
    else:
        summary_text = scratchpad.get("summary")
        if isinstance(summary_text, str) and summary_text.strip():
            summary = summary_text.strip()
        else:
            papers = scratchpad.get("papers") or scratchpad.get("candidates") or []
            if papers and isinstance(papers, list):
                lines = ["Research papers found:"]
                for p in papers[:5]:
                    if not isinstance(p, dict):
                        continue
                    title = p.get("title") or "(no title)"
                    detail = p.get("detail_url") or p.get("pdf_path") or ""
                    line = f" - {title}"
                    if detail:
                        line += f" / {detail}"
                    lines.append(line)
                summary = "\n".join(lines)
            else:
                goal = refined.get("refined_goal") or state.get("user_input", "")
                if react_history:
                    # ReAct mode: surface the last decision and how many
                    # observe→reason→act iterations we ran.
                    last = react_history[-1] or {}
                    summary = (
                        f"ReAct loop finished in {len(react_history)} iteration(s).\n"
                        f"Goal: {goal}\n"
                        f"Last decision: action={last.get('action')} "
                        f"assignee={last.get('assignee')} "
                        f"progress={last.get('progress_estimate')}\n"
                        f"Reason: {last.get('rationale', '')}"
                    )
                else:
                    summary = (
                        f"Task completed in {len(history)} step(s).\n"
                        f"Goal: {goal}\n"
                        f"Last direction verdict: {direction_history[-1] if direction_history else 'n/a'}"
                    )

    log_event("agent.supervisor.result", "DONE", steps=len(history))
    return {"final_answer": summary}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_direction_before(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("current_step_idx", 0) >= len(state.get("plan") or []):
        return "finalize"
    if state.get("iteration_count", 0) > MAX_ITERATIONS:
        log_event("agent.supervisor.error", "iteration cap exceeded", max=MAX_ITERATIONS)
        return "finalize"
    return "operation"


def _route_after_direction_after(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"
    # If we still have the same step queued (fallback retry), go to operation.
    if state.get("pending_operation") and state.get("current_step_idx", 0) < len(state.get("plan") or []):
        return "operation"
    if state.get("current_step_idx", 0) >= len(state.get("plan") or []):
        return "finalize"
    return "direction_before"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_supervisor(mode: str = "plan"):
    """Build (but do not invoke) the LangGraph supervisor.

    Args:
        mode: "plan" (default) — fixed 3-step plan + Operation Master.
              "react"           — plan-free ReAct loop, observe→reason→act.

    Both modes share the same specialists and the same entry
    (refine → classify).  They differ in the inner orchestration:

        plan: refine → classify → plan → direction_before → operation
                                                     ⇣
                                              execute → direction_after
                                                     ⇣
                                          direction_before / finalize

        react: refine → classify → react ⇄ execute → finalize
                    (no plan, no direction_before/after, react thinks
                     independently each loop)
    """
    if mode not in ("plan", "react"):
        raise ValueError(f"unknown supervisor mode: {mode!r}")

    setup_logging()
    graph = StateGraph(AgentState)

    # Wrap every node with the observability middleware so we get
    # NODE_IN/NODE_OUT log lines around every LangGraph transition.
    graph.add_node("refine", wrap_node("refine", refine_node))
    graph.add_node("classify", wrap_node("classify", classify_node))
    graph.add_node("plan", wrap_node("plan", plan_node))
    graph.add_node("direction_before", wrap_node("direction_before", direction_before_node))
    graph.add_node("operation", wrap_node("operation", operation_node))
    graph.add_node("execute", wrap_node("execute", execute_node))
    graph.add_node("direction_after", wrap_node("direction_after", direction_after_node))
    graph.add_node("react", wrap_node("react", react_node))
    graph.add_node("finalize", wrap_node("finalize", finalizer_node))

    graph.set_entry_point("refine")
    graph.add_edge("refine", "classify")

    if mode == "plan":
        graph.add_edge("classify", "plan")
        graph.add_edge("plan", "direction_before")
        graph.add_conditional_edges(
            "direction_before",
            _route_after_direction_before,
            {"operation": "operation", "finalize": "finalize"},
        )
        graph.add_edge("operation", "execute")
        graph.add_edge("execute", "direction_after")
        graph.add_conditional_edges(
            "direction_after",
            _route_after_direction_after,
            {"operation": "operation", "direction_before": "direction_before", "finalize": "finalize"},
        )
    else:  # react
        graph.add_edge("classify", "react")
        graph.add_conditional_edges(
            "react",
            _route_after_react,
            {"execute": "execute", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "execute",
            _route_after_execute_react,
            {"react": "react", "finalize": "finalize"},
        )

    graph.add_edge("finalize", END)

    compiled = graph.compile()
    compiled.exposed = True  # marker so the debug wrapper can find it
    _compiled_ref = compiled  # capture for the class below

    # Inject the LLM once so every node reuses the same retried client.
    llm = default_llm()
    # Hook middleware so every LLM call (master + specialist) is logged.
    wrap_llm(llm, label="supervisor")

    # Build the specialist roster and wrap their tools for logging.
    specialists = build_specialists(llm)
    wrap_specialist_tools(specialists)

    # Populate the module-level runtime context so the node callables
    # (which have to be ``(state) -> dict`` to fit LangGraph) can find
    # the shared masters and specialist dicts.
    _RUNTIME_CTX.clear()
    _RUNTIME_CTX.update({
        "llm": llm,
        "refiner": PromptRefiner(llm=llm),
        "router": IntentRouter(llm=llm),
        "planner": TaskPlanner(llm=llm),
        "direction": DirectionMaster(llm=llm),
        "operation": OperationMaster(llm=llm),
        "react": ReactMaster(llm=llm),
        "specialists": specialists,
        "mode": mode,
    })

    class _BoundGraph:
        compiled = _compiled_ref  # raw LangGraph StateGraph for streaming/debug

        def __init__(self) -> None:
            self.mode = mode
            self.llm = llm

        def invoke(self, input: dict, config=None, **kw) -> dict:
            input = dict(input or {})
            input.setdefault("_llm", llm)
            input.setdefault("mode", mode)
            return _compiled_ref.invoke(input, config=config, **kw)

    return _BoundGraph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_data(result: dict, scratchpad: dict) -> dict:
    """Lift the sub-agent's ``data`` into the top-level scratchpad."""
    scratchpad = dict(scratchpad)
    scratchpad["last_result"] = result
    data = result.get("data")
    if isinstance(data, dict):
        for k, v in data.items():
            if v not in (None, "", [], {}):
                scratchpad.setdefault(k, v)
    return scratchpad


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run(user_input: str, mode: str = "plan") -> dict:
    """Build the supervisor and run it once on ``user_input``.

    Args:
        user_input: Raw user request.
        mode: "plan" (default) — fixed 3-step plan + Operation Master.
              "react"           — plan-free ReAct loop.
    """
    app = build_supervisor(mode=mode)
    final = app.invoke({"user_input": user_input, "iteration_count": 0, "mode": mode})
    return final


__all__ = [
    "AgentState",
    "build_supervisor",
    "run",
    "default_llm",
]
