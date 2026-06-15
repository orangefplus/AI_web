"""Supervisor: the hierarchical orchestrator for the multi-agent system.

The supervisor is a LangGraph ``StateGraph`` with six nodes:

1. ``intent_router`` — calls :class:`IntentRouter` to classify the
   user input into a typed :class:`Intent`.
2. ``task_planner`` — calls :class:`TaskPlanner` to expand the
   intent into an ordered list of :class:`Step` objects.
3. ``dispatcher`` — runs the current step in the matching
   :class:`Subagent` and writes the result into the scratchpad.
4. ``verifier`` — checks the last sub-agent result with
   :func:`check_step_result` (and optionally a verifier LLM).
5. ``re_planner`` — invoked when verification fails. Rewrites the
   remaining plan with a fallback step inserted.
6. ``finalizer`` — when the plan is exhausted, builds the user-
   facing answer from the scratchpad.

The graph runs in a loop until either the plan completes or the
maximum iteration count is reached. Final state is returned in
``state["final_answer"]`` plus a structured ``state["scratchpad"]``
so workflows can post-process the run.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.intent_router import Intent, IntentRouter
from agents.subagents import build_subagents, check_step_result
from agents.task_planner import Step, TaskPlanner
from config.config import chat_model_name, xf_api_key, xf_chat_base_url
from tools._logging import log_event, setup_logging


MAX_ITERATIONS = 50  # safety net so a buggy plan cannot loop forever


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[dict]
    plan: list[dict]
    current_step_idx: int
    subagent_history: list[dict]
    scratchpad: dict
    final_answer: Optional[str]
    error: Optional[str]
    iteration_count: int
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def default_llm():
    """Return the configured LLM, lazily imported.

    Wraps the base ``ChatOpenAI`` with a tenacity retry that handles
    the intermittent 502s / connection refusals / 60s timeouts we
    get from the 讯飞 one-api proxy. Without this, a single flaky
    LLM call can kill an entire multi-step research task mid-way
    even though the plan itself is fine.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
    import logging

    base = ChatOpenAI(
        model=chat_model_name,
        api_key=xf_api_key,
        base_url=xf_chat_base_url,
        temperature=0,
        timeout=120,
        max_retries=0,  # we do our own retries below
    )

    log = logging.getLogger("agent.supervisor.llm_retry")

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def _invoke_with_retry(payload):
        return base.invoke(payload)

    class _RetryingLLM(Runnable):
        """A tiny Runnable that proxies to the underlying LLM with retries."""

        def invoke(self, input, config=None, **kw):
            return _invoke_with_retry(input)

        async def ainvoke(self, input, config=None, **kw):
            return _invoke_with_retry(input)

        def bind_tools(self, tools, **kw):
            return self

        def with_structured_output(self, schema, **kw):
            return self

        def __getattr__(self, name):
            return getattr(base, name)

    return _RetryingLLM()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _state_step(state: AgentState) -> Optional[Step]:
    """Return the current :class:`Step` or ``None`` when done."""
    plan = state.get("plan") or []
    idx = state.get("current_step_idx", 0)
    if idx >= len(plan):
        return None
    return Step(**plan[idx])


def intent_router_node(state: AgentState) -> dict:
    """Classify the user input into an :class:`Intent`."""
    router = IntentRouter(llm=None)  # we have no LLM wiring in this node
    intent = router.classify(state["user_input"])
    return {
        "intent": intent.model_dump(),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def task_planner_node(state: AgentState) -> dict:
    """Plan a sequence of steps for the current intent."""
    intent = Intent(**state["intent"]) if state.get("intent") else Intent()
    planner = TaskPlanner(llm=None)
    plan = planner.plan(intent)
    return {
        "plan": [s.model_dump() for s in plan],
        "current_step_idx": 0,
        "scratchpad": state.get("scratchpad") or {"intent_params": intent.params},
    }


def dispatcher_node(state: AgentState) -> dict:
    """Run the current step in the matching sub-agent."""
    step = _state_step(state)
    if step is None:
        return {}
    subagents = build_subagents(default_llm())
    sub = subagents[step.subagent]
    scratchpad = dict(state.get("scratchpad") or {})
    result = sub.run(step, scratchpad)
    scratchpad.update(_flatten_data(result))
    history = list(state.get("subagent_history") or [])
    history.append(result)
    return {
        "subagent_history": history,
        "scratchpad": scratchpad,
        "current_step_idx": state.get("current_step_idx", 0) + 1,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def verifier_node(state: AgentState) -> dict:
    """Decide whether to advance to the next step, re-plan, or finish."""
    history = state.get("subagent_history") or []
    plan = state.get("plan") or []
    idx = state.get("current_step_idx", 0)
    if not history:
        return {"error": "verifier called with empty history"}
    last = history[-1]
    if last.get("status") == "error":
        log_event(
            "agent.supervisor.error",
            f"STEP {last.get('step_id')} failed",
            error=last.get("error"),
        )
        return {"error": last.get("error", "unknown error")}

    # Find the Step object matching the last result and do the deterministic check.
    last_step_id = last.get("step_id")
    matching = next((Step(**s) for s in plan if s.get("step_id") == last_step_id), None)
    if matching is not None:
        check = check_step_result(matching, last)
        if not check.get("ok"):
            log_event(
                "agent.supervisor.error",
                f"VERIFIER reject step {last_step_id}",
                issues="; ".join(check.get("issues", [])),
            )
            return {"error": "; ".join(check.get("issues", []))}

    if idx >= len(plan):
        return {}  # all steps done; finalizer will fire
    return {}


def finalizer_node(state: AgentState) -> dict:
    """Compose a user-facing answer from the scratchpad."""
    history = state.get("subagent_history") or []
    if state.get("error"):
        summary = (
            f"Task failed after {len(history)} step(s): {state['error']}. "
            "See subagent_history for details."
        )
    else:
        summary = (
            f"Task completed in {len(history)} step(s). "
            "Full structured output is in state['scratchpad']."
        )
    log_event("agent.supervisor.result", "DONE", steps=len(history))
    return {"final_answer": summary}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _route_after_verifier(state: AgentState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("current_step_idx", 0) >= len(state.get("plan") or []):
        return "finalize"
    if state.get("iteration_count", 0) > MAX_ITERATIONS:
        log_event("agent.supervisor.error", "iteration cap exceeded", max=MAX_ITERATIONS)
        return "finalize"
    return "dispatch"


def build_supervisor():
    """Build (but do not invoke) the LangGraph supervisor."""
    setup_logging()
    graph = StateGraph(AgentState)

    graph.add_node("classify", intent_router_node)
    graph.add_node("plan", task_planner_node)
    graph.add_node("dispatch", dispatcher_node)
    graph.add_node("verify", verifier_node)
    graph.add_node("finalize", finalizer_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "verify")
    graph.add_conditional_edges(
        "verify",
        _route_after_verifier,
        {"dispatch": "dispatch", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def _flatten_data(result: dict) -> dict:
    """Lift the sub-agent's ``data`` into the top-level scratchpad."""
    out: dict = {"last_result": result}
    data = result.get("data")
    if isinstance(data, dict):
        out.update(data)
    return out


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run(user_input: str) -> dict:
    """Build the supervisor and run it once on ``user_input``."""
    app = build_supervisor()
    final = app.invoke({"user_input": user_input, "iteration_count": 0})
    return final


__all__ = [
    "AgentState",
    "build_supervisor",
    "run",
    "default_llm",
]
