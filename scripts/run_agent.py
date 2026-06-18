"""Universal CLI entry point for the multi-agent supervisor.

This is the **primary** way most users should interact with the
system.  It accepts any natural-language instruction, picks the
best execution mode (ReAct by default), runs the supervisor, and
prints the final answer in a human-readable way.

Usage::

    # 1. Direct (REPL-style)
    python -m scripts.run_agent "关闭除当前外的其他标签"
    python -m scripts.run_agent "去 arxiv 搜 5 篇企业风险预测论文"
    python -m scripts.run_agent "打开百度,搜索 1924 年建立的中国大学"
    python -m scripts.run_agent "去京东找 iPhone 16 Pro 价格"

    # 2. Explicit mode
    python -m scripts.run_agent "..." --mode plan
    python -m scripts.run_agent "..." --mode react   (default)

    # 3. JSON output (for piping)
    python -m scripts.run_agent "..." --json

    # 4. From another Python script
    from scripts.run_agent import run_once
    result = run_once("...", mode="react", verbose=True)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _bootstrap_path() -> None:
    """Add the AI_web project root + browser-harness to sys.path.

    This is necessary when invoking the script directly via
    ``python scripts/run_agent.py`` (rather than ``python -m``).
    """
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # AI_web/
    for p in (project_root, project_root.parent / "browser-harness" / "src"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap_path()


def _attach_live_progress_handler() -> None:
    """Stream every supervisor / middleware event to stdout in real time.

    Without this, a multi-minute LLM task would print nothing until
    the final answer is ready, which feels frozen.  We attach a
    :class:`logging.Handler` that filters for the framework's key
    event loggers (``agent.supervisor.*`` and ``agent.middleware.*``)
    and prints a short, single-line, flushed progress message for
    each one.

    Idempotent: registering twice is a no-op.
    """
    import logging

    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_live_progress_marker", False):
            return  # already attached

    # Only show the framework events; quiet down DEBUG chatter.
    interesting = (
        "agent.supervisor",
        "agent.middleware",
        "agent.react",
        "agent.direction",
        "agent.refiner",
        "agent.classifier",
        "agent.operation",
    )

    class _ProgressHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            if not any(record.name.startswith(p) for p in interesting):
                return
            msg = record.getMessage()
            # Collapse multi-line structured-field payloads to a short
            # "key=value" preview; the user wants progress, not JSON.
            short = msg.replace("\n", " ⏎ ")
            if len(short) > 220:
                short = short[:217] + "..."
            # Timestamp in HH:MM:SS so the user can see liveness.
            t = time.strftime("%H:%M:%S")
            print(f"  [{t}] {record.name:<32s} | {short}", flush=True)

    h = _ProgressHandler()
    h._live_progress_marker = True  # type: ignore[attr-defined]
    h.setLevel(logging.INFO)
    root.addHandler(h)


def _load_dotenv_early() -> None:
    """Load .env at import time so other modules see XF_API_KEY.

    Must run BEFORE ``from config.config import ...`` so that
    ``xf_api_key`` is not bound to the empty default.
    """
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


_load_dotenv_early()


def run_once(user_input: str, mode: str = "react", verbose: bool = True) -> dict:
    """Run the supervisor once and return the result dict.

    Args:
        user_input: the raw user instruction in any language.
        mode: "react" (default — observe/reason/act loop) or "plan"
            (LLM-generated 3-step plan + operation master).
        verbose: when True, prints progress to stdout.

    Returns:
        A dict containing at least ``final_answer``, ``error``,
        ``refined``, ``react_history``, ``subagent_history``.
    """
    from agents.supervisor import run
    from tools._logging import setup_logging
    import os

    # .env was loaded at import time (see _load_dotenv_early); this is
    # a sanity check.
    if not os.environ.get("XF_API_KEY"):
        sys.stderr.write(
            "ERROR: XF_API_KEY not set. Make sure .env exists at the project root.\n"
        )
        return {"error": "missing XF_API_KEY", "final_answer": "(no credentials)"}

    setup_logging()

    # Attach a live progress handler that mirrors every framework
    # event (NODE_IN/OUT, TOOL_CALL/OK/ERR, LLM call, STEP start/end)
    # to stdout with flush=True, so the user sees the system working
    # in real time.  Without this, an 18-minute task looks frozen
    # until the final answer appears.
    if verbose:
        _attach_live_progress_handler()

    t0 = time.time()
    if verbose:
        print(f"\n>>> 指令: {user_input}\n", flush=True)
        print(f">>> 模式: {mode}\n", flush=True)
    result = run(user_input, mode=mode)
    result.setdefault("user_input", user_input)
    result["_elapsed_s"] = round(time.time() - t0, 2)

    if verbose:
        print()
        print("=" * 64)
        print(f"FINAL ANSWER ({result['_elapsed_s']}s)")
        print("=" * 64)
        print(result.get("final_answer") or "(empty)")
        print()
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        if result.get("react_history"):
            print()
            print("REACT LOOPS:")
            for i, d in enumerate(result["react_history"]):
                print(f"  iter{i+1:02d}: action={d.get('action')!s:8s} "
                      f"assignee={d.get('assignee')!s:8s} "
                      f"progress={d.get('progress_estimate')}")
        if result.get("subagent_history"):
            print()
            print("STEPS:")
            for i, h in enumerate(result["subagent_history"]):
                print(f"  step{i+1:02d}: subagent={h.get('subagent')!s:8s} "
                      f"status={h.get('status')!s:6s} "
                      f"err_cat={h.get('error_category')!s:24s}")

    return result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="通用多智能体浏览器自动化 (默认 react 模式)",
    )
    p.add_argument(
        "instruction",
        help="自然语言指令 (中文/英文均可)",
    )
    p.add_argument(
        "--mode", choices=("plan", "react"), default="react",
        help="执行模式: react (默认, LLM 自驱循环) | plan (固定3步计划)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出最终结果(适合管道处理)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="不打印中间过程,只输出 final_answer",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    result = run_once(args.instruction, mode=args.mode, verbose=not args.quiet)

    if args.json:
        # Strip very large fields for the JSON output
        out = {k: v for k, v in result.items()
               if k not in ("messages",)}
        out["react_history"] = [
            {k: v for k, v in (d or {}).items() if k != "raw_payload"}
            for d in (result.get("react_history") or [])
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
