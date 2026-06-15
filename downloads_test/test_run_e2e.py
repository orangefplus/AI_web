"""End-to-end run with real browser daemon + real LLM.

This script:
  1. Boots the browser-harness daemon.
  2. Builds the supervisor.
  3. Feeds it the user's research request.
  4. Prints the structured log events as the workflow runs.
  5. Renders the final state for the operator.

It does NOT need a real Chrome window — the daemon accepts CDP commands
even if no browser tab is open. The browser step will only fail when the
LLM-driven sub-agent actually invokes a tool that needs a tab. We cap
the run to a small number of iterations so we can observe the intent
classification + plan generation in the wild.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# Make project root importable without install.
PROJECT_ROOT = Path(r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web")
HARNESS_SRC = PROJECT_ROOT / "browser-harness" / "src"
for p in (str(PROJECT_ROOT), str(HARNESS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    from tools._logging import setup_logging
    setup_logging(level="INFO", noisy_level="WARNING")

    # 1. Boot the daemon so browser tools work later in the flow.
    from tools._bootstrap import setup_browser
    print("\n[BOOT] starting browser-harness daemon...")
    try:
        setup_browser(wait=10.0)
        print("[BOOT] daemon ready")
    except Exception as exc:
        print(f"[BOOT] daemon not available: {type(exc).__name__}: {exc}")
        print("[BOOT] continuing without a live browser — only intent/plan will run")

    # 2. Build the supervisor (this also wires the LLM via config).
    from agents.supervisor import build_supervisor
    print("\n[INIT] compiling LangGraph supervisor...")
    app = build_supervisor()

    # 3. Compose a realistic user request.
    user_input = (
        "请打开 https://webvpn.swufe.edu.cn/https/77726476706e69737468656265737421e3f449932b317a1e7b0c9ce29b5b/?wrdrecordvisit=1781442174000 "
        "(这是西南财经大学 WebVPN 入口),"
        "让智能体使用这个页面里能跳转到的数据库 / 资源,"
        "挑一个可以搜英文期刊论文的数据库(例如 Web of Science / Scopus / EBSCO / ProQuest 之一),"
        "在里面用关键词 'enterprise risk prediction' 检索 2 篇期刊论文,"
        "找到论文后**不要下载,直接停留在论文详情页 / 摘要页**,"
        "最后把 2 篇论文的标题 / 作者 / 期刊 / 详情页 URL 记到 scratchpad。"
    )

    # 4. Run it. We do NOT cap iterations here — the dispatcher will hand
    #    off to the LLM-driven BrowserAgent which may need many round-trips.
    #    For a smoke test we can stop after the first plan/iter and inspect.
    print(f"\n[RUN] user_input = {user_input!r}")
    print("[RUN] invoking supervisor (this will call the LLM)...")

    final = app.invoke(
        {
            "user_input": user_input,
            "iteration_count": 0,
        },
        # Allow enough iterations for the full 6-step plan plus any verifier
        # round-trips. Each sub-agent step can take multiple LLM turns.
        config={"recursion_limit": 80},
    )

    # 5. Render the result.
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(f"  user_input   : {final.get('user_input', '')}")
    print(f"  intent       : {json.dumps(final.get('intent'), ensure_ascii=False)}")
    plan = final.get("plan") or []
    print(f"  plan steps   : {len(plan)}")
    for s in plan:
        print(f"    step {s.get('step_id'):>2} [{s.get('subagent'):>9}] {s.get('description')}")
    history = final.get("subagent_history") or []
    print(f"  steps run    : {len(history)}")
    for h in history:
        print(
            f"    step {h.get('step_id'):>2} [{h.get('subagent'):>9}] "
            f"status={h.get('status'):>6} elapsed={h.get('elapsed_ms', 0)}ms"
        )
    scratchpad = final.get("scratchpad") or {}
    # ``ensure_ascii=False`` lets us see CJK as-is; ``errors="replace"`` keeps
    # the GBK terminal from choking on emoji (e.g. Google Scholar's 🐴).
    scratchpad_repr = json.dumps(
        {k: v for k, v in scratchpad.items() if k != "last_result"},
        ensure_ascii=False,
        default=str,
    )[:1200].encode("gbk", errors="replace").decode("gbk", errors="replace")
    print(f"  scratchpad   : {scratchpad_repr}")
    print(f"  final_answer : {final.get('final_answer', '')}")
    if final.get("error"):
        print(f"  error        : {final.get('error')}")
    print("=" * 70)
    return 0 if not final.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
