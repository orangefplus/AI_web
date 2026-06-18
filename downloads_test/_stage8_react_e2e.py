"""End-to-end ReAct test: validate the observe→reason→act loop actually
drives the real browser via the supervisor(..., mode="react").

Scenario: seed the browser with several tabs, then ask the ReAct master
to "close all but the active tab". We expect:

    iterate 1: react thinks "observe first"
               -> dispatches observe agent
               -> execute runs browser_get_page_info / browser_list_tabs
    iterate 2: react thinks "tab agent should call browser_close_other_tabs"
               -> dispatches tab agent
               -> execute closes the tabs
    iterate 3: react confirms goal met
               -> action=stop
               -> finalize

The harness is shared with stage 7 so the result is directly comparable
to the plan-based run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "browser-harness" / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from tools import (  # noqa: E402
    browser_close_other_tabs,
    browser_list_tabs,
    browser_new_tab,
    setup_browser,
)
from agents.supervisor import run  # noqa: E402


GOAL = "仅保留当前活动窗口,删掉其余的窗口"


def main() -> int:
    print("=" * 60)
    print("ReAct mode end-to-end test")
    print("=" * 60)

    # 1. Clean slate.
    setup_browser(wait=10.0)
    r0 = browser_close_other_tabs.invoke({})
    print("[setup] tabs before seeding:", len(r0.get("remaining", [])))

    # 2. Seed a few tabs.
    browser_new_tab.invoke({"url": "https://www.baidu.com"})
    browser_new_tab.invoke({"url": "https://www.zhihu.com"})
    browser_new_tab.invoke({"url": "https://github.com"})
    time.sleep(0.3)
    seeded = browser_list_tabs.invoke({})
    print("[setup] tabs after seed:", len(seeded))
    for t in seeded:
        print("    -", (t.get("title") or "?")[:30], "|", (t.get("url") or "?")[:60])

    # 3. Run supervisor in ReAct mode.
    print()
    print(f"--- supervisor.run(goal={GOAL!r}, mode='react') ---")
    t0 = time.time()
    result = run(GOAL, mode="react")
    dt = time.time() - t0

    # 4. Report.
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print("elapsed_s:", round(dt, 2))
    print("final_answer:", result.get("final_answer", ""))
    print("react_history_len:", len(result.get("react_history") or []))
    print("subagent_history_len:", len(result.get("subagent_history") or []))
    print("react_history:")
    for i, d in enumerate(result.get("react_history") or []):
        print(f"  iter{i+1:02d}: action={d.get('action')!s:8s} "
              f"assignee={d.get('assignee')!s:8s} "
              f"progress={d.get('progress_estimate')} "
              f"reason={d.get('rationale','')[:80]}")
    print("subagent_history:")
    for i, h in enumerate(result.get("subagent_history") or []):
        print(f"  step{i+1:02d}: subagent={h.get('subagent')!s:8s} "
              f"status={h.get('status')!s:6s} "
              f"task={(h.get('task') or '')[:60]}")
    print()
    print("error:", result.get("error"))

    # 5. Verify browser state.
    after = browser_list_tabs.invoke({})
    print()
    print("BEFORE tabs:", len(seeded))
    print("AFTER  tabs:", len(after))
    for t in after:
        print("    -", (t.get("title") or "?")[:30], "|", (t.get("url") or "?")[:60])

    return 0 if (len(after) == 1 and not result.get("error")) else 1


if __name__ == "__main__":
    sys.exit(main())
