"""End-to-end ReAct test: navigate to Weibo hot search, pick the top 5
trending topics, click into each, and grab the top 5 posts per topic.

This is a real multi-step browser task that exercises:

  - navigation           (browser_navigate)
  - text reading         (browser_read_page_text)
  - clicking             (browser_click_xy / browser_run_js)
  - loop-and-back        (click topic -> read top 5 -> go back)
  - error recovery       (ReAct re-thinks when click misses)
  - extraction           (browser_read_page_text)
"""
from __future__ import annotations

import json
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
    setup_browser,
)
from agents.supervisor import run  # noqa: E402


GOAL = (
    "打开微博，找到微博热搜榜（热搜榜的 URL 是 https://s.weibo.com/top/summary），"
    "从前 5 条热搜中点进每一条，进入每个热搜话题页面后抓取热度最高的 5 条帖子"
    "（帖子的作者昵称、帖子正文片段、点赞数）。最后把所有结果按话题整理成清单返回。"
)


def main() -> int:
    print("=" * 60)
    print("ReAct 端到端：微博热搜榜前 5 话题 -> 每话题前 5 帖")
    print("=" * 60)

    # Clean slate.
    setup_browser(wait=10.0)
    r0 = browser_close_other_tabs.invoke({})
    print("[setup] tabs:", len(r0.get("remaining", [])))

    # Run supervisor in ReAct mode.
    print()
    print(f"--- supervisor.run(goal=..., mode='react') ---")
    t0 = time.time()
    result = run(GOAL, mode="react")
    dt = time.time() - t0

    # Report.
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print("elapsed_s:", round(dt, 2))
    print("error:", result.get("error"))
    print("final_answer:", result.get("final_answer", ""))
    print()
    print("react_history_len:", len(result.get("react_history") or []))
    print("subagent_history_len:", len(result.get("subagent_history") or []))
    print()
    print("react_history (truncated):")
    for i, d in enumerate(result.get("react_history") or []):
        print(f"  iter{i+1:02d}: action={d.get('action')!s:8s} "
              f"assignee={d.get('assignee')!s:8s} "
              f"progress={d.get('progress_estimate')}")
        print(f"           reason: {d.get('rationale','')[:120]}")
    print()
    print("subagent_history (truncated):")
    for i, h in enumerate(result.get("subagent_history") or []):
        data_keys = list((h.get("data") or {}).keys())[:6]
        print(f"  step{i+1:02d}: subagent={h.get('subagent')!s:8s} "
              f"status={h.get('status')!s:6s} task={(h.get('task') or '')[:50]}")
        print(f"           data_keys={data_keys}")

    # Final browser state.
    after = browser_list_tabs.invoke({})
    print()
    print("FINAL tabs:", len(after))
    for t in after:
        print("    -", (t.get("title") or "?")[:30], "|", (t.get("url") or "?")[:80])
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
