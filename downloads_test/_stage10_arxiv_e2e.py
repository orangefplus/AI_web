"""Search arxiv for authoritative papers on enterprise risk prediction.

Uses the ReAct supervisor in a NEW tab (the academic resource portal
stays open).  The flow:

  1. open a new tab so the academic portal is preserved
  2. navigate to https://arxiv.org/search/?searchtype=all&query=enterprise+risk+prediction
  3. read the result list (titles + authors + abstract links)
  4. pick the top 5 by relevance
  5. close the search tab; portal tab stays untouched
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "browser-harness" / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from tools import (  # noqa: E402
    browser_list_tabs,
    browser_close_tab,
    browser_new_tab,
    browser_navigate,
    setup_browser,
)
from agents.supervisor import run  # noqa: E402


GOAL = (
    "新开一个浏览器标签页（**绝对不要关掉**当前 SWUFE 学术资源门户那个标签页），\n"
    "然后**直接**导航到 arxiv 的搜索结果页:\n"
    "  https://arxiv.org/search/?searchtype=all&query=enterprise+risk+prediction\n"
    "等待搜索结果加载完毕,使用 browser_extract_links 提取结果页上所有指向 /abs/ 链接(每篇论文的详情页),\n"
    "从中挑出最权威最相关的 5 篇(优先选择高引用 / 综述 / 经典方法 / 近年工作),\n"
    "记录每篇的: 论文标题、作者列表、arxiv ID/链接、摘要核心结论 1-2 句。\n"
    "最后用 final_answer 返回结构化清单。**整个流程不要关掉任何 tab**。"
)


def main() -> int:
    setup_browser(wait=10.0)

    print("=" * 60)
    print("BEFORE: tabs already open")
    print("=" * 60)
    before = browser_list_tabs.invoke({"include_chrome": True})
    for t in before:
        print("  -", t.get("title", "")[:50], "|", t.get("url", "")[:80])

    print()
    print(f"--- supervisor.run(goal=..., mode='react') ---")
    t0 = time.time()
    result = run(GOAL, mode="react")
    dt = time.time() - t0

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print("elapsed_s:", round(dt, 2))
    print("error:", result.get("error"))
    print("final_answer:", result.get("final_answer", ""))
    print()
    print("react_history_len:", len(result.get("react_history") or []))
    for i, d in enumerate(result.get("react_history") or []):
        print(f"  iter{i+1:02d}: action={d.get('action')!s:8s} "
              f"assignee={d.get('assignee')!s:8s} "
              f"progress={d.get('progress_estimate')}")
    print()
    print("subagent_history_len:", len(result.get("subagent_history") or []))
    for i, h in enumerate(result.get("subagent_history") or []):
        d = h.get("data")
        if isinstance(d, dict):
            data_keys = list(d.keys())[:6]
        elif isinstance(d, list):
            data_keys = [f"list[{len(d)}]"]
        else:
            data_keys = [type(d).__name__]
        print(f"  step{i+1:02d}: subagent={h.get('subagent')!s:8s} "
              f"status={h.get('status')!s:6s} "
              f"err_cat={h.get('error_category')!s:24s}")
        print(f"           data_keys={data_keys}")

    print()
    after = browser_list_tabs.invoke({"include_chrome": True})
    print(f"AFTER: {len(after)} tabs (porta + research should both be here)")
    for t in after:
        print("  -", t.get("title", "")[:50], "|", t.get("url", "")[:80])
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
