"""End-to-end: arxiv 找 5 篇企业风险预测论文。

策略:
  1. 先用工具直接导航到 arxiv 公开搜索结果页(不浪费 LLM 调用去点搜索框)
  2. 让多智能体在当前页面:
       - observe (screenshot)
       - extract (browser_run_js 抽取论文标题/作者/链接)
       - stop
  3. final_answer 返回 5 篇论文题目
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
    browser_navigate,
    browser_new_tab,
    browser_screenshot,
    setup_browser,
)
from agents.supervisor import run  # noqa: E402


GOAL = (
    "我已经在 arxiv 的搜索结果页上,当前 URL 是类似\n"
    "  https://arxiv.org/search/?searchtype=all&query=enterprise+risk+prediction\n"
    "请按以下步骤完成:\n"
    "  1. **observe**: 调用 browser_screenshot + browser_get_page_info 拿到当前\n"
    "     页面状态。\n"
    "  2. **extract**: 调用 browser_extract_links(text_filter='', limit=50)\n"
    "     拿到所有链接,从中筛出所有 href 以 /abs/ 开头的链接(这些是论文\n"
    "     详情页),每个链接的 text 字段就是论文标题。\n"
    "  3. **observe** (可选): 滚动一下页面 browser_scroll(0, 1500) 让更多\n"
    "     论文加载出来(如果需要的话)。\n"
    "  4. 选最权威最相关的 5 篇(优先高引用/综述/经典方法/近年工作),\n"
    "     用 final_answer 直接返回论文标题列表,每行一篇。\n"
    "**整个流程不要开新 tab、也不要关任何 tab,直接在当前页操作。**"
)


def main() -> int:
    setup_browser(wait=10.0)

    # 直接导航到 arxiv 公开搜索结果页 - 用 search 分类的 q-fin.RM 也是 OK 的
    # 但用关键词搜索更精准
    target_url = (
        "https://arxiv.org/search/?searchtype=all&query=enterprise+risk+prediction&start=0"
    )
    print(f"[init] navigating to: {target_url}")
    nav = browser_navigate.invoke({"url": target_url})
    print(f"[init] nav result: {nav}")

    # 截一张初始图,给 ReAct Master 看
    shot = browser_screenshot.invoke({"full_page": False, "max_dim": 1800})
    print(f"[init] screenshot: {shot[:120]}")

    print()
    print(f"--- supervisor.run(goal=..., mode='react') ---")
    t0 = time.time()
    result = run(GOAL, mode="react")
    dt = time.time() - t0

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"elapsed_s: {round(dt, 2)}")
    err = result.get("error")
    print(f"error: {err}")
    print()
    print("=" * 60)
    print("FINAL ANSWER (paper titles the agent returned):")
    print("=" * 60)
    print(result.get("final_answer", ""))
    print()
    print("=" * 60)
    print("REACT HISTORY")
    print("=" * 60)
    for i, d in enumerate(result.get("react_history") or []):
        print(f"  iter{i+1:02d}: action={d.get('action')!s:8s} "
              f"assignee={d.get('assignee')!s:8s} "
              f"progress={d.get('progress_estimate')}")
    print()
    print("=" * 60)
    print("SUBAGENT HISTORY")
    print("=" * 60)
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
              f"err_cat={h.get('error_category')!s:24s} "
              f"err_short={h.get('error_short')!s}")
        print(f"           data_keys={data_keys}")
    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())
