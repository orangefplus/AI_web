"""Multi-agent picks top 5 papers from the 50 extracted.

This test is FAST (no more expensive browser navigation/JS calls):
  1. The list of 50 papers is already in _stage11_papers.json
  2. We feed the list to the supervisor in react mode with a tiny
     goal: read the list, pick the 5 most authoritative
     enterprise-risk-prediction papers, return titles.

This proves the multi-agent can do the *reasoning* step in a few
seconds rather than the 376s of the previous run.
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

from agents.supervisor import run  # noqa: E402


def main() -> int:
    src = ROOT / "downloads_test" / "_stage11_papers.json"
    papers = json.loads(src.read_text(encoding="utf-8"))
    print(f"[init] loaded {len(papers)} papers from {src.name}")

    # Format as a compact reference list for the multi-agent.
    paper_list = "\n".join(
        f"{p['i']+1:2d}. {p['title']}  ({p['abs_url']})"
        for p in papers
    )

    goal = (
        "我已经在 arxiv 搜索结果页上提取了 50 篇关于『enterprise risk prediction』的论文。"
        "**请不要重新打开浏览器,不要再调用任何 browser_* 工具**。\n\n"
        "【论文列表】(来自 https://arxiv.org/search/?query=enterprise+risk+prediction)\n"
        f"{paper_list}\n\n"
        "【任务】\n"
        "从中挑选**最权威最相关**的 5 篇企业风险预测相关论文,要求:\n"
        "  - 优先选: 综述/高引用/经典方法/近 2 年工作/专门针对 enterprise / corporate /\n"
        "    firm-level / bankruptcy / financial distress 主题的论文\n"
        "  - 排除: 只是网络安全/AI agents/IoT/医疗等不相关主题的论文\n"
        "  - 用 final_answer **只**返回 5 行,每行格式: `编号. 论文标题`\n"
    )

    print("\n--- supervisor.run(goal=..., mode='react') ---")
    t0 = time.time()
    result = run(goal, mode="react")
    dt = time.time() - t0

    print()
    print("=" * 60)
    print(f"elapsed: {round(dt, 2)}s")
    print("=" * 60)
    err = result.get("error")
    print(f"error: {err}")
    print()
    print("=" * 60)
    print("FINAL ANSWER:")
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
              f"err_cat={h.get('error_category')!s:24s}")
        print(f"           data_keys={data_keys}")
    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())
