"""Multi-agent picks top 5 - direct LLM call, no supervisor loop.

The supervisor's ReAct loop was truncated by the Refiner (long lists
in the user goal got lost).  For this specific reasoning task the
simplest path is a single LLM call: read the paper list, pick 5,
return the titles.
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

from agents.agent import chat_model_name, xf_api_key, xf_chat_base_url  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


def get_llm():
    return ChatOpenAI(
        model=chat_model_name,
        api_key=xf_api_key,
        base_url=xf_chat_base_url,
        temperature=0,
    )


def main() -> int:
    src = ROOT / "downloads_test" / "_stage11_papers.json"
    papers = json.loads(src.read_text(encoding="utf-8"))
    print(f"[init] loaded {len(papers)} papers")

    paper_list = "\n".join(
        f"[{p['i']+1}] {p['title']}\n    {p['abs_url']}\n    {p['authors'][:140]}"
        for p in papers
    )

    sys_prompt = (
        "你是一个金融科技/学术论文筛选专家。"
        "我会给你一个 arxiv 搜索 'enterprise risk prediction' 的前 50 个结果，"
        "请你从中挑出 5 篇最权威、最相关、最有代表性的企业风险预测相关论文，"
        "要求：\n"
        "  - 优先选：综述 / 高引用 / 经典方法 / 近 2 年工作 / "
        "    专门针对 enterprise / corporate / firm-level / "
        "    bankruptcy / financial distress / credit risk 主题的论文\n"
        "  - 排除：纯网络安全 / AI agents / IoT / 医疗 / 客户流失等不相关主题\n"
        "  - 输出格式：每行 1 篇，按权威度排序，"
        "    格式为「编号. 论文标题 — arxiv ID/URL — 1 句核心结论」\n"
    )

    user_prompt = (
        "【50 篇候选论文】\n"
        f"{paper_list}\n\n"
        "请挑出 5 篇最权威的企业风险预测论文。"
    )

    print("[init] calling LLM...")
    t0 = time.time()
    llm = get_llm()
    resp = llm.invoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    dt = time.time() - t0

    print(f"\n[init] LLM call took {round(dt, 2)}s\n")
    print("=" * 60)
    print("TOP 5 PAPERS")
    print("=" * 60)
    print(resp.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
