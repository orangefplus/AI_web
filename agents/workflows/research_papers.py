"""Research-papers workflow.

Top-level entry point: :func:`run_research_papers_workflow`.

The workflow:

1. Builds the supervisor.
2. Feeds it a user input that includes the topic, count, and
   download/summary flags.
3. Returns the supervisor's final state so the caller can render
   papers, summaries, and downloaded PDF paths.

For the first iteration we rely on the supervisor's plan template
in :mod:`agents.task_planner` so we do not duplicate logic. A
future iteration can override specific steps (e.g. to call a
domain-specific extractor) without changing this file.
"""
from __future__ import annotations

from typing import Any, Optional

from agents.intent_router import Intent
from agents.supervisor import build_supervisor
from tools._logging import setup_logging


def build_user_input(
    topic: str,
    count: int = 3,
    *,
    download: bool = True,
    must_be_published: bool = True,
    summary: bool = True,
    language: str = "zh",
) -> str:
    """Compose the user input string the supervisor sees."""
    parts = [f"找 {count} 篇关于 {topic} 的"]
    if must_be_published:
        parts.append("已发表")
    parts.append("期刊或会议论文")
    if download:
        parts.append("并下载 PDF")
    if summary:
        parts.append("，每篇生成中文摘要")
    return "".join(parts)


def run_research_papers_workflow(
    topic: str,
    count: int = 3,
    *,
    download: bool = True,
    must_be_published: bool = True,
    summary: bool = True,
    language: str = "zh",
) -> dict[str, Any]:
    """Run the multi-agent research-papers workflow.

    Args:
        topic: Free-form research topic, e.g. "企业风险预测".
        count: How many papers to surface.
        download: Whether to download PDFs locally.
        must_be_published: Skip arXiv preprints and prefer
            journal/conference papers.
        summary: Generate a short Chinese summary per paper.
        language: Output language for summaries.

    Returns:
        The supervisor's final state dict, including the
        ``scratchpad`` with paper metadata and download paths.
    """
    setup_logging()
    user_input = build_user_input(
        topic,
        count,
        download=download,
        must_be_published=must_be_published,
        summary=summary,
        language=language,
    )

    app = build_supervisor()
    final = app.invoke({"user_input": user_input, "iteration_count": 0})
    final.setdefault("user_input", user_input)
    final.setdefault("topic", topic)
    return final


__all__ = [
    "build_user_input",
    "run_research_papers_workflow",
]
