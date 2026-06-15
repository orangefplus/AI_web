"""Task planner for the multi-agent supervisor.

Given an :class:`Intent`, the planner produces a list of
:class:`Step` objects that the supervisor's dispatcher walks
through in order. The planner is deliberately a thin layer: it
holds a small library of "plan templates" for each supported
domain, falls back to LLM planning, and finally falls back to a
generic 3-step plan.

Templates here are intentionally **goal-oriented, not procedural**:
each ``Step.action`` describes the destination (e.g. "land on the
detail page for these 2 papers") and lets the LLM-driven sub-agent
look at the page to figure out the next click. This keeps the
templates short and prevents them from going stale when a website
redesigns itself.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tools._logging import log_event

from .intent_router import Intent


Subagent = Literal["api", "browser", "extractor", "verifier"]


class Step(BaseModel):
    """One executable step in a plan.

    Attributes:
        step_id: 1-based position in the plan, used for depends_on.
        description: Human-readable summary of the step.
        subagent: Which sub-agent should execute the step.
        action: Free-form instructions for the sub-agent.
        expected_output: Shape description used by the verifier.
        depends_on: Step ids this step waits for.
        fallback_steps: Step ids to invoke if this step fails.
        max_retries: Per-step retry budget before giving up.
    """

    step_id: int
    description: str
    subagent: Subagent
    action: str
    expected_output: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    fallback_steps: list[int] = Field(default_factory=list)
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Plan templates per domain
# ---------------------------------------------------------------------------

def _research_papers_plan(intent: Intent) -> list[Step]:
    """Goal-oriented research-papers plan.

    Each step hands the sub-agent a single target and trusts it to
    figure out the clicks by looking at the page. URLs supplied by
    the user (``intent.params['url']``) override the default Google
    Scholar entry so requests like "use the WebVPN gateway at
    https://webvpn.swufe.edu.cn/..." actually work.
    """
    topic = intent.params.get("topic", "")
    count = int(intent.params.get("count", 3))
    must_be_published = bool(intent.params.get("must_be_published", True))
    download = bool(intent.params.get("download", True))
    summary = bool(intent.requires_summary)
    language = intent.params.get("language", "zh")
    user_url = (intent.params.get("url") or "").strip()

    # Step 1: open the entry page. When the user gave a URL we open
    # that; otherwise we default to Google Scholar.
    if user_url:
        entry_label = user_url
        entry_action = (
            f"browser_new_tab('{user_url}')。"
            "如果该 URL 已经在 tab 里则先 browser_navigate 切回。"
        )
    else:
        entry_label = "Google Scholar"
        entry_action = (
            "browser_new_tab('https://scholar.google.com')。"
            "如果已经打开了则先 browser_navigate 切回。"
        )

    plan: list[Step] = [
        Step(
            step_id=1,
            description=f"打开 {entry_label}",
            subagent="browser",
            action=entry_action,
            expected_output={"tab_opened": True, "url": "str"},
        ),
        Step(
            step_id=2,
            description=f"截图查看 {entry_label} 主页",
            subagent="browser",
            action=(
                "browser_screenshot(max_dim=1400),看清楚页面上有什么"
                "(搜索框?数据库列表?登录墙?),把截图路径记入 scratchpad。"
            ),
            expected_output={"screenshot_path": "str"},
            depends_on=[1],
        ),
        Step(
            step_id=3,
            description=f"在站内提交主题 {topic!r} 的搜索",
            subagent="browser",
            action=(
                f"看 step2 截图,自己判断怎么到达搜索框并提交 {topic!r}。"
                f"可能要先点'数据库/资源'再点'检索';"
                f"也可能直接是顶部搜索框。严格走浏览器(截图 + click_xy)。"
            ),
            expected_output={"search_submitted": True, "typed": "str"},
            depends_on=[2],
        ),
        Step(
            step_id=4,
            description=f"从搜索结果中挑 {count} 篇最相关的论文",
            subagent="browser",
            action=(
                f"截图看搜索结果,必要时滚动一次再截,"
                f"挑出和 {topic!r} 最相关的 {count} 篇,记标题和详情页位置。"
            ),
            expected_output={"candidate_papers": f"list[length={count}]"},
            depends_on=[3],
        ),
    ]
    next_id = 5

    if download:
        plan.append(Step(
            step_id=next_id,
            description=f"逐个进入 {count} 个详情页并下载 PDF",
            subagent="browser",
            action=(
                f"对 step4 选出的每篇:点标题进详情页,看到 PDF 链接就点,"
                f"落进 PDF viewer 就点工具栏下载按钮,"
                f"实在不行 printToPDF 兜底,把下载到的本地路径记入 scratchpad。"
                f"遇到登录墙就跳过该篇。"
            ),
            expected_output={"downloaded": f"list[length={count}]", "failed": "list[dict]"},
            depends_on=[4],
        ))
        next_id += 1
    else:
        plan.append(Step(
            step_id=next_id,
            description=f"逐个进入 {count} 个论文详情页(不下载,停在详情页)",
            subagent="browser",
            action=(
                f"对 step4 选出的每篇:点标题进详情页/摘要页,截图保存,"
                f"停在详情页即可,不要点 PDF 下载按钮,不要调用任何下载工具。"
                f"把每篇的标题/作者/期刊/详情页 URL 记入 scratchpad。"
                f"遇到登录墙就跳过该篇。"
            ),
            expected_output={
                "papers": f"list[length={count}]",
                "detail_shots": "list[str]",
                "failed": "list[dict]",
            },
            depends_on=[4],
        ))
        next_id += 1

    if summary:
        plan.append(Step(
            step_id=next_id,
            description="读 PDF 文本生成中文摘要",
            subagent="extractor",
            action=(
                f"对每个成功下载的 PDF,用 pdfplumber 抽取前 3 页文本,"
                f"为每篇生成 100-200 字的中文摘要,输出 language={language}。"
            ),
            expected_output={"summaries": f"list[length={count}]"},
            depends_on=[next_id - 1] if download else [4],
        ))
        next_id += 1

    plan.append(Step(
        step_id=next_id,
        description=(
            f"校验:{count} 篇都有标题/作者/详情页(及可选文件路径/摘要)"
        ),
        subagent="verifier",
        action=(
            f"检查每篇 paper 都包含 title/authors/venue/detail_url;"
            f"如果 download=true 还要 pdf_path 且文件 size > 10KB;"
            f"如果 summary=true 还要摘要长度 >= 80 字(中文)。"
        ),
        expected_output={"all_ok": "bool", "issues": "list[str]"},
        depends_on=[s.step_id for s in plan if s.subagent != "verifier"],
    ))

    if not must_be_published:
        # Optional arXiv fallback. The agent can run it only if the
        # main entry yielded nothing; this is driven by inspecting
        # the prior scratchpad at run time.
        plan.append(Step(
            step_id=next_id + 1,
            description="fallback: arXiv 搜索(仅当主入口无结果)",
            subagent="browser",
            action=(
                "如果前面步骤没有下载到任何 PDF:"
                "  1. browser_new_tab('https://arxiv.org/search/?query=...');"
                "  2. browser_screenshot + browser_click_xy 点击前 3 个结果标题;"
                "  3. 进到 paper 页面后,browser_screenshot 找 PDF 按钮,"
                "browser_click_xy 点击 'PDF' 链接;"
                "  4. 浏览器内触发下载。"
            ),
            expected_output={"arxiv_downloaded": "list[str]"},
            depends_on=[s.step_id for s in plan if s.subagent != "verifier"],
        ))

    return plan


def _shopping_plan(intent: Intent) -> list[Step]:
    item = intent.params.get("item", "")
    site = intent.params.get("site", "")
    return [
        Step(step_id=1, description=f"打开 {site or '目标网站'}",
             subagent="browser",
             action=f"browser_new_tab('https://{site}')" if site else "browser_new_tab()",
             expected_output={"tab": "opened"}),
        Step(step_id=2, description=f"搜索 {item!r}",
             subagent="browser",
             action=f"在搜索框中填入 {item!r} 并提交。",
             expected_output={"results_page_loaded": True},
             depends_on=[1]),
        Step(step_id=3, description="提取前 N 个商品",
             subagent="extractor",
             action="从结果页 DOM 中提取商品名/价格/链接。",
             expected_output={"items": "list[dict]"},
             depends_on=[2]),
        Step(step_id=4, description="校验结果",
             subagent="verifier",
             action="检查 items 非空且每个 item 都有 title/price。",
             expected_output={"all_ok": True},
             depends_on=[3]),
    ]


def _generic_plan(intent: Intent) -> list[Step]:
    return [
        Step(step_id=1, description="用浏览器打开目标",
             subagent="browser",
             action=intent.params.get("description", intent.domain),
             expected_output={"page_loaded": True}),
        Step(step_id=2, description="提取信息",
             subagent="extractor",
             action="从打开的页面中提取关键信息。",
             expected_output={"data": "dict"},
             depends_on=[1]),
        Step(step_id=3, description="校验",
             subagent="verifier",
             action="检查 data 完整可用。",
             expected_output={"all_ok": True},
             depends_on=[2]),
    ]


_TEMPLATES = {
    "research_papers": _research_papers_plan,
    "shopping": _shopping_plan,
    "form_filling": _generic_plan,
    "data_scraping": _generic_plan,
    "general_browser_task": _generic_plan,
    "unknown": _generic_plan,
}


def _requests_quote(s: str) -> str:
    """Tiny shim so the template string reads cleanly."""
    import urllib.parse
    return urllib.parse.quote_plus(s or "")


# ---------------------------------------------------------------------------
# LLM-based planner (optional, used when the template is missing/insufficient)
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """You are a task planner. Given a domain and params, return a JSON plan:
{"steps": [{"step_id": int, "description": str, "subagent": "api|browser|extractor|verifier",
            "action": str, "expected_output": dict, "depends_on": [int]}]}

Constraints:
- Step 1 must be either 'api' (for known API domains) or 'browser' (otherwise).
- The last step should usually be 'verifier' to check success.
- Use 'extractor' to convert raw page/API output into structured data.
- 'depends_on' lists prior step ids; most steps depend on step_id-1.

Respond with JSON only."""


class TaskPlanner:
    """Produces an ordered list of :class:`Step` for a given intent."""

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def plan(self, intent: Intent) -> list[Step]:
        """Return the plan for ``intent``.

        Order of operations:
        1. Look up a domain-specific template.
        2. If none, ask the LLM to produce a plan.
        3. If the LLM is unavailable, fall back to a generic 3-step
           plan (open -> extract -> verify).
        """
        template = _TEMPLATES.get(intent.domain)
        if template is not None:
            plan = template(intent)
        elif self._llm is not None:
            try:
                plan = self._llm_plan(intent)
            except Exception as exc:  # pragma: no cover
                logging.getLogger("agent.supervisor.error").info(
                    "PLAN-LLM fallback (reason=%s)", exc
                )
                plan = _generic_plan(intent)
        else:
            plan = _generic_plan(intent)

        log_event(
            "agent.supervisor.plan",
            f"PLAN: {len(plan)} steps",
            domain=intent.domain,
            subagents=",".join(s.subagent for s in plan),
        )
        return plan

    def _llm_plan(self, intent: Intent) -> list[Step]:
        llm = self._llm
        from langchain_core.messages import HumanMessage, SystemMessage
        raw = llm.invoke([
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "domain": intent.domain,
                "params": intent.params,
                "needs_browser": intent.needs_browser,
                "needs_api": intent.needs_api,
            }, ensure_ascii=False)),
        ])
        text = getattr(raw, "content", str(raw))
        blob = re.search(r"\{.*\}", text, re.DOTALL)
        if not blob:
            raise ValueError("planner LLM returned no JSON")
        data = json.loads(blob.group(0))
        return [Step(**{**s, "depends_on": s.get("depends_on", []) or []}) for s in data["steps"]]


__all__ = ["Step", "Subagent", "TaskPlanner"]
