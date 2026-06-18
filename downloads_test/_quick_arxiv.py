"""Quick e2e test for the browse_summary domain.

Loads env from .env.example, runs the supervisor on a single arxiv
user input, and prints a tight summary of what happened. We
explicitly set stdout to UTF-8 so the GBK Windows console can show
Chinese without mojibake.
"""
import atexit
import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr BEFORE any import that might print.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)


def p(*a, **kw):
    print(*a, **kw, flush=True)

ROOT = Path(r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "browser-harness" / "src"))

p("[BOOT] paths configured, ROOT=", str(ROOT))

from tools._logging import setup_logging
p("[BOOT] importing setup_logging")
setup_logging(level="INFO", noisy_level="WARNING")
p("[BOOT] logging ready")

from agents.supervisor import build_supervisor
p("[BOOT] supervisor loaded")

app = build_supervisor()
p("[BOOT] supervisor built")

user_input = (
    "请打开 https://arxiv.org/ 严格走浏览器,看完首页后总结一下首页上展示了什么内容"
    "(板块、推荐论文、热门类别等),最后停留在 arxiv 首页,不要下载任何东西。"
)
p("[RUN]", user_input)
final = app.invoke(
    {"user_input": user_input, "iteration_count": 0},
    config={"recursion_limit": 60},
)

p()
p("=" * 70)
p("INTENT  :", json.dumps(final.get("intent"), ensure_ascii=False))
plan = final.get("plan") or []
p(f"PLAN    : {len(plan)} steps")
for s in plan:
    p(f"  step {s.get('step_id'):>2} [{s.get('subagent'):>9}] {s.get('description')}")
hist = final.get("subagent_history") or []
p(f"HISTORY : {len(hist)} entries")
for h in hist:
    p(
        f"  step {h.get('step_id'):>2} [{h.get('subagent'):>9}] "
        f"status={h.get('status'):>6} elapsed={h.get('elapsed_ms', 0)}ms"
    )
sp = final.get("scratchpad") or {}
p("SCRATCH :")
for k, v in sp.items():
    if k == "last_result":
        continue
    if isinstance(v, str):
        snippet = v[:300].replace("\n", " ")
        p(f"  {k} (len={len(v)}): {snippet}")
    else:
        p(f"  {k}: {v!r}")
p("FINAL   :", (final.get("final_answer") or "")[:2000])
p("ERROR   :", final.get("error") or "(none)")
