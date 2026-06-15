"""CLI entry point: research-papers workflow.

Usage::

    python -m scripts.run_research "企业风险预测" --count 3 --download --lang zh
    python -m scripts.run_research "transformer attention" --count 5 --no-download

The script imports the workflow from ``agents.workflows`` and
prints the supervisor's final state in a human-readable format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """Add the AI_web project root to sys.path for direct script execution."""
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # AI_web/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    harness_src = project_root / "browser-harness" / "src"
    if harness_src.exists() and str(harness_src) not in sys.path:
        sys.path.insert(0, str(harness_src))


_bootstrap_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent research-papers workflow.",
    )
    parser.add_argument("topic", help="Research topic, e.g. '企业风险预测'.")
    parser.add_argument("--count", type=int, default=3, help="How many papers to fetch (default 3).")
    parser.add_argument("--download", action="store_true", default=True,
                        help="Download PDFs locally (default on).")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--no-summary", dest="summary", action="store_false", default=True)
    parser.add_argument("--allow-preprints", dest="must_be_published", action="store_false", default=True,
                        help="Include arXiv preprints (default: published only).")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="Summary language.")
    parser.add_argument("--save-state", default="", help="Optional path to write full state JSON.")
    return parser.parse_args()


def render(state: dict) -> str:
    """Render the supervisor state as a human-readable summary."""
    history = state.get("subagent_history") or []
    final = state.get("final_answer") or "(no final answer)"
    err = state.get("error")
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"Topic: {state.get('topic', '')}")
    lines.append(f"User input: {state.get('user_input', '')}")
    lines.append(f"Steps executed: {len(history)}")
    if err:
        lines.append(f"Error: {err}")
    lines.append(f"Final: {final}")
    lines.append("-" * 70)
    for h in history:
        lines.append(
            f"  step {h.get('step_id'):>2} [{h.get('subagent'):>9}] "
            f"{h.get('status'):>6}  {h.get('elapsed_ms', 0):>5}ms"
        )
    lines.append("=" * 70)
    scratchpad = state.get("scratchpad") or {}
    if scratchpad:
        lines.append("Scratchpad keys: " + ", ".join(sorted(scratchpad.keys())))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    from agents.workflows.research_papers import run_research_papers_workflow

    state = run_research_papers_workflow(
        topic=args.topic,
        count=args.count,
        download=args.download,
        must_be_published=args.must_be_published,
        summary=args.summary,
        language=args.lang,
    )
    print(render(state))

    if args.save_state:
        out = Path(args.save_state).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(f"State saved to: {out}")

    return 0 if not state.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
