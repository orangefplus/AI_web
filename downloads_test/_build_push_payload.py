"""Build the JSON payload for MCP push_files from the latest git commit.

Reads every file touched in HEAD (excluding the deleted one) and
packs them into a single JSON file ``push_payload.json`` next to
this script.  The payload is ready to be passed verbatim to the
GitHub MCP server's push_files tool.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-C", str(ROOT)]


def list_commit_files() -> list[str]:
    """Return all paths in HEAD's commit (new + modified + deleted)."""
    out = subprocess.check_output(
        GIT + ["show", "--name-only", "--pretty=format:", "HEAD"],
        encoding="utf-8",
    )
    return [p.strip() for p in out.splitlines() if p.strip()]


def main() -> int:
    files_in_commit = list_commit_files()
    deleted: list[str] = []
    to_push: list[dict] = []
    for rel in files_in_commit:
        full = ROOT / rel
        if not full.exists():
            # deleted file -> cannot push content, will note in summary
            deleted.append(rel)
            continue
        content = full.read_text(encoding="utf-8", errors="replace")
        to_push.append({"path": rel, "content": content})

    payload = {
        "owner": "orangefplus",
        "repo": "AI_web",
        "branch": "main",
        "files": to_push,
        "message": (
            "feat: 4-layer multi-agent refactor with ReAct mode + error diagnosis\n\n"
            "See local commit 5ffe0ff for the full diff.  This MCP push includes "
            "all new and modified files in that commit.  The deleted file "
            "agents/subagents/extractor_agent.py is not removed by this tool "
            "and will remain on the remote (consider a follow-up cleanup commit)."
        ),
    }

    out_path = Path(__file__).resolve().parent / "push_payload.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}")
    print(f"  files to push:    {len(to_push)}")
    print(f"  files deleted:    {deleted}")
    print(f"  payload size:     {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
