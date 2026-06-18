"""Directly extract arxiv paper titles from the search results page.

Skips the slow LLM loop and uses a single browser_run_js to get
(title, abs_url, authors) for the 50 results.  The LLM is then
asked to pick the 5 most authoritative/relevant ones in a single
call.
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

from tools import browser_navigate, browser_run_js, setup_browser  # noqa: E402

EXTRACT_JS = r"""
(() => {
  const out = [];
  document.querySelectorAll('li.arxiv-result').forEach((li, i) => {
    const titleEl = li.querySelector('p.title');
    const absLink = li.querySelector('a[href*="/abs/"]');
    const authorsEl = li.querySelector('.authors');
    if (titleEl && absLink) {
      out.push({
        i,
        title: (titleEl.innerText || titleEl.textContent).replace(/\n/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 250),
        abs_url: absLink.href,
        authors: authorsEl ? (authorsEl.innerText || '').replace(/Authors:\s*/, '').trim().slice(0, 250) : ''
      });
    }
  });
  return out;
})()
"""


def main() -> int:
    setup_browser(wait=10.0)

    print("[init] navigate to arxiv search...")
    browser_navigate.invoke({"url": "https://arxiv.org/search/?searchtype=all&query=enterprise+risk+prediction&start=0"})
    time.sleep(2.5)

    print("[init] extract paper list...")
    r = browser_run_js.invoke({"expression": EXTRACT_JS})
    if isinstance(r, dict) and "result" in r:
        data = r["result"] or []
    else:
        data = r if isinstance(r, list) else []
    print(f"[init] found {len(data)} papers\n")

    if not data:
        print("no papers found")
        return 1

    print("=" * 80)
    print("ALL PAPERS (raw)")
    print("=" * 80)
    for d in data:
        print(f"{d['i']+1:2d}. {d['title']}")
        print(f"    {d['abs_url']}")
        print(f"    authors: {d['authors'][:120]}")
        print()

    # Save full data
    out = ROOT / "downloads_test" / "_stage11_papers.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[init] saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
