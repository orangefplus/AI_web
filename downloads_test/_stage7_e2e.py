"""End-to-end test: drive supervisor.run() on the demo task."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from tools._bootstrap import setup_browser
from tools import browser_list_tabs, browser_new_tab

setup_browser(wait=30.0)

print("BEFORE tabs:", len(browser_list_tabs.invoke({})))
browser_new_tab.invoke({"url": "https://www.baidu.com"})
browser_new_tab.invoke({"url": "https://www.zhihu.com"})
browser_new_tab.invoke({"url": "https://github.com"})
print("SEED tabs:", len(browser_list_tabs.invoke({})))
for t in browser_list_tabs.invoke({}):
    print("  ", t.get("title", "?")[:40], "|", t.get("url", "?")[:60])

print("\n=== supervisor.run('仅保留当前活动窗口,删掉其余的窗口') ===")
from agents.agent import run_demo

result = run_demo("仅保留当前活动窗口,删掉其余的窗口")

print("--- final_answer:", result.get("final_answer", ""))
print("--- refined_goal:", (result.get("refined") or {}).get("refined_goal", ""))
print("--- subagent_history:")
for h in result.get("subagent_history", []):
    data = h.get("data")
    if isinstance(data, dict):
        keys = list(data.keys())
    else:
        keys = type(data).__name__
    print(
        f"    step#{h.get('step_id')} subagent={h.get('subagent')} "
        f"status={h.get('status')} elapsed_ms={h.get('elapsed_ms')} data_keys={keys}"
    )
print("--- direction_history:")
for d in result.get("direction_history", []):
    print(
        f"    verdict={d.get('verdict')} pct={d.get('progress_pct')} "
        f"reason={d.get('reason', '')[:60]}"
    )
print("--- operation_history:")
for o in result.get("operation_history", []):
    print(f"    assignee={o.get('assignee')} rationale={o.get('rationale', '')[:60]}")

print("AFTER tabs:", len(browser_list_tabs.invoke({})))
for t in browser_list_tabs.invoke({}):
    print("  ", t.get("title", "?")[:40], "|", t.get("url", "?")[:60])
