"""Debug end-to-end with LangGraph streaming to see each node's effect."""
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
print("SEED tabs:", len(browser_list_tabs.invoke({})))

# Build supervisor and stream
from agents.supervisor import build_supervisor

app = build_supervisor()
compiled = app.compiled  # raw LangGraph StateGraph
print("--- streaming ---")
for chunk in compiled.stream(
    {"user_input": "仅保留当前活动窗口,删掉其余的窗口", "iteration_count": 0},
    config={"recursion_limit": 80},
):
    for node_name, state_after in chunk.items():
        if not isinstance(state_after, dict):
            continue
        sub_h = state_after.get("subagent_history") or []
        op_h = state_after.get("operation_history") or []
        d_h = state_after.get("direction_history") or []
        last_op = (state_after.get("subagent_history") or [{}])[-1]
        last_data = last_op.get("data") if isinstance(last_op, dict) else None
        data_keys = list(last_data.keys()) if isinstance(last_data, dict) else type(last_data).__name__
        print(
            f"  [{node_name}] sub_h={len(sub_h)} op_h={len(op_h)} dir_h={len(d_h)} "
            f"step_idx={state_after.get('current_step_idx')} "
            f"err={state_after.get('error')!r} "
            f"pending_op={state_after.get('pending_operation') and state_after['pending_operation'].get('assignee')} "
            f"last_subagent={last_op.get('subagent') if last_op else None} "
            f"last_status={last_op.get('status') if last_op else None} "
            f"last_keys={data_keys}"
        )

print("--- invoking ---")
try:
    final = app.invoke(
        {"user_input": "仅保留当前活动窗口,删掉其余的窗口", "iteration_count": 0},
        config={"recursion_limit": 80},
    )
    print("--- final keys:", list(final.keys()) if isinstance(final, dict) else type(final))
    print("--- subagent_history len:", len(final.get("subagent_history", [])))
    print("--- direction_history len:", len(final.get("direction_history", [])))
    print("--- operation_history len:", len(final.get("operation_history", [])))
    print("--- final_answer:", final.get("final_answer", "")[:200])
    print("--- err:", final.get("error"))
except Exception as e:
    import traceback
    traceback.print_exc()

print("\nAFTER tabs:", len(browser_list_tabs.invoke({})))
for t in browser_list_tabs.invoke({}):
    print("  ", t.get("title", "?")[:40], "|", t.get("url", "?")[:60])
