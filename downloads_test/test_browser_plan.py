"""End-to-end smoke test for the new browser-driven plan."""
import sys, json
sys.path.insert(0, r'c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web')
sys.path.insert(0, r'c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web\browser-harness\src')

from tools._logging import setup_logging
setup_logging()

print('=== Test 1: Import new modules ===')
from tools import browser_download_pdf, browser_set_download_dir
print('[OK] tools.download')
from agents.intent_router import IntentRouter, Intent
from agents.task_planner import TaskPlanner, Step
print('[OK] intent + planner')
from agents.subagents.browser_agent import BrowserAgent, BROWSER_AGENT_PROMPT
print('[OK] BrowserAgent + BROWSER_AGENT_PROMPT (length:', len(BROWSER_AGENT_PROMPT), 'chars)')
from agents.subagents import build_subagents
print('[OK] subagents builder')

print()
print('=== Test 2: Plan for "find 3 published papers about enterprise risk prediction" ===')
router = IntentRouter(llm=None)
intent = router.classify('帮我找 3 篇关于企业风险预测的期刊论文并下载 PDF，每篇总结')
print(f'Intent: {intent.short()}')
print(f'Params: {intent.params}')

planner = TaskPlanner(llm=None)
plan = planner.plan(intent)
print(f'Plan has {len(plan)} steps:')
for s in plan:
    print(f'  step {s.step_id} [{s.subagent:>9}]: {s.description}')

# Verify the plan is all browser-driven (no api steps)
api_steps = [s for s in plan if s.subagent == 'api']
browser_steps = [s for s in plan if s.subagent == 'browser']
print(f'\nAPI steps: {len(api_steps)} (should be 0)')
print(f'Browser steps: {len(browser_steps)} (should be most of them)')
assert len(api_steps) == 0, 'Plan should be browser-driven'
print('PASS')

print()
print('=== Test 3: Verify download tool has no urllib usage ===')
import inspect
from tools import download as download_mod
src = inspect.getsource(download_mod)
has_urllib = 'urllib.request' in src
has_print_to_pdf = 'printToPDF' in src
print(f'Contains urllib.request: {has_urllib} (should be False)')
print(f'Contains printToPDF: {has_print_to_pdf} (should be True)')
assert not has_urllib, 'Should not have urllib'
assert has_print_to_pdf, 'Should have printToPDF fallback'
print('PASS')

print()
print('=== Test 4: Build subagents list with download tools ===')
# We need an LLM object; use a minimal stand-in
class FakeLLM:
    def with_structured_output(self, *a, **kw): return self
    def invoke(self, *a, **kw): return type('M', (), {'content': '{}'})()

subagents = build_subagents(FakeLLM())
browser_agent = subagents['browser']
print(f'Browser agent tool count: {len(browser_agent.tools)}')
tool_names = [getattr(t, "name", "?") for t in browser_agent.tools]
print(f'Browser tool names: {tool_names}')
has_dl = 'browser_download_pdf' in tool_names
has_set = 'browser_set_download_dir' in tool_names
print(f'Has browser_download_pdf: {has_dl}')
print(f'Has browser_set_download_dir: {has_set}')
assert has_dl and has_set
print('PASS')

print()
print('=== ALL TESTS PASSED ===')
