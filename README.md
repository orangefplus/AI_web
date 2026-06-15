# AI_web — Multi-agent Browser Orchestrator

A small multi-agent system that drives a **real Chrome window** through
the Chrome DevTools Protocol (CDP), making the browser behave like a
human reading and clicking through a website. The system plans a
sequence of steps, dispatches each step to a specialist sub-agent, and
recovers from flaky network / LLM calls automatically.

## What it can do

- **Strict-browser research workflow**: "Open this URL → click search
  box → type the keyword → click the first result → land on the paper's
  detail page" — no API shortcuts, no CSS-selector cheating, every
  step is a real `click_xy` on a pixel the agent just saw in a
  screenshot.
- **Plug-in portals**: the user can pass any entry URL (a WebVPN
  gateway, a library catalogue, a publisher landing page) and the
  planner will route the entire plan through that URL instead of
  defaulting to Google Scholar.
- **Modal / native dialog auto-dismiss**: campus WebVPN notices,
  Elsevier / IEEE / NIH GDC consent modals, JS `alert()` /
  `confirm()` popups are all detected and dismissed automatically.
- **Resilient to flaky LLM backends**: the LLM wrapper retries on
  502 / connection-refused / 60-s timeouts with exponential backoff
  (2 → 4 → 8 → 20 s, up to 4 attempts), so a momentary blip from
  the upstream one-api proxy no longer kills the run.
- **PDF download that respects the viewer UI**: when a paper opens
  in Chrome's built-in PDF viewer the agent first tries to click
  the toolbar's download icon, and only falls back to `printToPDF`
  if the click doesn't yield a file.
- **Per-step verbose logging** in the terminal (INFO level,
  ANSI-stripped) so the user can watch the flow without scrolling
  through raw LLM messages.

## Architecture

```
            user_input
                │
                ▼
      ┌────────────────────┐
      │  intent_router     │  Intent(domain, params, needs_browser, …)
      └────────────────────┘
                │
                ▼
      ┌────────────────────┐
      │  task_planner      │  List[Step]  (one per sub-agent call)
      └────────────────────┘
                │
        ┌───────┼───────┬───────────┬─────────────┐
        ▼       ▼       ▼           ▼             ▼
   api_agent  browser  extractor  verifier    (re-planner)
                       agent      agent
```

The supervisor is a small LangGraph `StateGraph` (see
[`agents/supervisor.py`](agents/supervisor.py)). The browser sub-agent
runs a ReAct agent over a curated tool surface: only
screenshot / click_xy / read-page-text / dismiss-overlay /
set-download-dir / download-PDF tools are exposed. The selector-based
tools (`browser_fill_input`, `browser_dispatch_key`,
`browser_wait_for_element`, `browser_http_get`, `browser_upload_file`)
are intentionally hidden so the LLM cannot bypass the "real browsing"
loop the user asked for.

## Repository layout

```
AI_web/
├── agents/
│   ├── intent_router.py     # Pydantic Intent + regex param extraction
│   ├── task_planner.py      # Domain-specific Step templates
│   ├── supervisor.py        # LangGraph orchestrator + LLM retry
│   ├── prompts.py           # Shared system-prompt rules
│   ├── subagents/
│   │   ├── base.py          # Subagent abstract base class
│   │   ├── browser_agent.py # ReAct agent over the curated tool set
│   │   ├── api_agent.py
│   │   ├── extractor_agent.py
│   │   └── verifier_agent.py
│   └── workflows/           # (legacy) alternate plan shapes
├── tools/
│   ├── browser.py           # CDP-backed browser tools (screenshot,
│   │                        # click_xy, dismiss_overlay, read_page_text, …)
│   ├── download.py          # printToPDF / set_download_dir helpers
│   ├── _logging.py          # ANSI-stripped, key-event-only logger
│   ├── _bootstrap.py        # browser-harness daemon launcher
│   ├── _tooling.py          # LangChain tool decorator shims
│   ├── admin.py / decorators.py / session.py
├── config/
│   └── config.py            # reads XF_API_KEY from env (.env)
├── downloads_test/
│   ├── start_agent_chrome.py   # spawns an isolated Chrome (port 9333)
│   ├── test_run_e2e.py         # end-to-end plan run
│   ├── test_browser_plan.py    # browser-only plan runner
│   ├── test_llm_only.py        # smoke-test the LLM wrapper
│   ├── test_daemon_direct.py   # CDP daemon round-trip
│   └── test_probe_only.py      # port probe
├── scripts/
│   └── run_research.py
├── .env.example             # copy to .env and fill in your key
├── .gitignore               # excludes __pycache__, .agent_chrome_profile, …
└── README.md
```

## Getting started

```powershell
# 1. install the browser-harness sidecar (editable install)
cd AI_web
pip install -e browser-harness/

# 2. set up the secret
cp .env.example .env
# edit .env and put your real XF_API_KEY in there

# 3. spawn a private Chrome (port 9333, isolated profile)
python downloads_test\start_agent_chrome.py

# 4. run the end-to-end plan
$env:BU_CDP_URL="http://127.0.0.1:9333"
$env:PYTHONPATH = "$(Get-Location);$(Get-Location)\browser-harness\src"
python downloads_test\test_run_e2e.py
```

## How the user actually drives it

The plan is shaped by `user_input`. The intent router extracts
keywords / counts / "do not download" flags / URLs out of the
sentence, and the planner expands that into steps.

Examples:

| User input                                                                                              | Result                                                                                      |
|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `帮我找 2 篇关于"企业风险预测"的期刊论文`                                                              | Google Scholar → top 2 → click into detail page → stop.                                    |
| `…关键词 "enterprise risk prediction"…不要下载,停在详情页`                                              | Same plan but `download=false`, and the last step is "stop on the detail page".            |
| `请打开 https://webvpn.swufe.edu.cn/... 用里面能跳到的数据库找论文,不要下载,停在详情页`                  | The whole plan operates **inside the WebVPN gateway**, with the agent picking a database.   |

## Logging

`tools/_logging.py` installs a single root handler that strips ANSI
escape codes, so logs render as plain text on Windows GBK consoles
even if a third-party library (colorlog, rich, click) tries to colour
its output. Only the key events are logged:

```
[agent.supervisor.intent]  INTENT research_papers (conf=0.40) | params={...}
[agent.supervisor.plan]    PLAN: 6 steps | domain=research_papers subagents=...
[agent.supervisor.step]    STEP 1 [browser]: 打开 https://...
[agent.supervisor.result]  STEP 1 OK in 7625ms | subagent=browser
[agent.supervisor.error]   STEP 4 ERR: <reason>
[agent.supervisor.result]  DONE | steps=6
```

## License

Personal / educational use.
