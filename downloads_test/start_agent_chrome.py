"""Start a dedicated Chrome instance for the AI agent (Way 2).

Why Way 2:
  * Does not touch the user's everyday Chrome profile.
  * No Chrome 144+ "Allow remote debugging?" popup.
  * Persistent cookies: a dedicated user-data-dir keeps the agent
    logged in across runs, so the user only logs in once.

Usage:
  python start_agent_chrome.py

After it opens, navigate to whatever you want the agent to access
(Scholar / arXiv / your publisher SSO) and log in. The agent will
then use this Chrome via BU_CDP_URL=http://127.0.0.1:9333.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web")
PROFILE_DIR = PROJECT_ROOT / ".agent_chrome_profile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REMOTE_DEBUGGING_PORT = 9333
START_URL = "https://scholar.google.com"

# Sanity: must not be the platform default profile.
default_win = Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data"
assert PROFILE_DIR.resolve() != default_win.resolve(), (
    f"profile dir {PROFILE_DIR} would collide with the default Chrome profile"
)

print(f"[agent-chrome] profile dir   : {PROFILE_DIR}")
print(f"[agent-chrome] debug port    : {REMOTE_DEBUGGING_PORT}")
print(f"[agent-chrome] start URL     : {START_URL}")
print(f"[agent-chrome] launching Chrome...")

flags = [
    f"--user-data-dir={PROFILE_DIR}",
    f"--remote-debugging-port={REMOTE_DEBUGGING_PORT}",
    "--no-first-run",
    "--no-default-browser-check",
    START_URL,
]
proc = subprocess.Popen([CHROME, *flags])
print(f"[agent-chrome] PID: {proc.pid}")
print()
print(f"[agent-chrome] Set this env var before running the agent:")
print(f"  set BU_CDP_URL=http://127.0.0.1:{REMOTE_DEBUGGING_PORT}")
print()
print(f"[agent-chrome] To stop: Stop-Process -Id {proc.pid}")
print(f"[agent-chrome] Cookie / session data persists at:")
print(f"  {PROFILE_DIR}")
