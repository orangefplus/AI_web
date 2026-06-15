"""Minimal connection probe — does NOT touch the user's tabs.

Goals:
  1. Spin up a browser-harness daemon under a unique name (no conflict).
  2. Send ONE harmless CDP call (Target.getTargets) to confirm reachability.
  3. Print exactly what's reachable and stop. No navigation, no clicking.

The user is expected to have Chrome running with remote debugging enabled
(their existing setup) and to click "Allow" on the Chrome 144+ popup
the first time the harness attaches.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web")
HARNESS_SRC = PROJECT_ROOT / "browser-harness" / "src"

# Isolate this run from any other browser-harness daemon by forcing a
# unique name *and* a unique temp dir. Nothing else in the system uses
# this name, so we can be sure we don't fight the user's setup.
UNIQUE_NAME = "aiweb_probe"
TMP_DIR = PROJECT_ROOT / "downloads_test" / "_probe_runtime"
TMP_DIR.mkdir(parents=True, exist_ok=True)

for p in (str(PROJECT_ROOT), str(HARNESS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["BU_NAME"] = UNIQUE_NAME
os.environ["BH_TMP_DIR"] = str(TMP_DIR)
os.environ["BH_RUNTIME_DIR"] = str(TMP_DIR)


def main() -> int:
    from tools._logging import setup_logging
    setup_logging(level="INFO", noisy_level="WARNING")

    print(f"[PROBE] using BU_NAME={UNIQUE_NAME}")
    print(f"[PROBE] runtime dir: {TMP_DIR}")

    # Bring up a daemon for our isolated name. We pass `name` so the
    # bootstrap does NOT clobber the global "default" daemon.
    from tools._bootstrap import setup_browser
    print("[PROBE] starting isolated browser-harness daemon...")
    print("[PROBE] if Chrome 144+ is showing the 'Allow remote debugging?'")
    print("[PROBE] popup, please click Allow within 60 seconds.")
    try:
        setup_browser(name=UNIQUE_NAME, wait=60.0)
        print("[PROBE] daemon ready")
    except Exception as exc:
        print(f"[PROBE] daemon start FAILED: {type(exc).__name__}: {exc}")
        log_path = TMP_DIR / "bu.log"
        if log_path.exists():
            print(f"[PROBE] daemon log: {log_path.read_text()!r}")
        return 2

    # Single harmless CDP call. We just want to know if the daemon
    # can talk to Chrome. If the user hasn't clicked Allow, this will
    # raise a websocket error and we exit cleanly without killing anything.
    from browser_harness import helpers
    print("[PROBE] sending Target.getTargets via CDP...")
    try:
        tabs = helpers.list_tabs(include_chrome=False)
    except Exception as exc:
        print(f"[PROBE] CDP call FAILED: {type(exc).__name__}: {exc}")
        print("[PROBE] this is harmless — daemon is still up, but Chrome")
        print("         has not yet granted remote-debugging permission.")
        print("[PROBE] click 'Allow' on the Chrome popup, then re-run this script.")
        return 3

    print(f"[PROBE] OK — Chrome is reachable, {len(tabs)} non-internal tab(s):")
    for t in tabs:
        print(f"         - {t.get('title', '<no title>')[:60]!r}  url={t.get('url', '')[:80]}")

    print("\n[PROBE] success — connection works. No tabs were opened or closed.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)
