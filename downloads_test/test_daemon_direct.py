"""Try to start the daemon with the isolated name and see why it failed."""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web")
HARNESS_SRC = PROJECT_ROOT / "browser-harness" / "src"
TMP_DIR = PROJECT_ROOT / "downloads_test" / "_probe_runtime"
TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["BU_NAME"] = "aiweb_probe"
os.environ["BH_TMP_DIR"] = str(TMP_DIR)
os.environ["BH_RUNTIME_DIR"] = str(TMP_DIR)

for p in (str(PROJECT_ROOT), str(HARNESS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

print("BU_NAME =", os.environ["BU_NAME"])
print("BH_TMP_DIR =", os.environ["BH_TMP_DIR"])

# Start the daemon directly
import browser_harness.daemon as daemon_mod
print("daemon.NAME =", daemon_mod.NAME)
print("daemon.LOG =", daemon_mod.LOG)
print("PID file =", daemon_mod.PID)
print("SOCK =", daemon_mod.SOCK)

# Launch the daemon subprocess
import subprocess
env = dict(os.environ)
print()
print("Launching daemon subprocess...")
p = subprocess.Popen(
    [sys.executable, "-m", "browser_harness.daemon"],
    env=env,
    cwd=str(HARNESS_SRC.parent),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(f"PID: {p.pid}")

# Wait up to 20 seconds for it to settle
for i in range(20):
    time.sleep(1)
    if p.poll() is not None:
        out = p.stdout.read().decode("utf-8", errors="replace")
        print(f"Daemon exited with code {p.returncode} after {i+1}s")
        print("--- stdout/stderr ---")
        print(out)
        break
    print(f"  [{i+1}s] still running...")
else:
    print("Daemon is still running after 20s")

# Read the log
log_path = Path(daemon_mod.LOG)
if log_path.exists():
    print("\n--- daemon log ---")
    print(log_path.read_text())
else:
    print(f"\n(no log file at {log_path})")
