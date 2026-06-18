"""Smoke-test the error diagnoser against the real failure shapes we
have seen in the test runs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents._error_diagnosis import diagnose, short_label

cases = [
    "Invalid parameters, Failed to deserialize params.targetId - BINDINGS: string value expected",
    "AttributeError: 'NoneType' object has no attribute 'getAttribute'",
    "JavaScript evaluation failed: TypeError: Cannot read properties of null",
    "Browser was disconnected.",
    "TimeoutError: Page load took too long",
    "net::ERR_NAME_NOT_RESOLVED",
    "Login required. Please login first.",
    "滑动验证 - please slide to verify",
    "429 Too Many Requests",
    "404 Not Found",
    "Cannot read properties of undefined (reading 'click')",
    "element click intercepted: Element is not clickable at point (200, 300)",
    "Some random text that does not match anything",
]

for c in cases:
    d = diagnose(c)
    print(f"{short_label(d.category):22s}  conf={d.confidence:.2f}  {d.short}")
    print(f"    in : {c[:80]}")
    print(f"    out: {d.recovery_hint[:90]}")
    print()
