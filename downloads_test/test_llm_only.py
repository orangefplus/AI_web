"""Isolated LLM connectivity test."""
import sys
import traceback

sys.path.insert(0, r"c:\Users\wangxin\Documents\trae_projects\Rag_heima\AI_web\browser-harness\src")

from tools._logging import setup_logging
setup_logging(level="INFO", noisy_level="WARNING")

print("=== Test A: bare LLM call (no tools) ===")
try:
    from agents.supervisor import default_llm
    llm = default_llm()
    from langchain_core.messages import HumanMessage
    out = llm.invoke([HumanMessage(content="用一句话说你好")])
    print("OK LLM responded:", repr(out.content[:200]))
except Exception as exc:
    print("LLM ERR:", type(exc).__name__, exc)
    traceback.print_exc()
