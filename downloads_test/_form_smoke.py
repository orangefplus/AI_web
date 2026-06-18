"""Smoke test: drive the arxiv advanced search form with the new tools.

The Chrome tab is already at https://arxiv.org/search/advanced (per
the screenshot).  We:
  1. list_form_fields()           — see what the page exposes
  2. find_field_by_label('Title') — resolve the Title <select>
  3. select_option(...)            — keep Title (or change to Abstract)
  4. fill_input(...'search')      — type the search keyword
  5. find_field_by_label('Quantitative Finance') — resolve the q-fin checkbox
  6. check_checkbox(...)          — tick q-fin only
  7. submit_form('form')          — submit
"""
import sys, os, json
from pathlib import Path

# Force UTF-8 stdout/stderr so Windows PowerShell doesn't garble the
# output via its default GBK pipeline.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# Tee to a UTF-8 file ourselves so we have a clean log even when
# PowerShell `Out-File` mangles the early lines.
_LOG = PROJECT / "downloads_test" / "_form_smoke.log"
_LOG_FH = open(_LOG, "w", encoding="utf-8")
def _log(msg=""):
    line = str(msg)
    sys.stdout.write(line + "\n")
    _LOG_FH.write(line + "\n")
    _LOG_FH.flush()
_log.__name__ = "print"

from tools import (
    setup_browser,
    browser_list_form_fields,
    browser_find_field_by_label,
    browser_select_option,
    browser_fill_input,
    browser_check_checkbox,
    browser_submit_form,
    browser_get_page_info,
)

# The @tool-decorated functions are exposed as LangChain StructuredTool
# objects (which is what the agent gets).  StructuredTool is *not*
# directly callable — you must use `.invoke(arg_dict)`.  Define thin
# shims so the test reads naturally.
def _call(tool, **kwargs):
    return tool.invoke(kwargs)


setup_browser()

_log("=" * 70)
_log("STEP 1: page info (sanity)")
_log("=" * 70)
try:
    info = _call(browser_get_page_info)
    _log(f"title: {info.get('title')}")
    _log(f"url:   {info.get('url')}")
except Exception as e:
    _log(f"  page info failed: {e!r}")
    info = {}

_log()
_log("=" * 70)
_log("STEP 2: list_form_fields  (read-only — should be safe)")
_log("=" * 70)
fields = _call(browser_list_form_fields)
_log(f"  raw len={len(fields)} repr(first 200)={fields[:200]!r}")
if not fields:
    _log("  ERROR: empty response from list_form_fields")
    sys.exit(1)
parsed = json.loads(fields)
_log(f"  -> {len(parsed)} visible form fields")
for f in parsed[:25]:
    _log(
        f"  - <{f.get('tag', '?').lower()} type={f.get('type', '?'):>10}> "
        f"id={f.get('id', '')[:20]:>20} name={f.get('name', '')[:18]:>18} "
        f"label={f.get('label', '')[:50]!r}"
    )

_log()
_log("=" * 70)
_log("STEP 3: find_field_by_label('Title')")
_log("=" * 70)
out = _call(browser_find_field_by_label, label_text="Title")
_log(f"  {out}")

_log()
_log("=" * 70)
_log("STEP 4: find_field_by_label('Quantitative Finance')")
_log("=" * 70)
out = _call(browser_find_field_by_label, label_text="Quantitative Finance")
_log(f"  {out}")
qfin = json.loads(out)
qfin_id = qfin.get("id", "")
_log(f"  resolved id: {qfin_id!r}")

_log()
_log("=" * 70)
_log("STEP 5: select 'Title' in the terms-0-field SELECT (this label IS 'Field to search')")
_log("=" * 70)
# The "Title" string the user sees is the *option text*, not the
# <label> for the select.  The <label> reads "Field to search".  To
# keep the test self-contained we just use the field's id directly.
title_id = "terms-0-field"
_log(f"  select id={title_id!r}")
out = _call(browser_select_option, selector=f"#{title_id}", text="Title")
_log(f"  {out}")

_log()
_log("=" * 70)
_log("STEP 6: fill the search box with 'Corporate Risk Prediction'")
_log("=" * 70)
# The top-right arxiv search box has name="query" (no id).  We also
# have a deeper 'terms-0-term' box on the advanced form.  Test both.
out = _call(browser_fill_input, selector="input[name=query]", text="Corporate Risk Prediction")
_log(f"  {out}")
out = _call(browser_fill_input, selector="#terms-0-term", text="Corporate Risk Prediction")
_log(f"  {out}")

_log()
_log("=" * 70)
_log("STEP 7: check 'Quantitative Finance' subject")
_log("=" * 70)
out = _call(browser_check_checkbox, selector=f'#{qfin_id}', value=True)
_log(f"  {out}")

_log()
_log("=" * 70)
_log("DONE (form is filled; not actually clicking Search to avoid losing state)")
_log("=" * 70)
