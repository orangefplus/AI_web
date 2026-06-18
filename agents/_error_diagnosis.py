"""Browser error classification.

When a browser tool call fails, the raw exception message is often
ambiguous (``"TypeError: Cannot read properties of null"``) and the
LLM is bad at guessing the recovery action. This module maps the
common failure modes to a small set of (category, recovery_hint)
tuples that the supervisor and the ReAct Master can act on.

The classifier is **regex-based** rather than LLM-based on purpose:
it has to be deterministic, fast, and cheap. The categories were
chosen after reading several real failure logs from the test runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Categories. Stable identifiers the ReAct Master can branch on.
CAT_OK = "ok"
CAT_TIMEOUT = "timeout"
CAT_NAVIGATION = "navigation"
CAT_NAVIGATION_PROXY = "navigation_proxy"
CAT_SELECTOR_NULL = "selector_null"
CAT_SELECTOR_TYPE_ERROR = "selector_type_error"
CAT_CLICK_MISS = "click_miss"
CAT_LOGIN_WALL = "login_wall"
CAT_CAPTCHA = "captcha"
CAT_OVERLAY = "overlay"
CAT_EXTRACT_EMPTY = "extract_empty"
CAT_RUNJS_INVALID_PARAMS = "runjs_invalid_params"
CAT_FILE_DOWNLOAD = "file_download"
CAT_RATE_LIMIT = "rate_limit"
CAT_NETWORK = "network"
CAT_UNEXPECTED = "unexpected"


@dataclass(frozen=True)
class ErrorDiagnosis:
    """Structured representation of what went wrong + how to fix it."""

    category: str
    short: str                # human-readable one-liner (Chinese)
    detail: str               # longer explanation for the LLM
    recovery_hint: str        # concrete next action the LLM should take
    can_retry: bool = True
    confidence: float = 1.0   # 0..1, used by the supervisor for triage


# Pattern -> (category, confidence, short, detail, recovery_hint, can_retry)
# Order matters: more specific patterns first.
_PATTERNS: list[tuple[re.Pattern, str, float, str, str, str, bool]] = [
    # 1. CDP-level "Invalid parameters" usually means a type mismatch
    #    in the targetId (e.g. LLM passed a dict or number when CDP
    #    expected a string).  NOT a real network/permission issue.
    (
        re.compile(r"Invalid parameters|deserialize params\.targetId", re.I),
        CAT_RUNJS_INVALID_PARAMS, 0.95,
        "CDP 参数类型错误",
        "调用 Chrome DevTools Protocol 时参数类型不对(常见 targetId 不是字符串)。"
        "原因可能是 LLM 把页面元数据字段误当成 targetId。",
        "不要用 browser_run_js 配 targetId 参数;改用 browser_extract_links 拿到"
        "真实可点击的链接表,再用 browser_navigate_to_link 跳转,或重新 navigate。",
        True,
    ),
    # 2. Captcha / 验证 / 安全验证
    (
        re.compile(r"captcha|验证|滑动|滑块|security[\s_]?check|verify[\s_]?human", re.I),
        CAT_CAPTCHA, 0.95,
        "遇到人机验证",
        "页面出现验证码/滑块/安全检查,非自动化可解。",
        "立即停止动作,记下当前状态,调用 browser_screenshot 保存证据,"
        "用 ask_user 模式向用户报告『需要人工验证』,并附截图。",
        False,
    ),
    # 3. Login wall
    (
        re.compile(r"login|登录|登[陆陆]|请[先登]登录|扫码|需要登录", re.I),
        CAT_LOGIN_WALL, 0.9,
        "遇到登录墙",
        "页面被登录拦截(可能 SSO/CAPTCHA/学校 VPN),当前会话没有凭证。",
        "停止自动化,ask_user 请用户登录或提供 cookie;"
        "或换一个不需要登录的来源(arxiv 公开页、谷歌学术镜像)。",
        False,
    ),
    # 4. Overlay / 弹窗
    (
        re.compile(r"overlay|modal|dialog|弹窗|遮罩|consent|cookie|同意", re.I),
        CAT_OVERLAY, 0.85,
        "页面有弹窗/遮罩",
        "页面上有弹窗/cookie 同意/模态框遮挡了目标元素。",
        "派 click 智能体调用 browser_dismiss_overlay 或点击弹窗的关闭/同意按钮,"
        "之后再继续原任务。",
        True,
    ),
    # 5. Selector null  - getElementById / querySelector 返回 null
    (
        re.compile(r"Cannot read propert.*of (null|undefined)|is null|is undefined|NoneType.*has no attribute", re.I),
        CAT_SELECTOR_NULL, 0.95,
        "DOM 元素未找到",
        "选择器(JS 表达式)返回 null,目标元素不在页面/被动态加载/选择器拼写错。",
        "先 browser_screenshot 看一下页面当前真实状态,再用 browser_extract_links"
        "枚举所有可见链接拿到真实结构,**不要再凭印象写选择器**。",
        True,
    ),
    # 5. JS TypeError: split / forEach / .href 之类
    (
        re.compile(r"TypeError:|Uncaught TypeError|'NoneType' object|object has no attribute|'NoneType' and", re.I),
        CAT_SELECTOR_TYPE_ERROR, 0.85,
        "JS 类型错误",
        "browser_run_js 的表达式中有类型错误(例如对 null 取属性)。",
        "重写 JS 表达式,先对选择器结果判空(?. 或 if/else),或改用"
        "browser_extract_links 工具代替手写 JS。",
        True,
    ),
    # 7. Click miss (element not clickable / not at point / hidden)
    (
        re.compile(
            r"not clickable|not at point|not visible|element is not"
            r"|is not visible|obscured|outside of the viewport|out of bounds",
            re.I,
        ),
        CAT_CLICK_MISS, 0.9,
        "点击落空",
        "click_xy 命中的位置被遮挡、视口外、或元素未渲染在点击点。",
        "先 browser_screenshot 看真实坐标;若有滚动条,先 browser_scroll;"
        "或改用 browser_navigate_to_link(按文字/href 匹配)不依赖坐标。",
        True,
    ),
    # 8. Timeout
    (
        re.compile(r"timeout|timed out|TimeoutError|connection timeout|net::ERR_TIMED_OUT", re.I),
        CAT_TIMEOUT, 0.9,
        "网络/加载超时",
        "页面/资源加载超过阈值,可能是网速慢、目标站限流或代理中断。",
        "先 browser_wait 1-2s 再试一次;若仍超时,降低页面要求(换成更轻的页面);"
        "连续 2 次超时则切到备选来源(例如 arxiv 不可达时改用 semanticscholar)。",
        True,
    ),
    # 9. Network / DNS
    (
        re.compile(r"net::ERR_|DNS|ENOTFOUND|ENETUNREACH|connection refused|ConnectionError", re.I),
        CAT_NETWORK, 0.95,
        "网络/DNS 错误",
        "目标站不可达,DNS 解析失败或被本地代理拦截。",
        "检查网络;若是 VPN 代理问题,改走直连站点;若目标站不可达则换源。",
        True,
    ),
    # 10. Navigation failure (404 / 5xx)
    (
        re.compile(r"404|403|500|502|503|Page not found|not found|forbidden|没有找到|没有权限", re.I),
        CAT_NAVIGATION, 0.9,
        "目标页不可访问",
        "URL 本身 4xx/5xx 或被反爬拒绝。",
        "从 arxiv 公开 search URL 重试,或换成 https://arxiv.org/list/q-fin.RM/ 全列表; "
        "若仍失败,降级到只列目标分类页(q-fin.RM, q-fin.ST)代替。",
        True,
    ),
    # 11. Proxy / VPN
    (
        re.compile(r"webvpn|swufe|easypass|代理|proxy|vpn", re.I),
        CAT_NAVIGATION_PROXY, 0.9,
        "代理/VPN 拦截",
        "通过 webvpn/校园代理访问,部分资源会被拦截或重写 URL。",
        "对于公开资源(arxiv/google scholar)改走 https 直连,不走 webvpn;"
        "对必须走代理的站点,提取真实 URL 后用 browser_navigate_to_link 跳转。",
        True,
    ),
    # 12. Rate limit
    (
        re.compile(r"429|rate limit|too many requests|频繁|限流|too frequent", re.I),
        CAT_RATE_LIMIT, 0.95,
        "触发限流",
        "目标站对当前 IP 限速(429/Too Many Requests)。",
        "browser_wait 10-30s 再重试;或换 IP/换来源(scholar -> arxiv 公开列表)。",
        True,
    ),
    # 13. File download
    (
        re.compile(r"download|file size|Content-Length|pdf|attachment", re.I),
        CAT_FILE_DOWNLOAD, 0.7,
        "文件下载相关",
        "可能是 PDF/附件被下载,非浏览器内嵌。",
        "如果是论文 PDF,直接用 download_url 工具保存,或记录 arxiv ID 用 semantic scholar API。",
        True,
    ),
    # 14. Extract empty
    (
        re.compile(r"no items|no result|empty list|没有匹配|未找到|container not found|没有找到", re.I),
        CAT_EXTRACT_EMPTY, 0.85,
        "提取结果为空",
        "extract/JS 表达式找到了容器/选择器但容器是空的。",
        "用 browser_extract_links 看页面真实链接结构,或换一页(如从搜索结果改到分类页)。",
        True,
    ),
]


def diagnose(error_text: str) -> ErrorDiagnosis:
    """Map a raw error message to a structured :class:`ErrorDiagnosis`.

    Falls back to :data:`CAT_UNEXPECTED` with the raw text when no
    pattern matches. Never raises.
    """
    if not error_text:
        return ErrorDiagnosis(
            category=CAT_UNEXPECTED,
            short="未知错误",
            detail="(empty error text)",
            recovery_hint="读 screenshot 重新观察,或 ask_user。",
            can_retry=True,
            confidence=0.0,
        )
    for pat, cat, conf, short, detail, hint, can_retry in _PATTERNS:
        m = pat.search(error_text)
        if not m:
            continue
        # Trim detail to keep prompts small.
        snippet = (error_text[:200] + "...") if len(error_text) > 200 else error_text
        return ErrorDiagnosis(
            category=cat,
            short=short,
            detail=detail + f"\n原始信息: {snippet}",
            recovery_hint=hint,
            can_retry=can_retry,
            confidence=conf,
        )
    snippet = (error_text[:200] + "...") if len(error_text) > 200 else error_text
    return ErrorDiagnosis(
        category=CAT_UNEXPECTED,
        short="未识别的错误",
        detail="错误信息没有匹配任何已知模式。",
        recovery_hint=f"读 screenshot 重新观察,或 ask_user。原文: {snippet}",
        can_retry=True,
        confidence=0.3,
    )


def short_label(category: str) -> str:
    """Return a one-word label the supervisor can print on a single line."""
    return {
        CAT_OK: "ok",
        CAT_TIMEOUT: "timeout",
        CAT_NAVIGATION: "4xx/5xx",
        CAT_NAVIGATION_PROXY: "proxy",
        CAT_SELECTOR_NULL: "selector-null",
        CAT_SELECTOR_TYPE_ERROR: "selector-typeerr",
        CAT_CLICK_MISS: "click-miss",
        CAT_LOGIN_WALL: "login-wall",
        CAT_CAPTCHA: "captcha",
        CAT_OVERLAY: "overlay",
        CAT_EXTRACT_EMPTY: "extract-empty",
        CAT_RUNJS_INVALID_PARAMS: "cdp-bad-params",
        CAT_FILE_DOWNLOAD: "file-download",
        CAT_RATE_LIMIT: "rate-limit",
        CAT_NETWORK: "network",
        CAT_UNEXPECTED: "unexpected",
    }.get(category, category)


__all__ = [
    "ErrorDiagnosis",
    "diagnose",
    "short_label",
    "CAT_OK",
    "CAT_TIMEOUT",
    "CAT_NAVIGATION",
    "CAT_NAVIGATION_PROXY",
    "CAT_SELECTOR_NULL",
    "CAT_SELECTOR_TYPE_ERROR",
    "CAT_CLICK_MISS",
    "CAT_LOGIN_WALL",
    "CAT_CAPTCHA",
    "CAT_OVERLAY",
    "CAT_EXTRACT_EMPTY",
    "CAT_RUNJS_INVALID_PARAMS",
    "CAT_FILE_DOWNLOAD",
    "CAT_RATE_LIMIT",
    "CAT_NETWORK",
    "CAT_UNEXPECTED",
]
