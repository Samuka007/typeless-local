"""LLM rephrase module: turn raw ASR output into polished, ready-to-paste text.

Strategy: a single fast non-streaming call with a tight prompt, optimized for
speed (small max_tokens, low-latency providers like DeepSeek/Qwen-turbo).
Falls back to the raw transcript on any failure so dictation never blocks hard.
"""
import asyncio

import httpx

from . import settings

# One shared async client = connection reuse (keeps p50 latency low).
# Rebuilt automatically whenever its config fingerprint changes.
_client: httpx.AsyncClient | None = None
_client_fp: tuple = ()


def _fingerprint(s: dict) -> tuple:
    # Everything that shapes the request payload or connection; the shared
    # client is rebuilt whenever any of these change in settings.json.
    return (s["llm_base_url"], s["llm_model"], s["llm_api_key"],
            s["llm_max_tokens"], s["llm_reasoning"])


def _completion_kwargs(s: dict, *, small: bool = False) -> dict:
    """Payload extras honoring the WebUI budget/thinking knobs.

    Reasoning models (deepseek-flash etc.) spend max_tokens on the thinking
    chain BEFORE the visible content, so an undersized budget yields
    finish_reason=length with EMPTY content. We also drop the thinking chain
    entirely by default (llm_reasoning=none -> reasoning_effort=none): polish
    doesn't need it and it triples the latency. Not every provider accepts
    reasoning_effort -- callers fall back without it on a 400.
    """
    kw: dict = {
        "max_tokens": 128 if small else s["llm_max_tokens"],
        "stream": False,
    }
    if s["llm_reasoning"] != "auto":
        kw["reasoning_effort"] = s["llm_reasoning"]
    return kw

BASE_PROMPT = (
    "You are a voice-dictation post-processor. The user speaks casually; you "
    "return only the cleaned-up text with no commentary. Rules: fix speech "
    "recognition errors and homophones from context; add correct punctuation; "
    "keep the speaker's language (Chinese in -> Chinese out, English in -> "
    "English out; do NOT translate); keep meaning and tone; merge obvious "
    "self-corrections; never answer questions contained in the dictation; "
    "output plain text only."
)

# Per-style additions layered on top of the base prompt (WebUI-selectable).
STYLE_PROMPTS: dict[str, str] = {
    "default": "",
    "email": (
        "Style: a ready-to-send email body. Salutation-free; complete "
        "sentences; polite and concise; expand dictation fragments like "
        "'meeting tomorrow' into a proper sentence."
    ),
    "chat": (
        "Style: instant-messaging message. Keep it short and natural, as the "
        "speaker would have typed it themselves; no formal wording."
    ),
    "notes": (
        "Style: personal notes. Compact bullet points if there are multiple "
        "items; keep keywords; drop filler words aggressively."
    ),
}


def _get_client(s: dict | None = None) -> httpx.AsyncClient:
    global _client, _client_fp
    s = s or settings.get()
    fp = _fingerprint(s)
    if _client is None or _client.is_closed or fp != _client_fp:
        if _client is not None and not _client.is_closed:
            try:
                asyncio.get_running_loop().create_task(_client.aclose())
            except RuntimeError:
                pass  # no running loop (first call); nothing to close
        _client = httpx.AsyncClient(
            base_url=s["llm_base_url"].rstrip("/"),
            headers={"Authorization": f"Bearer {s['llm_api_key']}"},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        _client_fp = fp
    return _client


def _system_prompt(style: str, custom: str) -> str:
    parts = [BASE_PROMPT]
    add = STYLE_PROMPTS.get(style, "")
    if add:
        parts.append(add)
    if custom.strip():
        # User's own rules go last and are phrased as overrides.
        parts.append("Additional user instructions (take precedence where "
                     "they conflict with the above): " + custom.strip())
    return " ".join(parts)


async def rephrase(text: str, timeout: float = 6.0, style: str = "default",
                  custom_prompt: str = "") -> str:
    """Polish dictated text via LLM; return the original on any failure."""
    raw = text.strip()
    if not raw:
        return raw
    s = settings.get()
    system = _system_prompt(style, custom_prompt or s["custom_prompt"])
    payload = {
        "model": s["llm_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": raw},
        ],
        "temperature": 0.1,
        **_completion_kwargs(s),
    }
    try:
        try:
            resp = await asyncio.wait_for(
                _get_client(s).post("/chat/completions", json=payload),
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response is None or e.response.status_code != 400:
                raise
            # Provider rejected an extra param (e.g. reasoning_effort): retry
            # once with the plain vanilla payload.
            print("[rephrase] 400 (param rejected); retrying vanilla payload")
            vanilla = {k: v for k, v in payload.items()
                       if k not in ("reasoning_effort",)}
            resp = await asyncio.wait_for(
                _get_client(s).post("/chat/completions", json=vanilla),
                timeout=timeout,
            )
            resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
        return out or raw
    except Exception as e:  # noqa: BLE001 - degrade gracefully, never lose text
        print(f"[rephrase] LLM failed ({type(e).__name__}: {e}); using raw text")
        return raw


async def test_connection() -> dict:
    """WebUI 'Test connection': try a tiny completion, report verdict + latency."""
    import time as _t

    s = settings.get()
    t0 = _t.monotonic()
    if not s["llm_api_key"]:
        return {"ok": False, "error": "API Key 为空", "ms": 0}
    try:
        resp = await asyncio.wait_for(
            _get_client(s).post(
                "/chat/completions",
                json={
                    "model": s["llm_model"],
                    "messages": [{"role": "user", "content": "ping"}],
                    **_completion_kwargs(s, small=True),
                },
            ),
            timeout=15.0,
        )
        resp.raise_for_status()
        return {"ok": True, "error": "", "ms": int((_t.monotonic() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        detail = ""
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            detail = e.response.text[:200]
        return {"ok": False, "error": f"{type(e).__name__}: {e} {detail}",
                "ms": int((_t.monotonic() - t0) * 1000)}


async def aclose() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
