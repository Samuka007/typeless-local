"""Runtime interaction settings: single source of truth for the WebUI.

Deployment-level knobs (ports, model paths, API keys) stay in `.env` --
they require a process restart and are machine-specific. Everything a user
may want to change while dictating lives in `settings.json`, is editable
from the built-in WebUI, and is hot-reloaded by the dictate client (it
re-reads the file on every utterance and on indicator refresh, so no
restart and no client reconnect is ever needed).
"""
import json
import logging
import threading
from pathlib import Path

log = logging.getLogger("settings")

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "settings.json"

# --- bootstrap values from .env (used ONLY the first time settings.json is
# --- created; after that the file is the single source of truth and .env
# --- values are ignored until the file is deleted) ---

def _env_seeds() -> dict:
    """Deployment defaults from .env, read lazily to avoid import cycles."""
    from . import config as cfg

    return {
        "whisper_model": str(Path(cfg.WHISPER_MODEL).relative_to(ROOT)).replace("\\", "/")
        if str(cfg.WHISPER_MODEL).startswith(str(ROOT)) else cfg.WHISPER_MODEL,
        "whisper_language": cfg.WHISPER_LANGUAGE,
        "whisper_prompt": cfg.WHISPER_PROMPT,
        "llm_base_url": cfg.LLM_BASE_URL,
        "llm_model": cfg.LLM_MODEL,
        "llm_api_key": cfg.LLM_API_KEY,
    }

DEFAULTS: dict = {
    # --- interaction (WebUI-editable, hot-reloaded) ---
    "hotkey": "alt_l",          # pynput key name (alt_l / alt_r->AltGr / f9 ...)
    "trigger": "toggle",        # "toggle" = click to start, click again to stop
    "mode": "paste",            # "paste" (Ctrl+V) | "type" (simulated keys)
    "polish": True,             # LLM rephrase on/off
    "style": "default",         # default | email | chat | notes
    "indicator": True,          # on-screen REC pill while recording
    "custom_prompt": "",        # extra user instructions appended to the system prompt
    # --- deployment (WebUI-editable; a whisper change needs the restart button) ---
    "whisper_model": "models/ggml-large-v3-turbo.bin",
    "whisper_language": "auto",  # auto | zh | en (auto costs ~+1.3s detection)
    "whisper_prompt": "",        # whisper steering prompt (CJK punctuation aid)
    "llm_base_url": "https://api.deepseek.com/v1",
    "llm_model": "deepseek-chat",
    "llm_api_key": "",
}

_lock = threading.Lock()


def _coerce(patch: dict) -> dict:
    """Whitelist + validate a patch before it touches the store."""
    out: dict = {}
    if "hotkey" in patch:
        hk = str(patch["hotkey"]).strip().lower()
        if hk:
            out["hotkey"] = hk
    if "trigger" in patch:
        out["trigger"] = "toggle" if patch["trigger"] == "toggle" else "hold"
    if "mode" in patch:
        out["mode"] = "type" if patch["mode"] == "type" else "paste"
    if "polish" in patch:
        out["polish"] = bool(patch["polish"])
    if "style" in patch:
        out["style"] = str(patch["style"]) if patch["style"] in (
            "default", "email", "chat", "notes") else "default"
    if "indicator" in patch:
        out["indicator"] = bool(patch["indicator"])
    for k in ("custom_prompt", "whisper_prompt", "llm_base_url",
              "llm_model", "llm_api_key"):
        if k in patch:
            out[k] = str(patch[k] or "")
    if "whisper_model" in patch:
        m = str(patch["whisper_model"] or "").strip().replace("\\", "/")
        if m:
            out["whisper_model"] = m
    if "whisper_language" in patch:
        out["whisper_language"] = (str(patch["whisper_language"]).strip().lower()
                                   if patch["whisper_language"] in ("auto", "zh", "en")
                                   else "auto")
    return out


def _read_file() -> dict | None:
    if not PATH.exists():
        return None
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - corrupt file: keep defaults
        log.warning("settings.json unreadable (%s); using defaults", e)
        return None


def get() -> dict:
    """Current settings.

    First run (no settings.json yet): DEFAULTS + .env seeds, persisted to
    disk. Every later run: file wins over everything -- .env is only the
    bootstrap. Delete settings.json to fall back to .env values.
    """
    with _lock:
        stored = _read_file()
        if stored is None:
            data = {**DEFAULTS, **_env_seeds()}
            PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            return data
        data = dict(DEFAULTS)
        data.update(stored)
        # Migration: files written before the LLM/whisper fields existed lack
        # those keys entirely -- seed them from .env. A key PRESENT in the
        # file (even empty, deliberately disabled) always wins.
        seeds = _env_seeds()
        for k, v in seeds.items():
            if k not in stored:
                data[k] = v
        return data


def available_models() -> list[str]:
    """Whisper models present in models/ (excludes VAD/probe files)."""
    models_dir = ROOT / "models"
    out = []
    for p in sorted(models_dir.glob("*.bin")):
        n = p.name.lower()
        if "silero" in n or "vad" in n or n.startswith("."):
            continue
        out.append(f"models/{p.name}")
    return out


def update(patch: dict) -> dict:
    """Validate, persist and return the new settings (atomic write)."""
    clean = _coerce(patch)
    if not clean:
        return get()
    with _lock:
        data = dict(DEFAULTS)
        stored = _read_file()
        if stored is None:
            data.update(_env_seeds())
        else:
            data.update(stored)
            seeds = _env_seeds()
            for k, v in seeds.items():
                if k not in stored:
                    data[k] = v
        data.update(clean)
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PATH)  # atomic on same volume
        return data
