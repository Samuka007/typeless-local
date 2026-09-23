"""FastAPI service exposing local ASR (+ optional LLM rephrase).

Endpoints:
  GET  /health         -> {"asr": true, "llm": true/false}
  POST /transcribe     -> raw whisper transcript        (multipart: file)
  POST /dictate        -> transcribe + LLM rephrase     (multipart: file, optional "raw"=true)
  POST /rephrase       -> LLM polish of pasted text     (json: {"text": "..."})

Lifecycle: whisper-server is a DETACHED long-lived service; this app only
ensures it exists (spawn if the port is dead) and never stops it, so the
model stays warm in VRAM across restarts. To stop everything manually:
  taskkill /F /IM whisper-server.exe

Run:  uv run typeless-server
"""
import logging
import tempfile
import threading
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import asr, rephrase, settings
from .config import DICTATE_ENGLISH_ONLY, SERVER_PORT, WHISPER_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("main")

# Optional hot-word steering for whisper; extend freely (comma-separated).
# Empty by default: on turbo the CJK steering prompt costs ~0.5s per request.
INITIAL_PROMPT = "" if DICTATE_ENGLISH_ONLY else WHISPER_PROMPT

# --- shared observable state (WebUI + dictate client) ---
_client_status = {"recording": False}
_history: list[dict] = []
_hist_lock = threading.Lock()


def _record_history(entry: dict) -> None:
    with _hist_lock:
        _history.append(entry)
        del _history[:-100]  # keep the last 100


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Warm the ASR backend in a background thread -- startup must never block
    # on a (potentially minutes-long) cold model load. Requests that arrive
    # early simply wait inside their own ensure_server() call.
    log.info("background warm-up of whisper-server (Vulkan, detached) ...")
    threading.Thread(target=asr.ensure_server, daemon=True, name="asr-warmup").start()
    yield
    # Intentionally NOT stopping whisper-server: it is detached and stays
    # warm so the next start (or the dictate client) pays no cold load.
    await rephrase.aclose()


app = FastAPI(title="typeless-local", lifespan=_lifespan)


@app.get("/health")
async def health():
    # Same probe the client trusts: a valid (tiny) inference request, run off
    # the event loop. asr=false simply means still warming up.
    asr_ok = await asyncio.to_thread(asr.wait_ready, 3.0)
    return {"asr": asr_ok, "llm": bool(settings.get()["llm_api_key"])}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        text = await asyncio.to_thread(asr.transcribe_wav, tmp, INITIAL_PROMPT)
    finally:
        tmp.unlink(missing_ok=True)
    return JSONResponse({"text": text})


@app.post("/dictate")
async def dictate(
    file: UploadFile = File(...),
    raw: str = Form("false"),
):
    """Transcribe, then LLM-polish per the runtime settings (unless raw=true)."""
    t0 = time.time()
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio file")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        s = settings.get()
        w_prompt = "" if DICTATE_ENGLISH_ONLY else s["whisper_prompt"]
        text = await asyncio.to_thread(asr.transcribe_wav, tmp, w_prompt)
    finally:
        tmp.unlink(missing_ok=True)
    if not text:
        return JSONResponse({"text": "", "polished": ""})
    polish = s["polish"] and raw.lower() != "true"
    polished = (await rephrase.rephrase(text, style=s["style"]) if polish else "")
    _record_history({
        "ts": time.strftime("%H:%M:%S"),
        "raw": text,
        "polished": polished or text,
        "style": s["style"],
        "dur": round(time.time() - t0, 2),
    })
    return JSONResponse({"text": text, "polished": polished or text})


class RephraseIn(BaseModel):
    text: str


@app.post("/rephrase")
async def rephrase_text(body: RephraseIn):
    out = await rephrase.rephrase(body.text)
    return JSONResponse({"text": out})


class ClientStatus(BaseModel):
    recording: bool = False


@app.post("/client-status")
async def client_status(body: ClientStatus):
    """The dictate client reports its live state (drives the WebUI dot)."""
    _client_status["recording"] = bool(body.recording)
    return JSONResponse({"ok": True})


@app.get("/status")
async def status():
    """Everything the WebUI polls: ASR/LLM health, recording, settings."""
    asr_ok = await asyncio.to_thread(asr._healthy_now)
    return {
        "asr": asr_ok,
        "llm": bool(settings.get()["llm_api_key"]),
        "recording": _client_status["recording"],
        "settings": settings.get(),
    }


@app.get("/history")
async def history():
    with _hist_lock:
        items = list(reversed(_history[-50:]))
    return JSONResponse({"items": items})


@app.get("/settings")
async def get_settings():
    return settings.get()


class SettingsPatch(BaseModel):
    hotkey: str | None = None
    trigger: str | None = None
    mode: str | None = None
    polish: bool | None = None
    style: str | None = None
    indicator: bool | None = None
    custom_prompt: str | None = None
    whisper_model: str | None = None
    whisper_language: str | None = None
    whisper_prompt: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_max_tokens: int | None = None
    llm_reasoning: str | None = None


@app.patch("/settings")
async def patch_settings(body: SettingsPatch):
    """Persist a partial update; clients hot-reload; LLM picks it up live."""
    return settings.update(body.model_dump(exclude_none=True))


@app.get("/whisper/models")
async def whisper_models():
    return {"models": settings.available_models()}


@app.post("/whisper/restart")
async def whisper_restart():
    """Apply the selected model/language: reap + respawn + wait for readiness."""
    result = await asyncio.to_thread(asr.restart_model)
    if not result["ok"]:
        return JSONResponse(result, status_code=502)
    return JSONResponse(result)


@app.post("/llm/test")
async def llm_test():
    """Try a tiny completion with the CURRENT saved settings."""
    return JSONResponse(await rephrase.test_connection())


@app.get("/ui")
async def ui():
    return FileResponse(Path(__file__).resolve().parent / "ui.html")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
