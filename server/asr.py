"""ASR client: send WAV files to the local whisper.cpp HTTP server.

Lifecycle contract (deliberately simple):
  - whisper-server is a detached, long-lived local service. Once up it stays
    up (warm model in VRAM) even when this FastAPI process exits, so the
    1.6 GB model load is paid once -- not on every restart.
  - Creation happens in a throwaway `python -c` process with pristine
    handles, never directly from inside asyncio machinery.
  - We never kill a healthy instance. Startup has one long patience window;
    repeated force-kill/restart cycles are what made startup unreliable.
  - Readiness = a VALID tiny-wav POST answered. This server build never
    answers malformed/empty probes, so probing must use a real request.
"""
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from . import settings
from .config import (
    WHISPER_BIN,
    WHISPER_PORT,
    WHISPER_THREADS,
    WHISPER_URL,
    WHISPER_VAD_MODEL,
)

log = logging.getLogger("asr")
BASE = WHISPER_URL.rsplit("/inference", 1)[0]

PROBE_TIMEOUT = 30.0    # per-probe: a loading server may be slow to answer
STARTUP_TIMEOUT = 300.0 # cold start variance observed: 20s .. ~3min
RETRY_TIMEOUT = 120.0


def kill_strays() -> None:
    """Kill leftover whisper-server.exe instances (Windows).

    whisper.cpp binds with SO_REUSEADDR, so orphaned instances keep
    answering on the port and pile up as zombies that steal requests.
    Only used before a fresh spawn -- never against a healthy server.
    """
    if os.name != "nt":
        return
    subprocess.run(
        ["taskkill", "/F", "/IM", "whisper-server.exe"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    time.sleep(1.5)  # let the Vulkan driver release the device


def build_command() -> list[str]:
    """whisper-server argv from settings.json (single source of truth).

    Used by both the detached spawner and the foreground pane entry point
    (server.whisper_serve), so a pane-run server and an auto-spawned one
    always agree on model/language with the WebUI settings.
    """
    s = settings.get()
    model_path = Path(s["whisper_model"])
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent.parent / model_path
    return [
        str(WHISPER_BIN),
        "--host", "127.0.0.1",
        "--port", str(WHISPER_PORT),
        "--model", str(model_path),
        "--vad", "--vad-model", WHISPER_VAD_MODEL,
        "--language", s["whisper_language"],
        "--threads", str(WHISPER_THREADS),
        "--flash-attn",
        "--no-timestamps",
    ]


def _spawn_detached() -> None:
    """Create whisper-server via a throwaway clean intermediate process.

    A bare `python -c` spawner has no asyncio/IOCP state, creates the child
    with pristine handles, and exits immediately. The child outlives it.
    """
    spawner = (
        "import subprocess,sys\n"
        "with open(sys.argv[1],'ab') as logf:\n"
        "    subprocess.Popen(sys.argv[2:], stdout=logf, stderr=subprocess.STDOUT)\n"
    )
    log_path = Path(__file__).resolve().parent.parent / "logs" / "whisper-server.log"
    log_path.parent.mkdir(exist_ok=True)
    cmd = build_command()
    log.info("starting whisper-server (detached): %s", " ".join(cmd))

    def _run() -> None:
        try:
            subprocess.Popen(
                [sys.executable, "-c", spawner, str(log_path)] + cmd,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to spawn whisper-server")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30.0)


def _instances_running() -> int:
    """Count running whisper-server.exe instances (Windows)."""
    if os.name != "nt":
        return 0
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq whisper-server.exe", "/FO", "CSV"],
        capture_output=True, text=True, check=False,
    ).stdout
    # CSV header row is "Image Name","PID",... and does NOT contain the
    # image name, so every data row is one real instance. (An earlier
    # `count(...) - 1` off-by-one made single-instance enforcement a no-op,
    # which allowed double-bound instances and connection routing lottery.)
    return out.count("whisper-server.exe")


def _probe_wav() -> Path:
    """0.2s of silence as a valid 16 kHz wav, for readiness probes."""
    p = Path(__file__).resolve().parent.parent / "models" / ".probe.wav"
    if not p.exists():
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<3200h", *([0] * 3200)))
    return p


def _post_inference(wav_path: str | Path, prompt: str = "", timeout: float = 60.0) -> str:
    """POST a wav to whisper-server with stdlib urllib; return transcript.

    Deliberately NOT httpx: httpx requests aimed at this server build from
    inside the uvicorn process intermittently never receive a response while
    curl/urllib from the same box answer instantly. urllib is as plain as
    curl (no pooling, no proxy magic, no async bindings) and is proven to
    work from every context we have tested.
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in (("response_format", "json"),
                        ("language", settings.get()["whisper_language"]),
                        ("temperature", "0.0"),
                        ("prompt", prompt)):
        if value:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                f'\r\n\r\n{value}\r\n'.encode()
            )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{Path(wav_path).name}"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'
        ).encode()
        + Path(wav_path).read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        WHISPER_URL,
        data=b"".join(parts),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("text", "").strip()


def wait_ready(timeout: float = STARTUP_TIMEOUT) -> bool:
    """Block until a valid inference request is answered (False on timeout)."""
    deadline = time.time() + timeout
    wav = _probe_wav()
    while time.time() < deadline:
        probe_to = max(3.0, min(PROBE_TIMEOUT, deadline - time.time()))
        try:
            _post_inference(wav, timeout=probe_to)
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            pass
        time.sleep(2.0)
    return False


_start_lock = threading.Lock()


def _healthy_now() -> bool:
    """One quick probe: is a live instance answering RIGHT NOW?

    Any failure (refusal or hang) counts as unhealthy. whisper-server
    instances occasionally wedge (process alive, listener gone) on this
    Vulkan build; a wedged instance must NOT get the long loading patience
    -- it gets reaped and respawned instead (~30-50s on a warm file cache).
    """
    try:
        _post_inference(_probe_wav(), timeout=6.0)
        return True
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


def ensure_server() -> None:
    """Make sure a single healthy whisper-server is answering."""
    with _start_lock:
        if _healthy_now():
            return
        # Single-instance discipline: the port binds with SO_REUSEADDR, so
        # two instances means every connection randomly reaches a healthy
        # or a wedged one -- probes and requests then flake for no reason.
        n = _instances_running()
        if n > 0:
            log.warning("reaping %d unresponsive whisper-server instance(s)", n)
            kill_strays()
            if _instances_running() > 0:
                raise RuntimeError("failed to reap stray whisper-server instances")
        for wait in (STARTUP_TIMEOUT, RETRY_TIMEOUT):
            _spawn_detached()
            if wait_ready(timeout=wait):
                return
            kill_strays()
        raise RuntimeError(
            f"whisper-server did not come up on port {WHISPER_PORT}; "
            f"see logs/whisper-server.log"
        )


def restart_model() -> dict:
    """Restart whisper-server with the model/language now in settings.json.

    WebUI 'Apply new model' path: reap the running instance, spawn with the
    new settings, wait for readiness. Cold load of the turbo model takes
    ~30s (warm cache) up to a few minutes (cold cache), so the caller gets
    a summary dict instead of exceptions.
    """
    with _start_lock:
        t0 = time.time()
        kill_strays()
        _spawn_detached()
        if wait_ready(timeout=STARTUP_TIMEOUT):
            return {"ok": True, "error": "", "ms": int((time.time() - t0) * 1000)}
        kill_strays()  # do not leave a half-loaded zombie on the port
        return {
            "ok": False,
            "error": "whisper-server failed to become ready; see logs/whisper-server.log",
            "ms": int((time.time() - t0) * 1000),
        }


def transcribe_wav(path: str | Path, prompt: str = "") -> str:
    """POST a 16kHz mono WAV; return the transcript text."""
    ensure_server()
    return _post_inference(path, prompt=prompt, timeout=120.0)
