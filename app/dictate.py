"""Typeless-like hotkey dictation client (Windows low-level-hook best practice).

Usage:
    uv run typeless-dictate                 # hotkey/trigger from settings.json
    uv run typeless-dictate --key f9        # CLI overrides (highest priority)
    uv run typeless-dictate --trigger hold  # press-and-hold instead of toggle
    uv run typeless-dictate --raw           # skip LLM polish for this session

Interaction (Typeless-style):
    toggle  -- single click starts recording, a second click stops it
    hold    -- hold the key while speaking, release to stop
    Esc while recording cancels the utterance.

Settings & hot-reload:
    Interaction settings live in settings.json (edited via the built-in WebUI
    at /ui, or by hand). The client re-reads the file before every utterance:
    hotkey, trigger, output mode and the REC indicator all apply without any
    restart. CLI flags, when given, win over the file.

Hook design (pynput -> WH_KEYBOARD_LL):
    The press/release callbacks ONLY raise asyncio events; every heavy step
    (opening the audio stream, ASR, LLM, text output, the tkinter overlay)
    runs in asyncio tasks / worker threads. On Windows a slow LL hook
    callback lags *every keystroke system-wide*, so the hook path must stay
    sub-millisecond.

Flow:
    hotkey -> 16kHz mono WAV -> local whisper (Vulkan GPU) -> optional LLM
    rephrase (fallback: raw text) -> paste/type at the cursor.
"""
import argparse
import asyncio
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from server.config import SERVER_URL
from server.settings import get as load_settings

SR = 16_000
MIN_SPEECH_S = 0.35
RMS_SILENCE = 0.0015

# Physical right Alt arrives as AltGr (VK_RMENU) on Windows layouts/IMEs --
# pynput reports Key.alt_gr, never Key.alt_r (verified by injection probe).
KEY_ALIASES = {
    "alt_r": "alt_gr", "ralt": "alt_gr",
    "right_alt": "alt_gr", "rightalt": "alt_gr",
}


# --- audio capture -----------------------------------------------------------
class Recorder:
    def __init__(self) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self.frames: list[np.ndarray] = []

    def start(self) -> None:
        self.frames = []
        self._stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="int16",
            blocksize=800, callback=self._cb,
        )
        self._stream.start()

    def _cb(self, indata, _frames, _time, _status) -> None:
        self._q.put(indata.copy())

    def stop(self) -> str | None:
        """Stop capture, save wav, return path (None if silence/too short)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._q.empty():
            self.frames.append(self._q.get())
        if not self.frames:
            return None
        audio = np.concatenate(self.frames)
        dur = len(audio) / SR
        if dur < MIN_SPEECH_S:
            return None
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) / 32768.0)
        if rms < RMS_SILENCE:
            return None
        path = "logs/utterance.wav"
        sf.write(path, audio, SR, subtype="PCM_16")
        return path


# --- recording indicator (tkinter overlay, stdlib only) -----------------------
class RecIndicator:
    """Small always-on-top ● REC pill at the top of the screen.

    All tkinter calls happen on ONE dedicated thread (Tcl is single-apartment:
    touching the root from any other thread raises 'Calling Tcl from different
    apartment'). Other threads just post show/hide requests to a queue. Every
    failure is swallowed: the indicator must never take dictation down.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[bool] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def show(self) -> None:
        self._t0 = time.time()  # each utterance counts from zero
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._tk_loop, daemon=True, name="rec-indicator")
            self._thread.start()
        self._q.put(True)

    def hide(self) -> None:
        self._q.put(False)

    def _tk_loop(self) -> None:
        try:
            import tkinter as tk
        except Exception:  # noqa: BLE001 - headless/broken tk: indicator off
            return
        root = None
        lbl = None
        while True:
            # Block while hidden (zero CPU); pump at 20 Hz while shown.
            try:
                want = self._q.get(timeout=0.05 if root is not None else None)
            except queue.Empty:
                want = None
            try:
                if want is True and root is None:
                    root = tk.Tk()
                    root.overrideredirect(True)  # borderless
                    root.attributes("-topmost", True)
                    sw = root.winfo_screenwidth()
                    sh = root.winfo_screenheight()
                    # bottom-center, just above the taskbar
                    root.geometry(f"+{max(0, sw // 2 - 70)}+{max(0, sh - 90)}")
                    root.configure(bg="#16181d")
                    lbl = tk.Label(
                        root, text="", fg="#ff4d5e", bg="#16181d",
                        font=("Segoe UI", 11, "bold"),
                    )
                    lbl.pack(padx=10, pady=5)
                elif want is False and root is not None:
                    root.destroy()
                    root = lbl = None
                if root is not None:
                    e = max(0, int(time.time() - self._t0))
                    lbl.config(text=f"  ● REC {e // 60}:{e % 60:02d}  ")
                    # No mainloop on this thread: update() paints the window
                    # and processes events (an earlier build skipped this and
                    # the window never rendered at all).
                    root.update()
            except Exception:  # noqa: BLE001 - window gone/tk broken: reset
                try:
                    if root is not None:
                        root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                root = lbl = None


# --- output ------------------------------------------------------------------
def output_text(text: str, mode: str) -> None:
    import pyperclip
    from pynput.keyboard import Controller, Key

    if mode == "paste":
        old = ""
        try:
            old = pyperclip.paste()
        except Exception:  # noqa: BLE001
            pass
        pyperclip.copy(text)
        time.sleep(0.05)
        ctrl = Controller()
        with ctrl.pressed(Key.ctrl):
            ctrl.press("v")
            ctrl.release("v")
        # restore user clipboard shortly after the paste lands
        threading.Timer(1.0, lambda: _safe_copy(pyperclip, old)).start()
    else:
        # Controller.type() uses KEYEVENTF_UNICODE on Windows -> CJK-safe
        Controller().type(text)


def _safe_copy(pyperclip, text: str) -> None:  # noqa: ANN001
    try:
        pyperclip.copy(text)
    except Exception:  # noqa: BLE001
        pass


# --- main loop ---------------------------------------------------------------
async def run(key: str | None, mode: str | None, trigger: str | None,
              raw: bool) -> None:
    import httpx
    from pynput.keyboard import KeyCode, Key, Listener

    rec = Recorder()
    ind = RecIndicator()
    http = httpx.AsyncClient(timeout=60.0, trust_env=False)

    # Preflight: the FastAPI server owns the whisper-server lifecycle and
    # warms it up in the background, so wait here until ASR reports ready.
    deadline = time.time() + 300
    while True:
        try:
            h = await http.get(f"{SERVER_URL}/health", timeout=5.0)
            h.raise_for_status()
            if h.json().get("asr"):
                break
            if time.time() > deadline:
                print("[!] ASR backend still not ready after 5 min; "
                      "see the server pane / logs/whisper-server.log")
                return
            print("    ...ASR warming up", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[!] typeless-server not reachable at {SERVER_URL}")
            print("    Start it first:  uv run typeless-server   (WebUI: http://127.0.0.1:8765/ui)")
            return
        await asyncio.sleep(3)

    loop = asyncio.get_running_loop()
    start_evt = asyncio.Event()   # recording should begin
    stop_evt = asyncio.Event()    # recording should end
    state = {"recording": False, "cancel": False, "t0": 0.0, "last": 0.0}

    # Effective interaction config: CLI override > settings.json (hot-reloaded
    # before every utterance, so WebUI edits apply live).
    cfg = {}

    def refresh_cfg() -> None:
        s = load_settings()
        cfg["key"] = resolve_key(key or s["hotkey"])
        cfg["trigger"] = trigger or s["trigger"]
        cfg["mode"] = mode or s["mode"]
        cfg["indicator"] = s["indicator"]

    def resolve_key(name: str):
        # 'f9'/'alt_l'/'esc'/'scroll_lock' -> Key enum; 'a' -> KeyCode char.
        # 'alt_r' (and friends) map onto alt_gr: that's what Windows reports
        # for the physical right-Alt key.
        n = name.lower().strip()
        n = KEY_ALIASES.get(n, n)
        return getattr(Key, n, None) or KeyCode.from_char(n)

    def on_press(k):
        # Hook thread: must stay fast. Only flags + event sets here.
        hk, trig = cfg.get("key"), cfg.get("trigger")
        if k != hk:
            if k == Key.esc and state["recording"]:
                state["cancel"] = True
                loop.call_soon_threadsafe(stop_evt.set)  # Esc stops immediately
            return
        now = time.time()
        if trig == "toggle":
            # Key-repeat debounce: a held key fires OS repeats that would
            # otherwise flip toggle start/stop many times per second.
            if now - state["last"] < 0.3:
                return
            state["last"] = now
            if not state["recording"]:
                state.update(recording=True, cancel=False, t0=now)
                loop.call_soon_threadsafe(start_evt.set)
            else:
                loop.call_soon_threadsafe(stop_evt.set)  # second click stops
        elif not state["recording"]:  # hold: start on press, stop on release
            state.update(recording=True, cancel=False, t0=now)
            loop.call_soon_threadsafe(start_evt.set)

    def on_release(k):
        if k == cfg.get("key") and state["recording"] and cfg.get("trigger") == "hold":
            loop.call_soon_threadsafe(stop_evt.set)

    refresh_cfg()
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    async def report(recording: bool) -> None:
        """Best-effort live state push for the WebUI dot; never fatal."""
        try:
            await http.post(f"{SERVER_URL}/client-status",
                            json={"recording": recording}, timeout=3.0)
        except Exception:  # noqa: BLE001
            pass

    output_lock = asyncio.Lock()
    tasks: set[asyncio.Task] = set()

    async def process(wav_path: str, t0: float) -> None:
        """ASR + LLM + output; runs as a task so the next utterance can start."""
        try:
            with open(wav_path, "rb") as fh:
                r = await http.post(
                    f"{SERVER_URL}/dictate",
                    files={"file": ("u.wav", fh, "audio/wav")},
                    data={"raw": "true" if raw else "false"},
                )
            r.raise_for_status()
            out = (r.json().get("polished") or r.json().get("text") or "").strip()
            dt = time.time() - t0
            if out:
                print(f"  ✔ {dt:.1f}s  {out}", flush=True)
                # serialize insertion so concurrent utterances keep order
                async with output_lock:
                    await asyncio.to_thread(output_text, out, cfg["mode"])
            else:
                print("  ○ (empty transcript)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ✖ pipeline error: {type(e).__name__}: {e}", flush=True)

    print(f"[*] Dictation ready. Click [{(key or load_settings()['hotkey']).upper()}] "
          f"to start, click again to stop (trigger={cfg['trigger']}).")
    print(f"[*] Esc during recording cancels. Ctrl+C exits. "
          f"mode={cfg['mode']} llm={'off' if raw else 'on'}  ·  WebUI: http://127.0.0.1:8765/ui")

    while True:
        refresh_cfg()  # pick up WebUI edits (hotkey/trigger/mode/indicator)
        await start_evt.wait()
        start_evt.clear()
        # stream open can take tens of ms -> keep it off the hook thread
        rec.start()
        if cfg["indicator"]:
            ind.show()  # just enqueue; the tk thread does the real work
        await report(True)
        print("  ● REC ...", flush=True)

        await stop_evt.wait()
        stop_evt.clear()
        cancel = state["cancel"]
        state["cancel"] = False
        t0 = state["t0"]
        state["recording"] = False
        wav = await asyncio.to_thread(rec.stop)
        ind.hide()
        await report(False)

        if cancel or wav is None:
            print("  ○ cancelled / silence", flush=True)
            continue
        task = asyncio.create_task(process(wav, t0))
        tasks.add(task)
        task.add_done_callback(tasks.discard)


def main() -> None:
    # Piped stdout defaults to cp1252/cp936; force UTF-8 so status glyphs
    # (●/✔/○) never crash the loop when output is captured or redirected.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="typeless-local hotkey dictation")
    ap.add_argument("--key", default=None,
                    help="hotkey (pynput name); default: settings.json hotkey")
    ap.add_argument("--trigger", choices=["toggle", "hold"], default=None,
                    help="click-to-toggle (default) or press-and-hold")
    ap.add_argument("--mode", choices=["paste", "type"], default=None,
                    help="text insertion; default: settings.json mode")
    ap.add_argument("--raw", action="store_true", help="skip LLM rephrase")
    args = ap.parse_args()

    try:
        asyncio.run(run(args.key, args.mode, args.trigger, args.raw))
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)


if __name__ == "__main__":
    main()
