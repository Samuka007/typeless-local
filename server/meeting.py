"""Meeting transcription: capture desktop audio + microphone, transcribe both.

Speaker attribution comes for free from the physical separation of the two
channels -- desktop/loopback audio is the remote party, the microphone is
you. No diarization model needed (whisper.cpp tinydiarize only exists for
small.en-tdrz models anyway; multiple remote speakers all appear as
"[对方]", which is a documented limitation, not a bug).

Capture libraries (deliberately split):
  - loopback: `soundcard` (WASAPI loopback via the default speaker exposed
    as a loopback microphone). sounddevice/PortAudio cannot do loopback.
  - microphone: `sounddevice` (PortAudio). soundcard's MediaFoundation path
    asserts on some USB interfaces (e.g. Shure MV7 wFormatTag) -- verified
    broken on this machine, so it is not used for capture here.

Pipeline: each channel has its own capture thread and adaptive-VAD
segmenter; finished segments go into a queue drained by ONE serial
transcription worker (the whisper-server is effectively single-request;
serial also keeps GPU latency predictable during live meetings).
"""

from __future__ import annotations

import difflib
import logging
import queue
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from . import asr, settings

log = logging.getLogger("meeting")

SAMPLE_RATE = 16000
CHUNK_MS = 100                       # capture block size
FRAME = SAMPLE_RATE * CHUNK_MS // 1000
SILENCE_END = 0.8                    # trailing silence that closes a segment
MIN_SEGMENT_S = 0.3                  # discard shorter blips (coughs, clicks)
MAX_SEGMENT_S = 25.0                 # force-cut very long continuous speech
NOISE_FLOOR_ALPHA = 0.05             # EMA adaptation rate for the noise floor
NOISE_FLOOR_INIT = 0.005
SPEECH_FACTOR = 2.2                  # threshold = floor * this
BLEED_WINDOW = 6.0                   # cross-channel dedup window (s)
BLEED_RATIO = 0.75                   # text similarity treated as the same utterance

_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = _ROOT / "logs" / "transcripts"


# ---------------------------------------------------------------------------
# adaptive VAD segmenter (per channel)

class Segmenter:
    """Streaming energy VAD with a self-adapting noise floor.

    Feeded fixed FRAME-sample chunks; emits finished segments (float32 mono
    16 kHz) via .segments. Speech > noise_floor * SPEECH_FACTOR opens a
    segment; SILENCE_END seconds of quiet close it. The floor adapts only
    while idle, so long meetings follow the room without ever swallowing
    speech.
    """

    def __init__(self) -> None:
        self.floor = NOISE_FLOOR_INIT
        self.buf: list[np.ndarray] = []
        self.buf_frames = 0
        self.sil_frames = 0
        self.max_frames = int(MAX_SEGMENT_S * SAMPLE_RATE)
        self.min_frames = int(MIN_SEGMENT_S * SAMPLE_RATE)
        self.segments: "queue.SimpleQueue[np.ndarray]" = queue.SimpleQueue()
        self.lock = threading.Lock()

    def feed(self, chunk: np.ndarray) -> None:
        # Loopback capture is stereo (frames, 2): a plain reshape(-1) would
        # interleave L/R samples into a chipmunk-speed chimera that whisper's
        # VAD silently rejects (empty transcripts). Downmix properly.
        if chunk.ndim == 2 and chunk.shape[1] > 1:
            chunk = chunk.mean(axis=1, dtype=np.float32, keepdims=True)
        mono = chunk.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(mono)))) or 1e-9
        speech = rms > self.floor * SPEECH_FACTOR
        with self.lock:
            if not speech:
                self.floor = (1 - NOISE_FLOOR_ALPHA) * self.floor + NOISE_FLOOR_ALPHA * rms
            if speech:
                self.buf.append(mono)
                self.buf_frames += mono.size
                self.sil_frames = 0
                done = None
                if self.buf_frames >= self.max_frames:      # hard cut: long monologue
                    done = self._pop_locked()
                if done is not None:
                    self.segments.put(done)
            elif self.buf_frames:
                # inside an open segment: tolerate brief quiet, close on pause
                self.buf.append(mono)
                self.buf_frames += mono.size
                self.sil_frames += mono.size
                if self.sil_frames >= int(SILENCE_END * SAMPLE_RATE):
                    seg = self._pop_locked()
                    if seg is not None:
                        self.segments.put(seg)
            # fully idle: nothing buffered, floor already adapted above

    def _pop_locked(self) -> np.ndarray | None:
        """Join the buffer into a segment; keep the tail after the silence as
        pre-roll for the next one. Caller holds self.lock."""
        data = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        tail = int(SILENCE_END * SAMPLE_RATE)
        keep = data[:-tail] if self.sil_frames >= tail and data.size > tail else data
        self.buf, self.buf_frames, self.sil_frames = [], 0, 0
        if keep.size >= self.min_frames:
            # keep the trailing silence trimmed but leave a little as natural gap
            return keep
        return None


# ---------------------------------------------------------------------------
# session

_PUNCT = re.compile(r"[\s,.!?:;，。！？：；、]")


def _norm_text(t: str) -> str:
    """Punctuation/space/case-insensitive form for bleed comparison."""
    return _PUNCT.sub("", t).lower()


def _is_bleed_pair(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:      # ASR may render the bleed copy shorter/truncated
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= BLEED_RATIO


class _Meeting:
    def __init__(self) -> None:
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.started = time.time()
        self.stopped: float | None = None
        self.error: str = ""
        self.mic_seg = Segmenter()
        self.loop_seg = Segmenter()
        self.q: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
        self.entries: list[dict] = []          # {t, who, text}
        self.lock = threading.Lock()
        self.counts = {"mic": 0, "loop": 0}    # segments emitted per channel
        self.bleed_dropped = 0                 # acoustic-echo duplicates removed
        self._recent: deque[tuple[float, str, str]] = deque(maxlen=24)
        self.stop_evt = threading.Event()
        self.threads: list[threading.Thread] = []
        self.polished: str = ""                # optional AI summary result

    # -- capture threads ----------------------------------------------------
    def _mic_thread(self) -> None:
        try:
            import sounddevice as sd
            seg = self.mic_seg

            def cb(indata, frames, time_info, status):  # noqa: ANN001
                if status:
                    log.debug("mic stream status: %s", status)
                seg.feed(indata.copy())

            stream = None
            for attempt in (1, 2, 3, 4):   # device may need a moment after a
                try:                       # previous session released it
                    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                            dtype="float32", blocksize=FRAME,
                                            callback=cb)
                    break
                except Exception as e:  # noqa: BLE001 (PortAudio busy/init)
                    log.warning("mic open attempt %d failed: %s", attempt, e)
                    if attempt == 4:
                        raise
                    time.sleep(1.5)
            with stream:
                while not self.stop_evt.is_set():
                    self.stop_evt.wait(0.2)
        except Exception as e:  # noqa: BLE001
            self.error = f"mic capture failed: {type(e).__name__}: {e}"
            log.exception("mic capture thread died")
            self.stop_evt.set()

    def _loop_thread(self) -> None:
        try:
            import soundcard as sc
            sp = sc.default_speaker()
            loop_mic = sc.get_microphone(id=str(sp.name), include_loopback=True)
            seg = self.loop_seg
            rec = None
            for attempt in (1, 2, 3, 4):   # WASAPI loopback right after a
                try:                       # previous session stop may refuse
                    rec = loop_mic.recorder(samplerate=SAMPLE_RATE)
                    rec.__enter__()
                    break
                except Exception as e:  # noqa: BLE001 (device busy/release)
                    log.warning("loopback open attempt %d failed: %s", attempt, e)
                    if attempt == 4:
                        raise
                    time.sleep(1.5)
            try:
                while not self.stop_evt.is_set():
                    data = rec.record(numframes=FRAME)
                    seg.feed(data)
            finally:
                rec.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            self.error = f"loopback capture failed: {type(e).__name__}: {e}"
            log.exception("loopback capture thread died")
            self.stop_evt.set()

    # -- transcription worker ----------------------------------------------
    def _worker(self) -> None:
        prompt = settings.get().get("whisper_prompt", "")
        while True:
            try:
                who, pcm = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_evt.is_set() and self.q.empty():
                    return
                continue
            try:
                text = _transcribe_pcm(pcm, prompt)
            except Exception as e:  # noqa: BLE001
                log.exception("segment transcription failed")
                text = ""
            if text and self._claim_text(who, text):
                entry = {
                    "t": time.strftime("%H:%M:%S"),
                    "who": who,
                    "text": text,
                }
                with self.lock:
                    self.entries.append(entry)

    def _claim_text(self, who: str, text: str) -> bool:
        """Register a transcribed segment; False means it is acoustic bleed.

        Headphones hanging next to the mic leak the remote party's voice into
        the mic channel a moment after it already arrived digitally via
        loopback. A near-identical text on the OTHER channel within
        BLEED_WINDOW seconds is that echo; the FIRST copy wins (loopback is
        normally first -- it is digital and instant). Wearing the headphones
        on your head avoids the problem entirely; this is the safety net.
        """
        norm = _norm_text(text)
        now = time.time()
        with self.lock:
            bleed = any(
                w != who and abs(now - t) < BLEED_WINDOW and _is_bleed_pair(norm, n)
                for t, w, n in self._recent
            )
            self._recent.append((now, who, norm))
            if bleed:
                self.bleed_dropped += 1
        if bleed:
            log.info("dropped acoustic bleed on %s channel: %.40s...", who, text)
        return not bleed

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        for tgt, name in ((self._loop_thread, "loop"), (self._mic_thread, "mic"),
                          (self._worker, "worker")):
            t = threading.Thread(target=tgt, name=f"meeting-{name}", daemon=True)
            t.start()
            self.threads.append(t)
        # drain finished segments into the worker queue
        d = threading.Thread(target=self._drain, name="meeting-drain", daemon=True)
        d.start()
        self.threads.append(d)

    def _drain(self) -> None:
        """Move finished segments from both segmenters into the work queue."""
        while not (self.stop_evt.is_set() and self.mic_seg.buf_frames == 0
                   and self.loop_seg.buf_frames == 0):
            got = False
            for seg, who in ((self.loop_seg, "对方"), (self.mic_seg, "我")):
                try:
                    while True:
                        pcm = seg.segments.get_nowait()
                        with self.lock:
                            self.counts["loop" if who == "对方" else "mic"] += 1
                        self.q.put((who, pcm))
                        got = True
                except queue.Empty:
                    pass
            if not got:
                time.sleep(0.1)
        # flush one last time in case a final segment landed after the check
        for seg, who in ((self.loop_seg, "对方"), (self.mic_seg, "我")):
            try:
                while True:
                    pcm = seg.segments.get_nowait()
                    self.q.put((who, pcm))
            except queue.Empty:
                pass

    def stop(self) -> None:
        self.stop_evt.set()
        for t in self.threads:
            t.join(timeout=4.0)
        self.stopped = time.time()

    # -- reporting ----------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            n = len(self.entries)
        return {
            "active": self.stopped is None and not self.stop_evt.is_set(),
            "id": self.id,
            "started": self.started,
            "elapsed": round((self.stopped or time.time()) - self.started, 1),
            "segments": n,
            "counts": dict(self.counts),
            "bleed_dropped": self.bleed_dropped,
            "error": self.error,
        }

    def live(self, after: int = 0) -> dict:
        with self.lock:
            items = self.entries[after:]
            total = len(self.entries)
        return {"entries": items, "total": total, **self.status()}

    def save(self) -> Path:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = TRANSCRIPTS_DIR / f"meeting-{self.id}.md"
        with self.lock:
            entries = list(self.entries)
        lines = [f"# 会议逐字稿 {self.id}", ""]
        cur = None
        for e in entries:
            if e["who"] != cur:
                lines.append(f"\n**[{e['t']} {e['who']}]** ")
                cur = e["who"]
            lines[-1] += e["text"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def plain_text(self) -> str:
        with self.lock:
            entries = list(self.entries)
        out = []
        cur = None
        line = ""
        for e in entries:
            if e["who"] != cur:
                if line:
                    out.append(line)
                line = f"[{e['t']} {e['who']}] {e['text']}"
                cur = e["who"]
            else:
                line += e["text"]
        if line:
            out.append(line)
        return "\n".join(out)


_session: _Meeting | None = None
_session_lock = threading.Lock()


def get_session() -> _Meeting | None:
    return _session


def start() -> dict:
    """Start a new meeting session (exactly one at a time)."""
    global _session
    with _session_lock:
        if _session is not None and _session.status()["active"]:
            return {"ok": False, "error": "already running", "status": _session.status()}
        m = _Meeting()
        m.start()
        _session = m
        return {"ok": True, "error": "", "status": m.status()}


def stop() -> dict:
    """Stop the session, flush remaining segments, persist the transcript.

    The stopped session object is kept in _session (read-only from now on)
    so the UI can still fetch the full transcript after stopping; the next
    start() simply replaces it.
    """
    with _session_lock:
        m = _session
        if m is None:
            return {"ok": False, "error": "no session", "path": ""}
        m.stop()
    try:
        path = m.save()
    except Exception as e:  # noqa: BLE001
        log.exception("failed to save transcript")
        return {"ok": True, "error": f"transcript save failed: {e}",
                "status": m.status(), "path": ""}
    return {"ok": True, "error": m.error, "status": m.status(), "path": str(path)}


# ---------------------------------------------------------------------------
# transcription helper

def _transcribe_pcm(pcm: np.ndarray, prompt: str = "") -> str:
    """Transcribe float32 mono 16 kHz PCM via the warm whisper-server."""
    import struct
    import tempfile

    ints = np.clip(pcm, -1.0, 1.0)
    ints = (ints * 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(ints.tobytes())
    try:
        return asr.transcribe_wav(tmp, prompt)
    finally:
        tmp.unlink(missing_ok=True)
