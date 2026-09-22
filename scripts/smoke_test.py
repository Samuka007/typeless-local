"""Smoke test: record 5s from mic -> transcribe -> LLM polish, print both.

Usage: .venv/Scripts/python.exe scripts/smoke_test.py [--no-record]
"""
import argparse
import time

import sounddevice as sd
import soundfile as sf

from server import asr, rephrase
from server.config import LLM_API_KEY

SR = 16_000
PROMPT = "以下是普通话的句子。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-record", action="store_true", help="use logs/utterance.wav instead of mic")
    args = ap.parse_args()

    if args.no_record:
        wav = "logs/utterance.wav"
    else:
        print(f"Recording 5 seconds — speak now ...")
        audio = sd.rec(int(5 * SR), samplerate=SR, channels=1, dtype="int16")
        sd.wait()
        sf.write("logs/smoke.wav", audio, SR, subtype="PCM_16")
        wav = "logs/smoke.wav"

    asr.ensure_server()
    t0 = time.time()
    text = asr.transcribe_wav(wav, prompt=PROMPT)
    t1 = time.time()
    print(f"[ASR {t1 - t0:.2f}s] {text!r}")

    if not LLM_API_KEY:
        print("[LLM] no API key in .env, skipping rephrase")
        return
    out = rephrase.rephrase_sync(text) if hasattr(rephrase, "rephrase_sync") else None
    if out is None:
        import asyncio

        out = asyncio.run(rephrase.rephrase(text))
    print(f"[LLM {time.time() - t1:.2f}s] {out!r}")


if __name__ == "__main__":
    main()
