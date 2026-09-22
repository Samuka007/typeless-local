"""Smoke test for the pynput hotkey hook (Windows WH_KEYBOARD_LL path).

Spawns the real dictation client as a subprocess, then injects a genuine
F9 press/hold/release through pynput's Controller (SendInput -> travels the
full low-level-hook chain exactly like a physical keypress) and asserts the
hook reacted: recording started, then the utterance was processed or
discarded (silence/short is fine - we are testing the hook, not the mic).

Run:  uv run python scripts/hook_smoke.py   (server must be running)
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:  # our own prints must survive cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
READ_TIMEOUT = 40.0


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.dictate", "--raw", "--key", "f9"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    done = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            if "cancelled" in line or "✔" in line or "✖" in line or "(empty" in line:
                done.set()

    threading.Thread(target=reader, daemon=True).start()
    print("[*] dictate client starting ...")
    time.sleep(6)  # let it boot (asr.ensure_server + listener start)

    from pynput.keyboard import Controller, Key

    kc = Controller()
    print("[*] injecting hotkey click #1 (start) ...")
    kc.press(Key.f9)
    time.sleep(1.2)   # held: OS key-repeat fires; toggle debounce must eat it
    kc.release(Key.f9)
    time.sleep(0.5)
    print("[*] injecting hotkey click #2 (stop)")
    kc.press(Key.f9)
    time.sleep(0.1)
    kc.release(Key.f9)

    if not done.wait(READ_TIMEOUT):
        print("[!] no pipeline result within timeout")

    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()

    print("--- client output ---")
    for ln in lines:
        print("   ", ln)

    # toggle must have started exactly one recording despite key-repeat
    rec_ok = sum("REC" in ln for ln in lines) == 1
    result_ok = any(
        ("cancelled" in ln) or ("✔" in ln) or ("✖" in ln) or ("(empty" in ln)
        for ln in lines
    )
    if rec_ok and result_ok:
        print("[PASS] LL hook fired on injected press+release; pipeline ran.")
        return 0
    print("[FAIL] hook did not react as expected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
