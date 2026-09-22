"""Foreground whisper-server runner for a zellij pane.

Runs whisper-server in the FOREGROUND with parameters from settings.json
(model/language stay in sync with the WebUI), and prints logs directly to
the pane -- visible monitoring instead of a detached log file.

Lifecycle contract: this is the long-lived service; FastAPI (typeless-server)
is merely its client and never kills a healthy instance. Ctrl+C here stops
ASR; the next dictate request auto-respawns it (detached, logged to
logs/whisper-server.log).

Run:  uv run python -m server.whisper_serve
"""
import logging
import os
import subprocess
import sys
import time

from . import asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("whisper_serve")


def main() -> int:
    # Single-instance discipline: SO_REUSEADDR lets a stray bind too, and
    # double-bound instances mean connection routing lottery.
    n = asr._instances_running()
    if n:
        print(f"[!] {n} whisper-server instance(s) already running -- reaping first.")
        asr.kill_strays()
        if asr._instances_running():
            print("[x] failed to reap strays; aborting so we never double-bind.")
            return 1

    cmd = asr.build_command()
    print("[*] " + " ".join(cmd), flush=True)
    print("[*] Ctrl+C to stop. FastAPI will auto-respawn (detached) on demand.", flush=True)
    t0 = time.time()
    try:
        # stdout/stderr inherit the pane directly -- live GPU/model/VAD logs.
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print(f"\n[*] stopped after {time.time() - t0:.0f}s", flush=True)
        return 0
    except FileNotFoundError:
        print(f"[x] whisper-server binary not found: {cmd[0]}", flush=True)
        return 1
    finally:
        # On any exit make sure no child lingers bound to the port.
        if os.name == "nt" and asr._instances_running():
            asr.kill_strays()


if __name__ == "__main__":
    sys.exit(main())
