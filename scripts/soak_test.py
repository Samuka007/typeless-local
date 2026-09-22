"""4-minute soak test: does a healthy whisper-server stay healthy untouched?

Decisive experiment for the recurring "instance wedges (alive but deaf)"
observation. One instance, zero interference, simultaneous curl + python
probes every 15s, instance count tracked to catch hidden spawners.

Run:  uv run python scripts/soak_test.py
"""
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from server import asr  # noqa: E402


def curl_probe() -> str:
    r = subprocess.run(
        ["curl", "-s", "-m", "8", "-o", "NUL", "-w", "%{http_code}",
         "-F", "file=@models/.probe.wav", "-F", "response_format=json",
         "http://127.0.0.1:8178/inference"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    return r.stdout.strip() or f"curl-exit-{r.returncode}"


def py_probe() -> str:
    t0 = time.time()
    try:
        asr._post_inference("models/.probe.wav", timeout=8)
        return f"py-200-{time.time()-t0:.2f}s"
    except Exception as e:  # noqa: BLE001
        return f"py-FAIL-{type(e).__name__}:{e}"[:90]


print("ensuring exactly one healthy instance ...", flush=True)
asr.kill_strays()
asr._spawn_detached()
if not asr.wait_ready(timeout=240):
    print("FATAL: fresh instance never became ready", flush=True)
    sys.exit(1)
print(f"t+0: ready. instances={asr._instances_running()}. Soaking 4 min ...\n", flush=True)

for i in range(1, 17):
    time.sleep(15)
    c = curl_probe()
    p = py_probe()
    n = asr._instances_running()
    flag = "OK" if (c == "200" and p.startswith("py-200")) else "<<< WEDGE/FAIL"
    print(f"t+{i*15:3d}s  curl={c}  {p}  instances={n}  {flag}", flush=True)

print("\nsoak complete", flush=True)
