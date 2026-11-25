#!/usr/bin/env python3
"""Wait for verified weights, run a bounded native CPU/CUDA study, and summarize."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    lease = (ROOT / "runtime/campaign.lock").open("w")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A campaign is already running in this workspace")
    state_file = ROOT / "runtime/campaign-state.json"
    runner = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Campaign interrupted")
    signal.signal(signal.SIGTERM, interrupt)

    def state(phase, **extra):
        value = {"phase": phase, "pid": os.getpid(),
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(state_file)
        print(json.dumps(value), flush=True)

    state("waiting_for_download")
    try:
        deadline = time.monotonic() + 24*60*60
        while True:
            path = ROOT / "runtime/download-state.json"
            if path.exists():
                download = json.loads(path.read_text())
                if download["status"] == "complete":
                    break
                if download["status"] == "error":
                    raise RuntimeError("Download failed; resume it and rerun this campaign")
                stat = Path(f"/proc/{download['pid']}/stat")
                if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    raise RuntimeError("Download process exited before completion")
            if time.monotonic() > deadline:
                raise TimeoutError("Download wait exceeded 24 hours")
            time.sleep(15)
        state("validating_checkpoint")
        subprocess.run([sys.executable, "scripts/validate_model.py"], cwd=ROOT, check=True)
        # CUDA first establishes real GPU activity. Then CPU uses identical dense
        # precision, context, RAM cache, workload and generation settings.
        for profile in ("cuda-16-32", "cpu-32"):
            state("benchmarking", profile=profile)
            runner = subprocess.Popen([sys.executable, "scripts/run_experiment.py", "--profile", profile,
                "--max-tokens", "64", "--repeats", "2", "--quality"], cwd=ROOT)
            if runner.wait():
                raise RuntimeError(f"Profile {profile} failed; raw attempt retained")
            runner = None
        subprocess.run([sys.executable, "scripts/summarize.py"], cwd=ROOT, check=True)
        state("complete", summary="results/summary.json", report="docs/RESULTS.md")
    except (Exception, KeyboardInterrupt) as exc:
        if runner is not None and runner.poll() is None:
            runner.terminate()
            try:
                runner.wait(timeout=30)
            except subprocess.TimeoutExpired:
                runner.kill()
                runner.wait()
        state("failed", error=f"{type(exc).__name__}: {exc}")
        subprocess.run([sys.executable, "scripts/summarize.py"], cwd=ROOT)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
