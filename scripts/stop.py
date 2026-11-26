#!/usr/bin/env python3
"""Stop only processes whose workspace-specific identity can be verified."""
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("download", "campaign", "server"):
        raise SystemExit("Usage: python3 scripts/stop.py download|campaign|server")
    target = sys.argv[1]
    state_name, marker = {"download": ("download-pid.json", "scripts/download_model.py"),
        "campaign": ("campaign-state.json", "scripts/campaign.py"),
        "server": ("server.json", "vendor/colibri/c/coli")}[target]
    state = json.loads((ROOT / "runtime" / state_name).read_text())
    pid = state["pid"]
    try:
        proc = Path(f"/proc/{pid}")
        args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = (proc / "cwd").resolve()
    except OSError:
        print("Process is no longer running")
        return
    valid_marker = marker in args or (target == 'campaign' and any(name in args for name in
        ('Tunning-Enhance-1/scripts/run_study.py', 'Tunning-Enhance-1/scripts/check_numerics.py')))
    if cwd != ROOT or not valid_marker:
        raise SystemExit("Process identity does not match this workspace; no signal sent")
    if target == "server":
        if os.getpgid(pid) != pid:
            raise SystemExit("Server has no independently owned process group; no signal sent")
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to owned {target} process")


if __name__ == "__main__":
    main()
