#!/usr/bin/env python3
"""Capture reproducibility facts without usernames, hostnames, tokens, or containers."""
import datetime
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/environment.json')
    args = parser.parse_args()
    mem = {line.split()[0].rstrip(":"): int(line.split()[1])*1024
        for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[-1:] == ["kB"]}
    data = {"captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": platform.machine(), "kernel": platform.release(),
        "python": platform.python_version(), "logical_cpus_available": len(os.sched_getaffinity(0)),
        "memory_total_bytes": mem["MemTotal"], "memory_available_bytes_at_capture": mem["MemAvailable"],
        "swap_total_bytes": mem["SwapTotal"],
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]),
        "nvme": command(["lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TRAN", "/dev/nvme0n1"]),
        "power_profile": command(["nvpmodel", "-q"]),
        "compiler": command(["gcc", "--version"]).splitlines()[0],
        "nvcc": command(["/usr/local/cuda/bin/nvcc", "--version"]),
        "jetson_linux": Path("/etc/nv_tegra_release").read_text().strip(),
        "scope": "One unified CPU/GPU memory pool; operating point is not a generic Thor performance claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
