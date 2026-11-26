#!/usr/bin/env python3
"""Start a bounded interactive chat with the validated native model settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "Tunning-Enhance-1"
sys.path.insert(0, str(ROOT / 'scripts'))


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def recorded(filename):
    path = ROOT / "runtime" / filename
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take-over", action="store_true", help="stop this lab's running campaign before chatting")
    parser.add_argument("--cpu", action="store_true", help="use the CPU instead of conservative CUDA")
    parser.add_argument("--max-tokens", type=int, default=256, help="maximum output tokens, including reasoning (default 256)")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192:
        parser.error("max-tokens must be between 1 and 8192")
    os.chdir(ROOT)
    campaign = recorded("campaign-state.json")
    if campaign.get("pid") and alive(campaign["pid"]):
        if not args.take_over:
            raise SystemExit("The benchmark campaign is running. Wait for it, or pass --take-over to stop it before chatting.")
        subprocess.run([sys.executable, str(ROOT / "scripts/stop.py"), "campaign"], check=True)
        deadline = time.monotonic() + 60
        while alive(campaign["pid"]) or alive(recorded("server.json").get("pid", -1)):
            if time.monotonic() > deadline:
                raise SystemExit("The previous lab server has not stopped. No second model was loaded.")
            time.sleep(.25)
    server = recorded("server.json")
    if server.get("pid") and alive(server["pid"]):
        raise SystemExit("A recorded lab server is still running. Stop it with python3 scripts/stop.py server first.")
    from validate_model import validate
    from run_experiment import meminfo
    validate()
    if meminfo()["MemAvailable"] < 80 * 2**30:
        raise SystemExit("Less than 80 GiB RAM is available. No new model was loaded.")
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "COLIBRI_", "GLM53_", "G53_", "CUDA_", "OMP_"))
        and k not in ("RAM_GB", "CAP", "PIN", "PIN_GB", "SNAP", "SEED", "SERVE", "SERVE_BATCH", "KV_SLOTS")}
    env.update(COLI_CUDA="0" if args.cpu else "1", COLI_VULKAN="0", COLI_CUDA_TC_GEMM="0",
        GLM53_BITS="4", GLM53_EXPERT_GB="32", COLI_VK_TIER_GB="32", COLI_VK_TIER_HOST="0",
        COLI_PREFIX_STORE_MB="512", G53_DIRECT="0", OMP_NUM_THREADS="14", USAGE_SAVE="0",
        COLI_USAGE=str(STUDY / "runtime/interactive-usage"))
    command = [sys.executable, str(ROOT / "vendor/colibri/c/coli"), "chat",
        "--model", str(ROOT / cfg["model_directory"]), "--gpu", "none" if args.cpu else "0",
        "--ctx", str(cfg["context"]), "--no-tune-profile", "--no-attach", "--effort", "low",
        "--ngen", str(args.max_tokens), "--temp", "0", "--stats", "full"]
    print("Starting native int4 chat on " + ("CPU" if args.cpu else "conservative CUDA") + ". Exit with Ctrl+C.", flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
