#!/usr/bin/env python3
"""Record actual build identity only after the matching correctness log passes."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=ROOT/'logs/cuda-glm53c-check.log')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(ROOT / "vendor/colibri"), "rev-parse", "HEAD"], text=True).strip()
    if commit != cfg["engine_commit"]:
        raise RuntimeError("Unexpected engine revision")
    log = args.log.read_text()
    if "\nPASS\n" not in log or "FAIL" in log:
        raise RuntimeError("The GLM-specific CUDA check did not pass")
    executable = ROOT / "vendor/colibri/c/glm53"
    cubins = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(executable)], text=True)
    if ".sm_110.cubin" not in cubins:
        raise RuntimeError("Engine has no Thor sm_110 code")
    result = {"status": "passed", "engine_commit": commit, "cuda_arch": "sm_110",
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "successful_kernel_cases": sum(line.strip().startswith("ok ") for line in log.splitlines()),
        "check_command": "make -C vendor/colibri/c cuda-glm53c-check CUDA=1 CUDA_ARCH=sm_110",
        "build_command": "make -C vendor/colibri/c -j 4 glm53 CUDA=1 CUDA_ARCH=sm_110",
        "cuda_toolkit": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
        "cubin_inventory": cubins,
        "scope": "GLM CUDA layer kernels against independent CPU references; not a full-model quality benchmark"}
    (ROOT / "runtime/cuda-check-passed.json").write_text(json.dumps(result, indent=2) + "\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    public_log = "\n".join(line.rstrip() for line in log.replace(str(ROOT), "<REPO>").splitlines()) + "\n"
    (args.output_dir / "cuda-glm53c-check.txt").write_text(public_log)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cuda_toolkit", "cubin_inventory")}, indent=2))


if __name__ == "__main__":
    main()
