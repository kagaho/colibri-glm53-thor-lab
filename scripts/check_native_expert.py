#!/usr/bin/env python3
"""Compare actual native int4 expert matvecs with an independent NumPy decode.

Can run once a complete expert shard is downloaded. It does not claim the whole
checkpoint is ready and does not measure inference performance.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    model = ROOT / cfg["model_directory"]
    text = json.loads((model / "config.json").read_text())["text_config"]
    subprocess.run(["gcc", "-O2", "-std=gnu11", "-I" + str(ROOT / "vendor/colibri/c"),
        "-o", str(ROOT / "runtime/check_glm53_container"),
        str(ROOT / "vendor/colibri/c/tools/check_glm53_container.c"), "-lm"], check=True)
    cases = []
    for shard in sorted(model.glob("*.safetensors")):
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= header_size <= 64*2**20:
                raise RuntimeError("Unreasonable safetensors header size")
            header = json.loads(stream.read(header_size))
            for name, spec in header.items():
                match = re.search(r"\.mlp\.experts\.\d+\.(gate_proj|down_proj)\.weight$", name)
                if not match or name + ".qs" not in header:
                    continue
                if any(case["projection"] == match[1] for case in cases):
                    continue
                rows, cols = ((text["hidden_size"], text["moe_intermediate_size"]) if match[1] == "down_proj"
                    else (text["moe_intermediate_size"], text["hidden_size"]))
                def read_tensor(item):
                    start, end = item["data_offsets"]
                    stream.seek(8 + header_size + start)
                    raw = stream.read(end-start)
                    if len(raw) != end-start:
                        raise RuntimeError("Truncated expert tensor")
                    return raw
                raw, scales_raw = read_tensor(spec), read_tensor(header[name + ".qs"])
                if spec["dtype"] != "U8" or len(raw) != rows*cols//2:
                    raise RuntimeError("Unexpected packed expert geometry")
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols//2)
                levels = np.empty((rows, cols), dtype=np.float32)
                levels[:, 0::2] = (packed & 15).astype(np.int16) - 8
                levels[:, 1::2] = (packed >> 4).astype(np.int16) - 8
                scales = np.frombuffer(scales_raw, dtype="<f4").reshape(rows, cols//64)
                weights = levels * scales.repeat(64, axis=1)
                x = (((np.arange(cols, dtype=np.int64)*37) % 199)-99).astype(np.float32)/np.float32(100)
                y = weights @ x
                output = subprocess.check_output([str(ROOT / "runtime/check_glm53_container"),
                    str(model), name, str(rows), str(cols)], text=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1"})
                first = re.search(r"y\[0\.\.4\] = (.*)", output)
                total = re.search(r"somma ([-+\deE.]+)", output)
                if not first or not total:
                    raise RuntimeError("Could not parse upstream matvec check output")
                actual = np.asarray([float(v) for v in first[1].split()])
                max_error = float(np.max(np.abs(actual-y[:5])))
                sum_error = abs(float(total[1])-float(y.sum(dtype=np.float64)))
                first_tolerance = 1e-5 + 1e-4*float(np.max(np.abs(y)))
                sum_tolerance = 1e-5 + 1e-4*float(np.abs(y).sum(dtype=np.float64))
                passed = bool(np.all(np.isfinite(y)) and max_error <= first_tolerance and sum_error <= sum_tolerance)
                cases.append({"tensor": name, "projection": match[1], "shard": shard.name,
                    "shape": [rows, cols], "packed_tensor_sha256": hashlib.sha256(raw).hexdigest(),
                    "first_five_max_abs_error": max_error, "first_five_tolerance": first_tolerance,
                    "output_sum_abs_error": sum_error, "output_sum_tolerance": sum_tolerance,
                    "passed": passed})
                if len(cases) == 2:
                    break
        if len(cases) == 2:
            break
    if len(cases) != 2:
        raise RuntimeError("A complete gate/down expert pair has not downloaded yet")
    result = {"status": "passed" if all(c["passed"] for c in cases) else "failed",
        "model_revision": cfg["model_revision"], "numpy_version": np.__version__,
        "method": "Independent nibble/gs64 decode and NumPy matvec vs upstream C matvec; compare first five rows and full-output sum",
        "scope": "Two actual expert projections; not a full-checkpoint oracle or a throughput benchmark", "cases": cases}
    (ROOT / "results/native-expert-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
