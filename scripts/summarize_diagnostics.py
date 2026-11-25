#!/usr/bin/env python3
"""Curate full-model numeric diagnostics without publishing large logit dumps."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "results/raw/diagnostics-20261125"
rows = []
for name in ("cpu-bits4", "cuda-exact-bits4", "cuda-default-long", "cuda-exact-long"):
    values = np.fromfile(raw / (name + ".f32"), dtype=np.float32)
    text = (raw / (name + ".log")).read_text(errors="replace")
    after = text.split("teacher_forcing", 1)[1].split("\n", 1)[1]
    answer = after.split("[CUDA]", 1)[0].split("\ndecode", 1)[0].strip()
    rows.append({"case": name, "logit_values": len(values),
        "nonfinite_logits": int((~np.isfinite(values)).sum()),
        "decoded_continuation": answer,
        "nonfinite_router_warning": "[router] non-finite logits" in text})
cpu = np.fromfile(raw / "cpu-bits4.f32", dtype=np.float32)
cuda = np.fromfile(raw / "cuda-exact-bits4.f32", dtype=np.float32)
result = {"engine_commit": json.loads((ROOT / "configs/experiment.json").read_text())["engine_commit"],
    "model_revision": json.loads((ROOT / "configs/experiment.json").read_text())["model_revision"],
    "hardware": "Jetson Thor sm_110, CUDA 13.2.86, driver 595.113",
    "cases": rows,
    "short_prompt_cpu_cuda_max_absolute_logit_difference": float(np.max(np.abs(cpu-cuda))),
    "short_prompt_max_cpu_absolute_logit": float(np.max(np.abs(cpu))),
    "finding": "Same 33-token raw prompt: default CUDA produced non-finite logits; disabling tensor-core GEMMs produced finite logits and Lisbon",
    "workaround": "COLI_CUDA_TC_GEMM=0",
    "limits": "One model/build/device and two prompts. This isolates a problematic execution path; the exact kernel defect is not identified. Diagnostics are not controlled throughput benchmarks."}
(ROOT / "results/numerical-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
