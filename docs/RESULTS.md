# Results

Full-model measurements are pending.

The build and GLM CUDA correctness evidence is in [build-validation.json](../results/build-validation.json).

| Run | Status | Profile | HTTP completion tok/s | First-output p50 (s) | Disk reads (GB) |
|---|---|---|---:|---:|---:|

HTTP throughput includes the whole measured batch. It must not be compared directly with steady-decode benchmarks.

See [summary.json](../results/summary.json) for per-attempt metrics, memory, integrity checks, and limitations.

## Limits

- One machine and two short synthetic prompts; not a model intelligence benchmark
- Buffered I/O and an uncontrolled OS page cache; first process is not a verified cold SSD run
- Initial profile order is CUDA then CPU; OS cache state and drift can confound comparisons
- CPU and GPU profiles differ in device expert capacity as well as compute; no isolated GPU-kernel speedup claim
- Quality checks are two strict known-answer smoke tests, not a general quality score
