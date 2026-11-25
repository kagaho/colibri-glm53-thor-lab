# Results

These are initial measurements on one Thor. Failed attempts remain in the dataset.

The build and GLM CUDA correctness evidence is in [build-validation.json](../results/build-validation.json).

| Run | Status | Profile | HTTP completion tok/s | First-output p50 (s) | Disk reads (GB) |
|---|---|---|---:|---:|---:|
| 20261125T091731Z-cuda-16-32 | invalid_output | cuda-16-32 | unavailable | 4.278 | 3.969 |
| 20261125T154401Z-cpu-32 | starting | cpu-32 | unavailable | unavailable | unavailable |

Invalid-output timings are excluded from the headline throughput column; raw diagnostic rates remain in JSON.

HTTP throughput includes the whole measured batch. It must not be compared directly with steady-decode benchmarks.

See [summary.json](../results/summary.json) for per-attempt metrics, memory, integrity checks, and limitations.

## Limits

- One machine and two short synthetic prompts; not a model intelligence benchmark
- I/O policy is recorded per attempt; the initial native-default direct-I/O run was mislabeled in its original metadata and is corrected in the curated summary
- Initial profile order is CUDA then CPU; OS cache state and drift can confound comparisons
- CPU and GPU profiles differ in device expert capacity as well as compute; no isolated GPU-kernel speedup claim
- Quality checks are two strict known-answer smoke tests, not a general quality score
