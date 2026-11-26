# Tunning-Enhance-1: 32 GiB GPU expert tier

Does increasing Colibri's CUDA expert tier from **16 to 32 GiB** improve native
GLM-5.3-Flash inference on Jetson Thor? The RAM expert cache stays at **32 decimal
GB**, and `COLI_CUDA_TC_GEMM=0` remains enabled as the validated workaround.

[Results](RESULTS.md) · [Method](METHODOLOGY.md) · [Reproduce](REPRODUCE.md) · [All study files](INDEX.md) · [Original study](../README.md)

![Independent-start comparison of throughput and first-output latency](assets/performance.png)

<!-- STUDY STATUS START -->

**Status:** 3 recorded attempt(s), 1/3 complete pairs; **pending_or_failed_checks**.

| Configuration | Independent starts | Median batch completion tok/s | Median server TTFT-proxy p50 | Answer checks |
|---|---:|---:|---:|---:|
| cpu-32 | 1 | 0.372 | 46.51 s | 2/2 |
| cuda-conservative-16-32 | 1 | 0.476 | 30.54 s | 2/2 |
| cuda-conservative-32-32 | 1 | 0.523 | 27.71 s | 2/2 |

<!-- STUDY STATUS END -->

## What is repeated?

- Native checkpoint inventory/geometry, two independent NumPy/C expert checks,
  the 46-case CUDA suite, and regression tests.
- One fresh CPU reference and three matched pairs of conservative 16/32 GiB
  CUDA starts. Four HTTP requests and two known-answer checks per start.
- Matched short/long raw-logit finiteness probes for CPU and both CUDA budgets.
- A new real Rust Fibonacci terminal recording using the 32 GiB configuration.

<img src="assets/study-workflow.png" alt="Repeat validation, alternate 16 and 32 GiB runs, check correctness and memory, then promote only if the fixed considerable-gain criterion is met." width="760">

## Repeatability and cost

![Per-pair improvements and the predeclared 20 percent target](assets/paired-gains.png)

![Separate reported phase timers and measured process reads](assets/phases-and-io.png)

![Reported tier capacity and global memory headroom](assets/placement-and-memory.png)

Each dot represents a fresh server start, not an independent request inside
one evolving cache. CPU/GPU allocations share physical RAM. Read the
[definitions and caveats](METHODOLOGY.md) before comparing systems.

## Startup trade-off

![Model startup and approximate startup plus first-request output](assets/startup-cost.png)

The right panel adds model/server startup to the first request's first-output
latency. It omits validation and harness gaps. More GPU residency can speed up
a loaded session while increasing the wait when starting one.

## New 30-second Rust demo

<!-- DEMO START -->
The new real 32 GiB recording is queued after the paired benchmark and numerical
checks. It will repeat the original Rust Fibonacci prompt and use an accelerated
elapsed-time clock, with exactly 30 seconds of playback.
<!-- DEMO END -->

## Promotion rule

The criterion was fixed before observing results: at least **20% median paired
throughput improvement or TTFT-proxy reduction**, improving in all three pairs
for that metric, with no more than **5% median paired regression** in the other
metric. Every quality, finite-output, placement, memory, and container check must
pass. A passing outcome can be promoted to the parent README and a new version;
a negative outcome remains documented in this folder.
