# Method and predeclared decision

[Overview](README.md) · [Results](RESULTS.md) · [Reproduce](REPRODUCE.md)

Only the CUDA expert-tier budget changes: `COLI_VK_TIER_GB=16` versus `32`.
Both profiles use native int4 experts, runtime dense bits 4, a 32 decimal GB
RAM expert cache, `COLI_CUDA_TC_GEMM=0`, buffered reads (`G53_DIRECT=0`), no extra
host tier, 14 CPU threads, one 8,192-token slot, low effort, and temperature 0.
The pinned engine binary and checkpoint are reused, not rebuilt or requantized
between profiles. Inputs are copied into [configs](configs/study.json) and
[workloads](workloads/smoke.jsonl); hashes are recorded per attempt.

## Sequence and unit of repetition

One CPU control, then **A B / B A / A B**, where A is 16 GiB and B is 32 GiB.
Each is a fresh server start. The two original short prompts are repeated twice
at concurrency one and a 64-token completion cap: four requests per start,
28 HTTP requests overall, plus 14 separate arithmetic/sorting answer checks.
The four requests in a start are correlated. The three starts per CUDA profile
are the independent repetitions. The CPU reference has one start and does not
support a repeated CPU comparison claim.

Here, a fresh server is the repetition unit. This does not establish statistical
independence: OS caches, thermal state, and user-service activity persist across
starts. Reported improvements are descriptive observations on this host.

The requested model seed is not invented; temperature-zero sampling is explicit.
No OS cache eviction is forced. Earlier validation, file reads, user services,
and prior runs may warm page caches. Pair order alternates to reduce a single
fixed-order bias; three pairs are still a small descriptive sample.

## Metrics

Batch completion tok/s = reported completion tokens / whole measured HTTP
batch duration, including prefill and overhead. **TTFT proxy** = time to first
nonempty streamed output, potentially reasoning; exact token-event timing,
inter-token latency, and steady decode throughput are not measured.

Each profile's headline is the median of its server-start values. TTFT uses
each server's upstream-reported p50, then the median across starts. The saved
per-request observations remain in `results/summary.json`; this does not pool
12 requests and call them 12 independent trials.

For each adjacent A/B pair, throughput gain is `B/A - 1`; TTFT reduction is
`1 - B/A`. The decision uses medians of these paired changes, not historical
results from an earlier session. A considerable gain is at least 20% in one
metric, an improvement in all three pairs for that metric, and no more than 5%
median regression in the other. This is a practical target, not a significance
test or a promised result.

All completed tuning results are retained and compared with the historical
baseline, even below this target. The criterion controls a considerable-gain
claim/new performance version, not whether an experiment is documented.

## Correctness and resource gates

The 46-case GPU suite, geometry/integrity validation, sampled independent expert
arithmetic, and regression checks run again. Both known-answer requests must
pass per server; non-finite router output invalidates an attempt. Additional
raw short/long prompts must have finite dumped logits and the expected first
continuation (`Paris`/`Lisbon`) on CPU and both CUDA budgets. Absolute logit
differences are reported without claiming whole-model numerical equivalence.
The previously broken default tensor-core path remains off.

Startup logs must report the requested CUDA expert capacity without clamping,
and show actual routed-expert GPU work. Global available memory is monitored;
the server stops below 8 GiB on three consecutive samples. CPU/GPU share one
physical pool. RSS sums can count shared pages twice; phase timers can overlap;
startup capacities are rounded budgets, not observed allocated-byte counts.
Existing running and stopped containers must retain their identities, mounts,
states, and start times. No power mode, clock, filesystem, swap, or user service
is changed. Failures remain evidence and are never silently replaced.

## Demonstration

The new GIF repeats the original Rust prompt with the 32 GiB helper and a
512-token output cap. It is a separate single coding demonstration, not the
source of the paired speedup claim. Actual terminal events are uniformly
accelerated to 27 seconds followed by a 3-second hold. Only the ready prompt
through the answer/statistics are shown; loading and exit are excluded.
