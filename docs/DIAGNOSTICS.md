# First full-model findings and corrections

## Integrity and kernel tests were necessary but insufficient

All 71 files downloaded and passed their immutable-revision content checks.
Header validation found 62 shards, 75,922 tensors, and 37,152 native int4/scale
pairs. The GLM CUDA layer-op suite and sampled native expert arithmetic had
passed, but they did not cover the full model's problematic execution path.

The initial CUDA HTTP run completed four requests at the protocol level, then
failed both known-answer checks with repeated `!` output. The engine reported
`[router] non-finite logits at layer 3`. Its 0.878 completion tok/s is retained
as a diagnostic observation and excluded from useful-output performance.

## Matched numeric isolation

With the same model, build, and raw five-token prompt, CPU and conservative
CUDA both continued `The capital of France is` with `Paris`. Every dumped
logit was finite. Their maximum absolute logit difference was approximately
0.00990 against a maximum reference magnitude of 21.777; greedy text agreed.

A second pair used the same 33-token prompt, requesting the capital of Portugal
after several examples. Both CUDA runs used a 1 GiB expert device tier and an
8 GB RAM expert cache; the relevant changed setting was:

```bash
COLI_CUDA_TC_GEMM=0
```

| CUDA math setting | Dumped logit values | Non-finite values | First continuation |
|---|---:|---:|---|
| Default | 5,265,920 | 5,265,920 | `!` |
| Tensor-core GEMMs disabled | 5,265,920 | 0 | `Lisbon` |

This isolates a problematic execution path on this model/build/device. It does
not identify the exact faulty kernel or establish a universal CUDA defect.
The short diagnostic itself is not a throughput benchmark. Exact inputs and
counts are in [numerical-diagnostics.json](../results/numerical-diagnostics.json).

## Runner corrections

- **Port reuse:** the CPU stage's initial preflight used a plain socket bind.
  After the owned HTTP server closed, a TIME_WAIT socket could cause a false
  `Address already in use`. The probe now matches HTTPServer's SO_REUSEADDR
  policy. Tests cover a recently closed port and refusal of a live listener.
  No user service was killed to reclaim a port.
- **I/O label:** native `glm53.c` defaults `G53_DIRECT` to 1. The first runner
  incorrectly described this as buffered. Historical raw metadata is retained
  with an explicit correction in the curated summary. New runs explicitly set
  `G53_DIRECT=0`; this is a changed operating point, not an identical rerun.
- **Ignored controls:** the native engine warned that `SEED`, `KVSAVE`, and
  `PROF` were unused. Those settings were removed from this runner; greedy
  temperature-zero sampling remains explicit. The upstream launcher also adds
  an unused `CUDA_DENSE` setting, which is documented rather than treated as
  an active optimization.
- **Quality classification:** protocol completion alone no longer establishes
  a valid run. Non-finite router warnings stop the owned server. Failed strict
  answer checks mark a run failed and keep its timings for diagnosis.
- **Placement:** the first CUDA run requested a 16 GiB device expert tier, but
  the engine actually clamped it to approximately 1.54 GiB. Requested capacity
  must not be presented as observed placement. Driver memory accounting and
  shared-memory pressure require a separate follow-up; no cause is assumed.

## Continued study

The resumed campaign runs CPU first, then `cuda-conservative-16-32` with
tensor-core GEMMs disabled. Source, weights, and native int4 format remain
pinned. It records quality checks and actual execution diagnostics before any
performance result is treated as usable. See the live generated results page.
