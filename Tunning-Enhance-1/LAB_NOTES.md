# Study notebook

- Created `Tunning-Enhance-1/` and branch `tuning/enhance-1` before candidate
  measurements. Preserved the parent baseline and its real terminal recording.
- Fixed the change to 16 versus 32 GiB CUDA expert tiers with a constant
  32 decimal GB RAM expert cache and `COLI_CUDA_TC_GEMM=0`.
- Declared three alternated pairs, one CPU reference, and the considerable-gain
  rule in `configs/study.json` before inspecting candidate results.
- Extended existing benchmark/validation tools with explicit study/output
  destinations; defaults continue to target the original study.
- Started the isolated campaign on 2026-11-26 UTC. Recompiled and reran the
  GLM-specific CUDA correctness executable on Thor rather than reusing its old
  pass receipt. Native expert checks and the original regression suite repeat
  before the first performance run.

Measurements, additional observations, and the final promotion decision are
recorded in [results](RESULTS.md) and `results/summary.json`.

- First matched pair: 16 GiB 0.4762 tok/s and 30.54 s first-output p50;
  32 GiB 0.5227 tok/s and 27.71 s. Both passed 2/2 checks. The initial gains
  (+9.8% throughput, 9.3% latency reduction) were below the predeclared target;
  all remaining pairs were retained in the plan.
- The first 32 GiB start took 59.72 s versus 42.11 s for its control. Added a
  startup-cost chart to expose this trade-off; the decision rule was unchanged.
- User clarified that old and new benchmarks should both remain visible, even
  when numbers are close. Publication will append a clearly labeled tuning
  history/comparison and retain every original result, chart, and GIF. The 20%
  rule still controls whether this merits a new performance version.
- All seven starts completed at 2026-11-26 07:36 UTC, with all 14 strict answer
  checks passing and protected containers unchanged. Three paired throughput
  gains were approximately 9.77%, 13.26%, and 9.83%; first-output reductions
  were 9.28%, 8.76%, and 11.88%.
- The queued post-check process had exited with code 143 while waiting (cause
  unconfirmed). Restarted only the post-check/GIF pipeline at 07:39 UTC;
  no benchmark measurement was rerun or replaced. Added a waiting heartbeat
  and retry of a partially written state snapshot for clearer future progress.
- All six raw-logit probes passed with finite values and expected Paris/Lisbon
  continuations. Maximum CPU/CUDA absolute logit differences were about 0.0102;
  this is a limited numerical check, not whole-model equivalence.
- New real 32 GiB Rust capture: 81 tokens, 210.26 seconds from ready prompt to
  statistics, replayed at 7.79x into a 30-second GIF. The original 16 GiB demo
  took 212.15 seconds: this single coding example shows little timing change,
  unlike the modest benefit in the repeated HTTP workload.
