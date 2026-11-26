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
