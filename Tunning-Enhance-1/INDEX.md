# Tunning-Enhance-1 file index

All 53 published study files. [Overview](README.md) · [Results](RESULTS.md) · [Parent repository](../README.md)

| File | Purpose |
|---|---|
| [INDEX.md](INDEX.md) | Every published file in this tuning folder. |
| [LAB_NOTES.md](LAB_NOTES.md) | Recorded actions, observations, failures and decisions. |
| [METHODOLOGY.md](METHODOLOGY.md) | Fixed controls, paired ordering, definitions and promotion rule. |
| [README.md](README.md) | Study overview, findings, charts and new accelerated GIF. |
| [REPRODUCE.md](REPRODUCE.md) | Commands to run checks, benchmarks, numeric probes and demo. |
| [RESULTS.md](RESULTS.md) | Generated per-profile and paired results and decision. |
| [assets/figures.json](assets/figures.json) | Reproducible visualization. |
| [assets/paired-gains.png](assets/paired-gains.png) | Reproducible visualization. |
| [assets/paired-gains.svg](assets/paired-gains.svg) | Reproducible visualization. |
| [assets/performance.png](assets/performance.png) | Reproducible visualization. |
| [assets/performance.svg](assets/performance.svg) | Reproducible visualization. |
| [assets/phases-and-io.png](assets/phases-and-io.png) | Reproducible visualization. |
| [assets/phases-and-io.svg](assets/phases-and-io.svg) | Reproducible visualization. |
| [assets/placement-and-memory.png](assets/placement-and-memory.png) | Reproducible visualization. |
| [assets/placement-and-memory.svg](assets/placement-and-memory.svg) | Reproducible visualization. |
| [assets/rust-fibonacci-30s.gif](assets/rust-fibonacci-30s.gif) | Reproducible visualization. |
| [assets/startup-cost.png](assets/startup-cost.png) | Reproducible visualization. |
| [assets/startup-cost.svg](assets/startup-cost.svg) | Reproducible visualization. |
| [assets/study-workflow.png](assets/study-workflow.png) | Reproducible visualization. |
| [assets/study-workflow.svg](assets/study-workflow.svg) | Reproducible visualization. |
| [configs/baseline-reference.json](configs/baseline-reference.json) | Versioned experiment input or measured validation evidence. |
| [configs/experiment.json](configs/experiment.json) | Versioned experiment input or measured validation evidence. |
| [configs/model-manifest.json](configs/model-manifest.json) | Versioned experiment input or measured validation evidence. |
| [configs/profiles.json](configs/profiles.json) | CPU, conservative 16 GiB, and conservative 32 GiB configurations. |
| [configs/study.json](configs/study.json) | Predeclared order, repetition counts and considerable-gain criterion. |
| [results/baseline-regression-final.txt](results/baseline-regression-final.txt) | Versioned experiment input or measured validation evidence. |
| [results/build-validation.json](results/build-validation.json) | Versioned experiment input or measured validation evidence. |
| [results/cuda-glm53c-check.txt](results/cuda-glm53c-check.txt) | Versioned experiment input or measured validation evidence. |
| [results/environment.json](results/environment.json) | Versioned experiment input or measured validation evidence. |
| [results/historical-summary.json](results/historical-summary.json) | Versioned experiment input or measured validation evidence. |
| [results/model-validation.json](results/model-validation.json) | Versioned experiment input or measured validation evidence. |
| [results/native-expert-check.json](results/native-expert-check.json) | Versioned experiment input or measured validation evidence. |
| [results/numerical-diagnostics.json](results/numerical-diagnostics.json) | Repeated finite-logit and known-continuation probes on all three profiles. |
| [results/regression-validation.json](results/regression-validation.json) | Versioned experiment input or measured validation evidence. |
| [results/rust-fibonacci-demo.json](results/rust-fibonacci-demo.json) | Versioned experiment input or measured validation evidence. |
| [results/rust-fibonacci-output.txt](results/rust-fibonacci-output.txt) | Versioned experiment input or measured validation evidence. |
| [results/rust-fibonacci.cast](results/rust-fibonacci.cast) | Versioned experiment input or measured validation evidence. |
| [results/summary.json](results/summary.json) | Curated attempts, per-request metrics, paired deltas and decision gates. |
| [results/tuning-regression-final.txt](results/tuning-regression-final.txt) | Versioned experiment input or measured validation evidence. |
| [results/tuning-regression-tests.txt](results/tuning-regression-tests.txt) | Versioned experiment input or measured validation evidence. |
| [results/unit-tests.txt](results/unit-tests.txt) | Versioned experiment input or measured validation evidence. |
| [scripts/chat.py](scripts/chat.py) | Start a bounded interactive chat with the validated native model settings. |
| [scripts/check_numerics.py](scripts/check_numerics.py) | Repeat CPU/16/32 GiB full-model logit-finiteness and known-continuation probes. |
| [scripts/finalize_local.py](scripts/finalize_local.py) | Wait for all paired runs, then repeat numeric checks and record/render the new demo. |
| [scripts/prepare_promotion.py](scripts/prepare_promotion.py) | Add verified tuning history to the README while preserving the original benchmark. |
| [scripts/record_demo.py](scripts/record_demo.py) | Record the same real Rust prompt using this study's 32 GiB chat helper. |
| [scripts/render_demo.py](scripts/render_demo.py) | Render this study's genuine 32 GiB terminal capture as an exact 30-second GIF. |
| [scripts/render_results.py](scripts/render_results.py) | Render reproducible tuning comparisons and a compact study workflow. |
| [scripts/run_study.py](scripts/run_study.py) | Repeat validation, one CPU reference, and three alternating 16/32 GiB pairs. |
| [scripts/summarize.py](scripts/summarize.py) | Curate isolated attempts and assess the predeclared paired improvement rule. |
| [tests/test_study.py](tests/test_study.py) | Study tool. |
| [workloads/quality.json](workloads/quality.json) | Versioned experiment input or measured validation evidence. |
| [workloads/smoke.jsonl](workloads/smoke.jsonl) | Versioned experiment input or measured validation evidence. |

Ignored local evidence: `results/raw/`, `runtime/`, and `logs/`. Weights and engine source are reused from the parent workspace.
