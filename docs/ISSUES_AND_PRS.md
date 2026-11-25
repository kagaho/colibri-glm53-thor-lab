# Issues, fixes, and related upstream work

The study's issue tracker separates observed backend behavior, runner mistakes,
and follow-up instrumentation. No issue or PR has been submitted to the upstream
project on the user's behalf.

| Finding | Study issue | Resolution |
|---|---|---|
| Default CUDA produces non-finite full-model logits | [#4](https://github.com/kagaho/colibri-glm53-thor-lab/issues/4) | Workaround validated: `COLI_CUDA_TC_GEMM=0`; kernel root cause remains open |
| Port preflight rejects TIME_WAIT after an owned server closes | [#2](https://github.com/kagaho/colibri-glm53-thor-lab/issues/2) | Fixed with matching SO_REUSEADDR policy; live-listener and recently-closed-port regressions pass |
| Initial I/O metadata mislabeled and protocol success confused with usable inference | [#3](https://github.com/kagaho/colibri-glm53-thor-lab/issues/3) | Historical correction, explicit `G53_DIRECT=0` for new runs, numeric/quality gates, and retained failed evidence |
| Requested GPU tier differs from engine-reported capacity | [#1](https://github.com/kagaho/colibri-glm53-thor-lab/issues/1) | Follow-up PR records observed budget/capacity, retains the evidence, and backfills existing runs |

## How the fixes were validated

The port and quality fixes are in the published commit history. Regression
tests reject a live listener, accept an owned recently closed port, and classify
non-finite output as invalid even when all HTTP requests succeeded.

For CUDA, the same 33-token raw prompt produced 5,265,920 non-finite logits under
default math and zero with conservative GEMMs; the latter continued `Lisbon`.
CPU and conservative CUDA then passed both HTTP known-answer checks. This
establishes a useful workaround on the pinned operating point, not an upstream
kernel repair. See [DIAGNOSTICS.md](DIAGNOSTICS.md).

The placement follow-up reads the actual startup line rather than assuming a
16 GiB request was honored. The initial failed CUDA run reported 1.54 GiB and
56 expert slots; the valid conservative CUDA run reported 16.00 GiB and 1088
slots. These are rounded startup capacities, not measured allocated bytes.
Why the earlier budget was clamped remains a separate unresolved question.

## Related upstream PRs and issues

- [Colibri PR #2675: GLM Flash CUDA chain](https://github.com/JustVugg/colibri/pull/2675)
  implements the KDA, MLA, and pooled-indexer chain used by this pinned engine.
  It is context for the reproduction, not a proposed fix for this study's bug.
- [Colibri issue #2355: Vulkan first-layer NaNs](https://github.com/JustVugg/colibri/issues/2355)
  reports similar `!` output and a non-finite router warning on Windows/Vulkan.
  Its synchronization investigation does not establish the same cause on CUDA.
- [Colibri discussion #1567](https://github.com/JustVugg/colibri/discussions/1567)
  describes historical Spark behavior before GLM Flash's CUDA support. It is
  not evidence that v2.2.0 lacks that backend.
- [Colibri issue #2608](https://github.com/JustVugg/colibri/issues/2608) reports
  low GLM Flash throughput on a different V100/GGUF setup. Compare its model
  format, backend, memory, and version before drawing parallels.
- [llama.cpp PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773) and
  [MTP PR #29928](https://github.com/ggml-org/llama.cpp/pull/29928) establish
  mainline support for a future GGUF comparison. They do not add support for
  this native Colibri container and are not part of the current benchmark.
