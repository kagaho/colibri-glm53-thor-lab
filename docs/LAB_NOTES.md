# Lab notebook

Dates below are UTC; the study machine uses America/Los_Angeles. The local date
crossed midnight while the initial audit and setup were in progress.

## 2026-11-24 to 2026-11-25: identify the intended experiment

1. Read the locally supplied Jetson Thor guide, including page 17 and the setup
   and engine-comparison sections. Its proposed 109 GB GGUF fits a different
   experiment from the native container: holding the model in shared memory.
2. Inspected hardware, memory, root filesystem, GPU, CUDA toolkit, and Docker
   mounts. The host has approximately 122 GiB usable shared memory, no swap, and
   one 1 TB WD PC SN5000S NVMe. Recorded CUDA 13.2.86, driver 595.113, Jetson Linux
   R39.2.2, and an unchanged 120W power mode.
3. Confirmed that the selected native checkpoint can be staged on the available
   NVMe. Kept the study focused on inference, memory placement, and measured
   model behavior; unrelated machine administration is outside this repository.
4. Researched current project documentation and first-person measurements.
   Historical reports of missing Colibri GLM CUDA support are obsolete for
   v2.2.0. No measured current Colibri/Thor throughput was assumed.
5. The user explicitly selected the **native int4 streaming experiment**, with
   benchmarks and a portfolio study. The scope is Colibri's native checkpoint;
   a resident GGUF is not used as a replacement.

## 2026-11-25: pin source and model

6. Cloned Colibri's `v2.2.0` tag into `vendor/colibri`. The checkout is detached
   at `e479104ae01bdc40ce81e675299b1d230891b321`; no upstream source edits were
   needed for the initial build. The upstream checkout is excluded from Git.
7. Read `docs/AI_SETUP.md`, `docs/glm53-flash.md`, `docs/benchmarking.md`, and the
   actual Makefile, downloader, launcher, engine, and tier implementation.
   Some prose still describes older backend behavior, so the pinned source and
   observed execution are the final checks for this experiment.
8. Queried the native model's public metadata. Selected immutable revision
   `fe1d408f252c9c53a0d688aa54e41d9bdfdfd510`: 71 files in total, 62 weight shards,
   194,747,643,763 bytes. Chose the preconverted release to avoid spending many
   hours converting the original checkpoint. This study tests inference rather
   than claiming to reproduce the quantizer.
9. Created `configs/experiment.json` and the downloadable-file manifest. Wrote
   a thin wrapper around the pinned upstream `setup_download.download_file`.
   SHA-256 verifies large files; ordinary Git files use their blob IDs. Partial
   files keep `.part` suffixes and are renamed only after verification.
10. Started with two download workers, then resumed with four. Completed and
    partial files were reused. This changed transfer concurrency only; it did
    not change model weights or their revision. The downloader was later given
    a workspace lock to prevent accidental concurrent invocations.

## 2026-11-25: build and validate CUDA

11. Built only the required GLM engine:

    ```bash
    make -C vendor/colibri/c -j 4 glm53 CUDA=1 CUDA_ARCH=sm_110
    ```

    Build succeeded using the existing GCC and CUDA toolchain. No system
    package installation, toolkit replacement, or power-mode change was needed.
12. Inspected dynamic linkage and cubin inventory. The binary links CUDA runtime
    libraries and contains three `sm_110` cubins. Discovery alone is not proof
    that full-model inference uses those kernels.
13. Ran the architecture-relevant correctness gate:

    ```bash
    make -C vendor/colibri/c cuda-glm53c-check CUDA=1 CUDA_ARCH=sm_110
    python3 scripts/record_build.py
    ```

    **Passed: 46 cases.** The suite compares KDA, MLA, pooled sparse-indexer,
    DSA, and related GLM CUDA operations with CPU references. The record includes
    the exact binary SHA-256. This is kernel validation, not full-model answer
    quality or an end-to-end oracle comparison.

## Benchmark design prepared while weights download

14. Defined explicit CPU and CUDA profiles. Both use native int4 experts, a
    4-bit runtime dense trunk, an 8K context, a 32 GB RAM expert-cache budget,
    one conversation slot, greedy sampling, low reasoning effort, and 14 CPU
    threads. The initial CUDA tier is limited to 16 GiB.
15. Disabled the optional extra tier host pool and automatic tuning profiles.
    Disabled persistent usage saves and assigned a fresh per-run usage path.
    Prefix storage is disabled for the first baseline so a prompt-cache win is
    not silently presented as a storage or compute win.
16. Reused the pinned upstream HTTP benchmark rather than writing a new SSE
    parser. This preserves its explicit protocol, token-accounting, and failure
    semantics. Added process I/O and memory telemetry around it.
17. Added a native-file geometry validator, strict known-answer smoke checks,
    immutable result directories, public summary generation, and a memory guard.
    No generated code is executed. No systemwide page-cache flush is performed.
18. Prepared an unattended campaign that waits for the verified download before
    loading the model. It validates the files, runs CUDA then CPU, stops only
    servers that it created, and retains failed runs before generating results.
19. Ran seven focused lab tests: valid native geometry, incomplete download,
    wrong revision, missing shard, out-of-file offsets, wrong scale geometry,
    and failure-preserving public summary sanitization. All passed. Python and
    shell syntax checks also passed.
20. Used an already verified shard to test actual native expert arithmetic
    before the complete model downloaded. The upstream C matvec agreed with an
    independent NumPy nibble/gs64 decode for layer 12, expert 130's gate and down
    projections. First-five-output absolute errors were about 1.37e-6 and
    6.56e-7. This sampled container check is not a full-model quality claim.
    The exact tensors, hashes, thresholds, and output-sum errors are recorded in
    `results/native-expert-check.json`. NumPy 1.26.4 was already installed; no
    Python packages were added.
21. Initialized a local Git repository on `main`. No remote was configured or
    pushed. Confirmed that model files, upstream sources, raw logs, local PDF,
    and private machine-administration records are ignored.
22. Hardened process ownership and resuming: added per-workspace download and
    campaign locks, exact-binary checks, bounded shutdown, and a stop helper that
    checks process identity. Resumed the transfer after adding the downloader
    lock, allowing the upstream checksum checks to revalidate existing files.
23. Queued the bounded campaign in a detached process. It waits for the download
    completion marker before loading weights, so transfer I/O does not overlap
    measured inference. Progress and errors are recorded under ignored `runtime/`
    and `logs/` paths; curated results are generated separately.

## Initial campaign outcome

The original campaign subsequently completed its CUDA requests but produced
invalid output; its CPU stage stopped at a port preflight error. The full-model
isolation tests, metadata corrections, runner fixes, and resumed campaign are
recorded in [DIAGNOSTICS.md](DIAGNOSTICS.md). These findings supersede the
earlier expectation that the initial campaign would run through uninterrupted.

The resumed campaign finished on November 25. CPU and conservative CUDA both
passed arithmetic (`391`) and sorting (`[-2, 0, 4, 4, 9]`). Across four short
HTTP requests each, the CPU configuration delivered 0.415 completion tok/s and
conservative CUDA 0.517. First-output p50 was 44.472 s and 28.940 s respectively.
These include prefill and request overhead, not steady decode. The single
configuration comparison is descriptive; independent interleaved trials remain
future work. See [RESULTS.md](RESULTS.md) and the structured summary.
