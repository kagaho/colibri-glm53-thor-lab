# Benchmark method

## Question

Can Colibri use a single Thor's shared RAM, GPU, and NVMe to serve the native
GLM-5.3-Flash int4 checkpoint, and what limits its latency and throughput?

The initial study compares two whole configurations. CPU and CUDA share the
same model, source revision, runtime dense precision, RAM expert-cache budget,
context, concurrency, prompts, requested output cap, seed, reasoning setting,
and thread count. CUDA adds an explicitly bounded device expert tier and a
different compute path. Additional residency is part of that configuration.

## Initial workload and settings

- Two short synthetic prompts in `workloads/smoke.jsonl`, each repeated twice.
- One concurrent request, 64 requested completion tokens, greedy temperature 0.
- Low reasoning effort. Reasoning tokens consume the same generated-token
  budget; first reasoning output may precede visible answer content.
- One 8,192-token conversation slot. No automatic tuning profile.
- Native group-scaled int4 experts; runtime dense precision fixed to 4 bits.
- 32 decimal GB RAM expert-cache budget; 0 or 16 GiB device expert-tier budget.
- Prefix store disabled and persistent usage saves disabled for the baseline.
- Fresh engine, recurrent state, and expert LRU for each profile. The model's
  OS page-cache state is uncontrolled and can be warm after downloading.

Disabling the content-addressed prefix store does not disable every form of
slot continuation or prefix reuse. Repeated requests can still benefit from
the gateway/engine's retained state. Inspect actual prompt-token/forward counts
and keep this behavior identical across profiles; do not label the requests
as uncached prefills.

The profile order is CUDA then CPU. This is a first functional study, not a
balanced performance trial. Follow-up comparisons must interleave profile
order and collect more independent trials. Requests within one server are
correlated by its evolving cache and must not be called independent runs.

## Measurements

1. Upstream HTTP benchmark: first nonempty output latency, per-request duration,
   finish reason, completion tokens reported by the server, protocol success,
   and completion-token throughput over the complete measured batch.
2. Engine `/profile`: whole-turn time, prompt and completion counts, expert
   disk/wait/matmul time, attention time, head time, and forward count.
3. Process group `/proc` counters: physical bytes read and logical read calls
   during the HTTP stage. Counters cover launcher, gateway, and engine rather
   than claiming to isolate expert reads from all other process reads.
   CUDA tier log lines additionally provide cumulative device expert counts,
   CPU RAM hits, and disk loads at each turn. Their denominators differ; an
   aggregate percent is not invented when a backend does not expose one.
4. One-second telemetry: global available memory, swap fields, and summed RSS
   of the owned process group. Shared pages can be counted more than once by RSS.
5. Startup/load duration, actual GPU expert-execution diagnostics, and existing
   container identity/state/mount checks before and after the experiment.

HTTP throughput includes prompt processing and overhead. It is **not** steady
decode tok/s. First nonempty streamed output is **not** guaranteed to be time to
a visible answer or exactly one token. Missing usage is not reconstructed from
words or characters. Failed requests remain in the dataset.

## Controls and quality

The build must pass the GLM-specific CUDA suite on the actual GPU. Checkpoint
integrity is verified on download and its headers/geometry validated before
loading. Two strict known-answer requests test arithmetic and sorting without
executing model-generated code. Their exact-match failures are retained,
including failures caused by formatting, truncated reasoning, or token limits.

These controls do not constitute a broad capability evaluation. Kernel parity
does not establish full-model numerical parity. Two smoke questions do not
establish native int4 quality relative to the original checkpoint.

## Storage interpretation

The baseline uses Colibri's default buffered I/O. Do not call its first request
a cold-SSD measurement: the download itself and earlier profiles can warm file
pages. No `drop_caches`, swap change, disk formatting, storage benchmark writes,
or service interruption is part of this study. Concurrent user-service disk
traffic remains a possible confound.

An expert-cache hit percentage is engine-specific and does not independently
prove reduced physical NVMe reads. Interpret it together with `/proc` I/O,
engine disk time, tier placement, and full request latency. Direct-I/O and
carefully scoped cold-file tests are separate follow-ups.

## Exclusion and failure policy

Stop and retain logs if the checksum fails, files are incomplete, the CUDA
correctness gate fails, loading fails, the memory guard triggers, GPU work is
absent from a requested CUDA run, or HTTP protocol validation fails. Never
silently rerun a failed attempt until only a good number remains.
