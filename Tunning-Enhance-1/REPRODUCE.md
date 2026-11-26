# Reproduce Tunning-Enhance-1

Start in the parent repository and complete its [engine/model setup](../docs/REPRODUCE.md).
This study reuses the pinned engine and 195 GB checkpoint; it does not download
or duplicate them. The experiment needs the host free of other model sessions.
Full validation uses NumPy; plotting uses Matplotlib, and GIF replay uses
Pillow, pyte, and DejaVu Sans Mono. The study records the versions used on Thor.

```bash
python3 Tunning-Enhance-1/scripts/run_study.py
```

The driver repeats validation, one CPU start and the six planned CUDA starts.
It resumes only a matching completed prefix recorded in its ignored runtime
state. Failed attempts are retained. To inspect live progress:

```bash
cat Tunning-Enhance-1/runtime/campaign-state.json
python3 scripts/status.py
```

To queue the remaining numerical checks, recording, and visualizations behind
the benchmark, run this in a second terminal:

```bash
python3 Tunning-Enhance-1/scripts/finalize_local.py
```

Its progress is in `Tunning-Enhance-1/runtime/finalization-state.json`. The
manual steps below perform the same post-check work; use one path per capture.

The existing `python3 scripts/stop.py campaign` stops only this owned driver;
`python3 scripts/stop.py server` targets its recorded owned server. Existing
user containers and services are preserved.

After the benchmark driver finishes, repeat full-model numeric probes:

```bash
python3 Tunning-Enhance-1/scripts/check_numerics.py
python3 Tunning-Enhance-1/scripts/summarize.py
python3 Tunning-Enhance-1/scripts/render_results.py
```

The positive numeric probes use the safe math path; the historical broken
default-CUDA negative control remains documented in the original study.

## Record and render the new real GIF

Wait for all server/probe processes to exit. The study chat helper uses 32 GiB
for CUDA experts and otherwise matches the original helper. The capture wrapper
uses the original recorder with this study's working/output directory.

```bash
python3 Tunning-Enhance-1/scripts/record_demo.py
python3 Tunning-Enhance-1/scripts/render_demo.py --curate
```

The optional replay dependencies are in [requirements-demo.txt](../requirements-demo.txt).
Review the source terminal recording before publishing. A completed focused cast
can regenerate the GIF without the checkpoint:

```bash
python3 Tunning-Enhance-1/scripts/render_demo.py
```

The chart renderer needs [Matplotlib](../requirements-docs.txt). Graphs read only
the public `results/summary.json`. Stage new study files before rendering the
folder index; otherwise untracked files are not listed.

Once the final evidence and GIF are verified, prepare an additive comparison in
the parent README. The original benchmark and recording are retained:

```bash
python3 Tunning-Enhance-1/scripts/prepare_promotion.py
```

This command does not push Git commits or publish a release. A new performance
version remains conditional on the predeclared considerable-gain criterion.

## Boundaries

`results/raw/`, `runtime/`, and `logs/` are ignored local evidence, containing
full server logs, process details and numeric dumps. Public summaries, source
inputs, sanitized validation evidence, charts, focused cast, and GIF live in
this folder. The parent baseline results and baseline GIF are preserved.
