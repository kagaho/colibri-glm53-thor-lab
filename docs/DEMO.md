# Real Rust Fibonacci terminal demos

[README demo](../README.md#watch-glm-write-rust-in-30-seconds) · [Interactive guide](INTERACTIVE.md) · [All files](INDEX.md)

Two configurations are recorded: a **16 GiB GPU expert tier** in the original
study and a **32 GiB GPU expert tier** in the follow-up. Both use the same
32 decimal GB RAM cache, conservative CUDA settings, and Rust Fibonacci question.

| Configuration | Real session duration | Playback | Source and timing |
|---|---:|---:|---|
| 16 GiB GPU expert tier | 212.2 s | 30 s | [Original metadata](../results/rust-fibonacci-demo.json) |
| 32 GiB GPU expert tier | 210.3 s | 30 s | [Follow-up demo](../Tunning-Enhance-1/README.md#32-gib-gpu-expert-tier--30-second-rust-demo) · [Metadata](../Tunning-Enhance-1/results/rust-fibonacci-demo.json) |

These are separate single chat sessions, not the source of the repeated HTTP
benchmark gains. Their waiting times are essentially unchanged. The instructions
below describe the original **16 GiB** recording; the
[follow-up reproduction guide](../Tunning-Enhance-1/REPRODUCE.md) covers 32 GiB.

The [30-second GIF](assets/rust-fibonacci-30s.gif) replays actual timestamped
input/output from `scripts/chat.py`, asking GLM-5.3-Flash to write Rust code.
The question was supplied automatically to a real pseudo-terminal (PTY); the
model response was not typed, substituted, or edited into the recording.

## What the clock means

This session produced **81 completion tokens in about 212 seconds**; Colibri's
chat footer reported **0.38 tok/s**. Its 212.15-second terminal interval is
replayed at **7.86×**. This includes the initial wait for model output.

The first 27 seconds replay the ready-prompt-to-completion interval at one
constant acceleration factor. The final answer is held for 3 seconds, making
the complete GIF exactly **30,000 ms**. The top-right `T+mm:ss.s` clock advances
in source-session time and freezes during the final hold; the adjacent `x`
value is the acceleration factor. The model was preloaded. Loading and exit
logs are retained locally and excluded from this focused demonstration.

See [the metadata](../results/rust-fibonacci-demo.json) for the actual source
duration, speed factor, UTC start, command, dependency versions, and source/GIF
hashes. The [cast](../results/rust-fibonacci.cast) retains original terminal
events and their relative times. A replay theme lifts very dark foreground
grays for contrast; terminal text and event order remain unchanged.

## Exact question

```text
Write only a concise Rust program: iterative fibonacci(n: u32) -> u64, F(0)=0 and F(1)=1, assume n<=92, and main printing fibonacci(10). No explanation.
```

The expected printed value is `55`. This is a coding demonstration, separate
from the four-request CPU/CUDA benchmark. The [final terminal screen](../results/rust-fibonacci-output.txt)
contains the model's response and the chat tool's own timing/token footer.
That footer is not substituted for the benchmark's HTTP throughput or TTFT
proxy. Do not treat this one response as a general Rust capability evaluation.
The generated loop was inspected: it follows the iterative Fibonacci recurrence
for the requested bounded inputs. It was not compiled or executed; no Rust
compiler was available on the recording host.

## Rebuild the GIF without running the model

The committed cast is sufficient; no GPU or checkpoint download is needed.
Use Python, a DejaVu Sans Mono font, and the optional replay dependencies:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-demo.txt
.venv-docs/bin/python scripts/render_demo.py
```

The renderer checks the exported animation duration and refreshes the output
screen and metadata. The recording and renderer are separate: replaying a
saved cast never generates another model answer. The committed GIF can be
viewed directly without installing dependencies.

## Record a new real session

First [build the engine and stage the pinned native weights](REPRODUCE.md).
Wait for any lab campaign/chat to finish; this recorder does not take over a
running service. It verifies that port 8000 is available and uses the existing
chat helper's memory guard and conservative CUDA settings.

```bash
python3 scripts/record_demo.py
```

The recorder writes `runtime/rust-fibonacci-demo/session.cast`, `terminal.txt`,
and `state.json`. That directory is ignored by Git. It refuses to overwrite an
existing capture. Preserve the old directory before recording another attempt.
Inspect progress with:

```bash
cat runtime/rust-fibonacci-demo/state.json
```

After reviewing the completed local recording, publish only its focused
ready-prompt-to-answer interval and rebuild the GIF:

```bash
.venv-docs/bin/python scripts/render_demo.py --curate
python3 scripts/render_docs.py
```

`--curate` rejects failed/truncated responses and private-looking output rather
than silently altering it. Inspect the final frames, source cast, and public
metadata before committing. No generated code is executed by these scripts.
