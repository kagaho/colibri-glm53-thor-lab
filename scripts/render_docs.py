#!/usr/bin/env python3
"""Render publication charts from committed results and index public repository files.

No model, GPU, private raw logs, or network access is needed. The figures show
observations from the saved study, not predicted performance.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/matplotlib"))

LABELS = {"cpu-32": "CPU", "cuda-conservative-16-32": "Conservative CUDA"}
COLORS = {"cpu-32": "#456d99", "cuda-conservative-16-32": "#008778"}
INK, MUTED, RED = "#172a3b", "#526674", "#b13d3d"


def checked_runs(summary):
    return [r for r in summary["runs"] if r["status"] == "complete"
        and not r.get("numerical_failure_detected") and r.get("quality_total", 0)
        and r.get("quality_passed") == r["quality_total"]]


def configure():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": INK, "axes.edgecolor": "#c9d5dd", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "colibri-thor-study",
        "savefig.facecolor": "white"})
    return matplotlib


def save(fig, name, description, source_hash):
    import matplotlib.pyplot as plt
    folder = ROOT / "docs/assets"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": description, "Description": "Source: results/summary.json, SHA256 " + source_hash,
                "Creator": "scripts/render_docs.py", "Date": None}
    svg = folder / (name + ".svg")
    fig.savefig(svg, metadata=metadata)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(folder / (name + ".png"), dpi=150, metadata={"Software": "scripts/render_docs.py"})
    plt.close(fig)


def title(fig, heading, subheading):
    fig.text(.06, .94, heading, size=19, weight="bold", va="top")
    fig.text(.06, .875, subheading, size=10.5, color=MUTED, va="top")


def overview(summary, runs, source_hash):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=.17, right=.96, top=.72, bottom=.24, wspace=.45)
    counts = {len(run["http_attempts"]) for run in runs}
    count_label = f"{next(iter(counts))} HTTP requests per configuration" if len(counts) == 1 else "request counts vary by configuration"
    title(fig, "What did the native int4 study measure?",
          f"Jetson Thor · valid output only · {count_label} · one server start each")
    for pos, run in enumerate(runs):
        label, color = LABELS.get(run["profile"], run["profile"]), COLORS.get(run["profile"], "#456d99")
        rate = run["http_summary"]["successful_completion_tokens_per_second"]
        axes[0].barh(pos, rate, height=.46, color=color)
        axes[0].text(rate+.012, pos, f"{rate:.3f}", va="center", weight="bold", color=color)
        attempts = run["http_attempts"]
        jitter = [(i-(len(attempts)-1)/2)*.065 for i in range(len(attempts))]
        first = [r["first_output_seconds"] for r in attempts]
        axes[1].scatter(first, [pos+j for j in jitter], s=55, color=color, alpha=.65)
        median = run["http_summary"]["successful_first_output_seconds"]["p50"]
        axes[1].plot([median, median], [pos-.23, pos+.23], color=color, lw=3)
        axes[1].text(median, pos+.38, f"p50 {median:.1f}s", ha="center", size=10, color=color)
    names = [LABELS.get(r["profile"], r["profile"]) for r in runs]
    for ax in axes:
        ax.set_yticks(range(len(runs)), names)
        ax.invert_yaxis()
        ax.set_ylim(len(runs)-.45, -.45)
        ax.xaxis.grid(True, color="#e7edf1")
        ax.set_axisbelow(True)
    axes[0].set_xlim(0, max(r["http_summary"]["successful_completion_tokens_per_second"] for r in runs)*1.3)
    axes[0].set_title("Whole-batch completion throughput", loc="left", size=12, pad=13)
    axes[0].set_xlabel("Completion tokens / second, including prefill")
    axes[1].set_xlim(0, max(a["first_output_seconds"] for r in runs for a in r["http_attempts"])*1.15)
    axes[1].set_title("Time to first nonempty output", loc="left", size=12, pad=13)
    axes[1].set_xlabel("Seconds · dots are individual requests")
    qualities = {run["quality_total"] for run in runs}
    quality_label = f"{next(iter(qualities))}/{next(iter(qualities))}" if len(qualities) == 1 else "all"
    fig.text(.06, .12, f"All displayed configurations passed {quality_label} strict smoke checks.", size=11, weight="bold")
    fig.text(.06, .065, "Default CUDA generated non-finite logits and is excluded. Small, ordered study; no confidence intervals or general speedup claim.", size=9.4, color=RED)
    save(fig, "results-overview", "Measured throughput and first-output latency of valid native int4 runs", source_hash)


def phases(runs, source_hash):
    import matplotlib.pyplot as plt
    fields = [("expert_disk_s", "Expert disk reads"), ("expert_matmul_s", "Expert compute timer"),
              ("attention_s", "Attention timer"), ("lm_head_s", "Output head timer")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=.22, right=.94, top=.73, bottom=.19)
    title(fig, "Where did each request spend time?",
          "Engine-reported phase timers · mean across measured whole HTTP turns per configuration")
    for index, run in enumerate(runs):
        values = [statistics.mean(t[key] for t in run["engine_profile"]["turns"]) for key, _ in fields]
        offset = (index-(len(runs)-1)/2)*.30
        ys = [i+offset for i in range(len(fields))]
        color = COLORS.get(run["profile"], "#456d99")
        ax.barh(ys, values, height=.26, color=color, label=LABELS.get(run["profile"], run["profile"]))
        for y, value in zip(ys, values):
            ax.text(value+.7, y, f"{value:.1f}s", va="center", color=color, size=10)
    ax.set_yticks(range(len(fields)), [name for _, name in fields])
    ax.invert_yaxis()
    ax.set_xlim(0, max(statistics.mean(t[k] for t in r["engine_profile"]["turns"]) for r in runs for k, _ in fields)*1.16)
    ax.set_xlabel("Reported seconds per turn")
    ax.xaxis.grid(True, color="#e7edf1")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    fig.text(.06, .08, "These timers are not assumed to be additive or causal; asynchronous work may overlap. See the methodology and raw phase fields.", size=9.3, color=MUTED)
    save(fig, "phase-times", "Reported mean phase times, shown separately without assuming additive timers", source_hash)


def resources(summary, runs, source_hash):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=.18, right=.96, top=.71, bottom=.30, wspace=.55)
    title(fig, "Requested memory is not observed placement",
          "Startup tier capacities and process-group reads describe different parts of the experiment")
    cuda = [r for r in summary["runs"] if r.get("observed_device_expert_tier")]
    for i, run in enumerate(cuda):
        tier = run["observed_device_expert_tier"]
        requested = tier["requested_budget_gib"]
        observed = tier["reported_budget_gib"]
        color = RED if run["status"] != "complete" else COLORS.get(run["profile"], "#008778")
        axes[0].barh(i, requested, height=.53, color="#e6edf1", label="Requested" if i == 0 else None)
        axes[0].barh(i, observed, height=.28, color=color, label="Engine-reported" if i == 0 else None)
        axes[0].text(max(observed, requested)+.4, i, f"{observed:.2f} / {requested:g}", va="center", size=10)
    axes[0].set_yticks(range(len(cuda)), ["Default CUDA\ninvalid output" if r["status"] != "complete" else "Conservative\nCUDA" for r in cuda])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(r["observed_device_expert_tier"]["requested_budget_gib"] for r in cuda)*1.45)
    axes[0].set_title("Expert device-tier capacity", loc="left", pad=13, size=12)
    axes[0].set_xlabel("GiB · reported / requested")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#e6edf1", label="Requested"),
        Patch(facecolor=RED, label="Reported: invalid run"),
        Patch(facecolor="#008778", label="Reported: valid run")],
        loc="center left", bbox_to_anchor=(.18, .165), frameon=False, fontsize=8, ncol=3)
    for i, run in enumerate(runs):
        value = run["measurement_read_bytes"]/1e9
        color = COLORS.get(run["profile"], "#456d99")
        axes[1].barh(i, value, color=color, height=.46)
        axes[1].text(value+7, i, f"{value:.1f}", va="center", color=color, weight="bold")
    axes[1].set_yticks(range(len(runs)), [LABELS.get(r["profile"], r["profile"]) for r in runs])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(r["measurement_read_bytes"]/1e9 for r in runs)*1.24)
    axes[1].set_title("Reads during measured HTTP requests", loc="left", pad=13, size=12)
    axes[1].set_xlabel("Decimal GB · owned process-group read_bytes")
    for ax in axes:
        ax.xaxis.grid(True, color="#e7edf1")
        ax.set_axisbelow(True)
    fig.text(.06, .12, "Tier numbers are rounded startup budgets, not measured allocated bytes.", size=10, color=MUTED)
    fig.text(.06, .065, "Read counters exclude startup and separate quality requests. Repeated weight reads can exceed the checkpoint's file size.", size=9.5, color=MUTED)
    save(fig, "placement-and-reads", "Requested versus reported GPU expert capacity and measured process reads", source_hash)


DESCRIPTIONS = {
    ".gitignore": "Keep weights, private runtime state, upstream checkout, and local logs out of Git.",
    "README.md": "Start here: learning outcomes, key measurements, study map, and run commands.",
    "requirements-docs.txt": "Optional plotting dependency; not required for inference or reading figures.",
    "configs/experiment.json": "Pinned engine/model revisions, model path, context, and resource guards.",
    "configs/model-manifest.json": "Immutable native-model file list, sizes, and content digests.",
    "configs/profiles.json": "Requested CPU/GPU expert budgets and conservative CUDA math setting.",
    "docs/ARCHITECTURE.md": "Inference and measurement diagrams, shared-memory interpretation, evidence flow.",
    "docs/DIAGNOSTICS.md": "Matched CUDA failure/workaround reproduction and corrected runner assumptions.",
    "docs/INDEX.md": "This linked inventory of every public tracked file.",
    "docs/INTERACTIVE.md": "Try terminal chat, stop only the lab campaign, and select CPU or CUDA.",
    "docs/ISSUES_AND_PRS.md": "Issue tracker, fixes, open questions, and relevant upstream PRs/issues.",
    "docs/LAB_NOTES.md": "Chronological study steps, commands, observations, and decisions.",
    "docs/METHODOLOGY.md": "Metric definitions, workload, cache state, controls, and interpretation limits.",
    "docs/NEXT_EXPERIMENTS.md": "Proposed repeated trials and residency, context, I/O, and precision sweeps.",
    "docs/REFERENCES.md": "Primary engine/model sources and contextual first-person measurements.",
    "docs/REPRODUCE.md": "Step-by-step prerequisites, build, validation, download, and benchmark procedure.",
    "docs/RESULTS.md": "Generated measured-results table, graphs, quality gates, and limitations.",
    "results/build-validation.json": "Pinned binary hash, CUDA architecture, and 46 kernel-check outcomes.",
    "results/cuda-glm53c-check.txt": "Sanitized output of the actual GLM CUDA layer-op check.",
    "results/environment.json": "Sanitized CPU/GPU, RAM, driver, compiler, storage, and power-mode facts.",
    "results/model-validation.json": "Native shard, tensor, packed-weight, and scale-geometry checks.",
    "results/native-expert-check.json": "Independent NumPy/C comparison on two actual expert projections.",
    "results/numerical-diagnostics.json": "CPU/default-CUDA/conservative-CUDA logit finiteness and continuations.",
    "results/summary.json": "Public per-attempt timings, quality gates, phase fields, I/O, and observed placement.",
    "tests/test_lab.py": "Regression tests for integrity, port reuse, numeric gates, publication, and placement.",
    "workloads/quality.json": "Two strict known-answer smoke cases; not a broad capability benchmark.",
    "workloads/smoke.jsonl": "Exact versioned performance prompts for the HTTP study.",
    "scripts/setup.sh": "Clone pinned engine, build sm_110, run CUDA gate, and record binary identity.",
    "docs/assets/figures.json": "Chart input hash, renderer version, interpretation notes, and asset inventory.",
}


def file_description(path):
    if path in DESCRIPTIONS:
        return DESCRIPTIONS[path]
    if path.startswith("docs/assets/"):
        label = {"results-overview": "Valid-run throughput and first-output latency",
                 "phase-times": "Separate engine-reported phase timers",
                 "placement-and-reads": "Requested/reported tier capacity and physical read counters"}.get(Path(path).stem, "Generated figure")
        return label + ("; scalable SVG." if path.endswith(".svg") else "; portable PNG.")
    if path.endswith(".py"):
        doc = ast.get_docstring(ast.parse((ROOT / path).read_text())) or "Python lab tool."
        return doc.splitlines()[0].replace("|", "/")
    return "Public study artifact."


def build_index():
    files = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().strip("\0").split("\0"))
    files.add("docs/INDEX.md")
    files.discard("")
    groups = [("Start and configuration", lambda p: "/" not in p or p.startswith("configs/")),
              ("Guides and diagrams", lambda p: p.startswith("docs/") and not p.startswith("docs/assets/")),
              ("Charts and provenance", lambda p: p.startswith("docs/assets/")),
              ("Measured evidence", lambda p: p.startswith("results/")),
              ("Executable lab tools", lambda p: p.startswith("scripts/")),
              ("Tests and workloads", lambda p: p.startswith(("tests/", "workloads/")))]
    text = ["# Repository file index", "", f"A linked inventory of all **{len(files)} public tracked files**, generated by `scripts/render_docs.py`.", "",
        "For a reading path: [README](../README.md) → [architecture](ARCHITECTURE.md) → [results](RESULTS.md) → [methodology](METHODOLOGY.md) → [reproduction](REPRODUCE.md).", "",
        "To inspect the evidence without downloading weights, read `results/summary.json` and the chart provenance. To run the model, follow the reproduction guide."]
    for title_, predicate in groups:
        text += ["", "## " + title_, "", "| File | What to find |", "|---|---|"]
        for path in sorted(p for p in files if predicate(p)):
            text.append(f"| [{path}](../{path}) | {file_description(path)} |")
    text += ["", "## Local-only directories", "", "These directories are intentionally absent from the public repository:", "",
        "| Path | Purpose |", "|---|---|", "| `models/` | Approximately 195 GB of native checkpoint weights. |",
        "| `vendor/` | Pinned upstream source checkout and compiled engine. |",
        "| `runtime/` | Process IDs, local progress, helper binaries, and plotting cache. |",
        "| `logs/` | Local build, transfer, and campaign logs. |",
        "| `results/raw/` | Detailed local attempts, logs, quality responses, and telemetry. |", "",
        "Regenerate after staging new public files: `python3 scripts/render_docs.py`."]
    (ROOT / "docs/INDEX.md").write_text("\n".join(text) + "\n")


def main():
    source = ROOT / "results/summary.json"
    summary = json.loads(source.read_text())
    runs = checked_runs(summary)
    if not runs:
        raise SystemExit("No quality-checked completed runs to graph; no performance figure was invented")
    matplotlib = configure()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    overview(summary, runs, source_hash)
    phases(runs, source_hash)
    resources(summary, runs, source_hash)
    provenance = {"source": "results/summary.json", "source_sha256": source_hash,
        "renderer": "scripts/render_docs.py", "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "performance_gate": "Complete run, finite output, and every strict smoke check passed",
        "notes": ["Observed measurements only; default invalid CUDA excluded from speed comparison",
            "HTTP requests within one server start per profile are correlated, not independent trials",
            "Phase timers shown separately without assuming additive timing",
            "SVG IDs and metadata are deterministic in the recorded local renderer; fonts/layout can vary across Matplotlib versions"],
        "figures": [name + suffix for name in ("results-overview", "phase-times", "placement-and-reads") for suffix in (".svg", ".png")]}
    (ROOT / "docs/assets/figures.json").write_text(json.dumps(provenance, indent=2) + "\n")
    build_index()
    print("Rendered 3 charts as SVG and PNG; wrote chart provenance and file index.")


if __name__ == "__main__":
    main()
