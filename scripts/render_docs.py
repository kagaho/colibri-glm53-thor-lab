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
import re
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
    source = "Source: results/summary.json, SHA256 " + source_hash if source_hash else "Diagram source: scripts/render_docs.py"
    metadata = {"Title": description, "Description": source,
                "Creator": "scripts/render_docs.py", "Date": None}
    svg = folder / (name + ".svg")
    fig.savefig(svg, metadata=metadata)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(folder / (name + ".png"), dpi=150, metadata={"Software": "scripts/render_docs.py"})
    plt.close(fig)


def title(fig, heading, subheading):
    fig.text(.06, .94, heading, size=19, weight="bold", va="top")
    fig.text(.06, .875, subheading, size=10.5, color=MUTED, va="top")


def workflow():
    """Draw a compact workflow with an opaque background for the README."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def box(x, y, text, color="#edf2f8", width=.25, height=.14):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle="round,pad=0.01,rounding_size=0.018", facecolor=color,
            edgecolor="#b8cad6", linewidth=1.2, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=12, color=INK, zorder=3)
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.6,
                        "mutation_scale": 15, "shrinkA": 0, "shrinkB": 0}, zorder=1)
    box(.16, .82, "Pin engine\n& native weights")
    box(.50, .82, "Build & run\nvalidation checks")
    box(.84, .82, "Benchmark &\ncollect telemetry", "#edf6f6")
    arrow((.295, .82), (.365, .82))
    arrow((.635, .82), (.705, .82))
    ax.add_patch(Polygon([(.50, .64), (.66, .52), (.50, .40), (.34, .52)],
        facecolor="#fff4d9", edgecolor="#d7ba71", linewidth=1.2, zorder=2))
    ax.text(.50, .52, "Finite output +\nanswer checks?", ha="center", va="center", fontsize=11.5, color=INK)
    arrow((.84, .74), (.628, .544))
    box(.22, .27, "Interpret valid\nresults", "#edf6f6", width=.28)
    box(.78, .27, "Keep failure\n& diagnose", "#fbeded", width=.28)
    arrow((.36, .50), (.22, .35))
    arrow((.64, .50), (.78, .35))
    ax.text(.25, .45, "Pass", ha="center", color="#008778", fontsize=11, weight="bold")
    ax.text(.75, .45, "Fail", ha="center", color=RED, fontsize=11, weight="bold")
    box(.50, .07, "Reproduce or extend", "#f1f4f6", width=.32, height=.10)
    arrow((.22, .19), (.329, .10))
    arrow((.78, .19), (.671, .10))
    save(fig, "study-workflow", "Study workflow: pin, validate, benchmark, check output, interpret or diagnose, and extend", None)


def overview(summary, runs, source_hash):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))
    fig.subplots_adjust(left=.17, right=.96, top=.55, bottom=.20, wspace=.45)
    counts = {len(run["http_attempts"]) for run in runs}
    count_label = f"{next(iter(counts))} HTTP requests per configuration" if len(counts) == 1 else "request counts vary by configuration"
    title(fig, "Native int4 on Jetson Thor: measured results",
          f"SSD-backed Colibri · {count_label} · one server start each")
    featured = next((run for run in reversed(runs) if run["profile"] == "cuda-conservative-16-32"), runs[-1])
    featured_http = featured["http_summary"]
    cards = [("Completion throughput", f'{featured_http["successful_completion_tokens_per_second"]:.3f} tok/s'),
             ("TTFT proxy · p50", f'{featured_http["successful_first_output_seconds"]["p50"]:.2f} s'),
             ("Known-answer smoke checks", f'{featured["quality_passed"]} / {featured["quality_total"]}')]
    for index, (label, value) in enumerate(cards):
        x = .06 + index * .31
        fig.add_artist(FancyBboxPatch((x, .68), .28, .145, transform=fig.transFigure,
            boxstyle="round,pad=0.008,rounding_size=0.012", facecolor="#edf6f6", edgecolor="none", zorder=0))
        fig.text(x+.015, .792, label, size=10.5, color=MUTED)
        fig.text(x+.015, .736, value, size=24, weight="bold", color=COLORS.get(featured["profile"], INK))
        fig.text(x+.015, .698, LABELS.get(featured["profile"], featured["profile"]), size=9.5, color=MUTED)
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
    axes[0].set_title("Completion tokens/s · higher is better", loc="left", size=12, pad=13)
    axes[0].set_xlabel("Whole HTTP batch, including prefill")
    axes[1].set_xlim(0, max(a["first_output_seconds"] for r in runs for a in r["http_attempts"])*1.15)
    axes[1].set_title("TTFT proxy · lower is better", loc="left", size=12, pad=13)
    axes[1].set_xlabel("Seconds · dots are individual requests")
    fig.text(.06, .095, "TTFT proxy = first nonempty streamed output; exact first-token timing and steady decode speed were not recorded.", size=9.5, color=MUTED)
    fig.text(.06, .045, "Default CUDA produced invalid output and is excluded. Small, ordered study; requests are correlated within each server start.", size=9.4, color=RED)
    save(fig, "results-overview", "Measured throughput and first-output latency of valid native int4 runs", source_hash)


def readme_metrics(runs):
    """Keep the front-page table in sync with the latest quality-checked profiles."""
    latest = {run["profile"]: run for run in runs}
    selected = [latest[name] for name in LABELS if name in latest]
    if not selected:
        selected = list(latest.values())
    def http(run):
        return run["http_summary"]
    rows = [
        ("**Completion throughput**", lambda r: f'**{http(r)["successful_completion_tokens_per_second"]:.3f} tok/s**'),
        ("**TTFT proxy — p50**", lambda r: f'**{http(r)["successful_first_output_seconds"]["p50"]:.2f} s**'),
        ("TTFT proxy — observed range", lambda r: f'{min(a["first_output_seconds"] for a in r["http_attempts"]):.2f}–{max(a["first_output_seconds"] for a in r["http_attempts"]):.2f} s'),
        ("Full request duration — p50", lambda r: f'{http(r)["successful_duration_seconds"]["p50"]:.2f} s'),
        ("Model/server startup", lambda r: f'{r["startup_seconds"]:.2f} s'),
        ("Known-answer smoke checks", lambda r: f'{r["quality_passed"]}/{r["quality_total"]} passed'),
        ("Measured HTTP requests / reported completion tokens", lambda r: f'{http(r)["requests"]} / {http(r)["reported_successful_completion_tokens"]}'),
        ("Storage reads during measured requests", lambda r: f'{r["measurement_read_bytes"]/1e9:.1f} GB'),
        ("Lowest global available memory during the run", lambda r: f'{r["minimum_global_memavailable_bytes"]/2**30:.2f} GiB')]
    lines = ["| Measured metric | " + " | ".join(LABELS.get(r["profile"], r["profile"]) for r in selected) + " |",
             "|---|" + "---:|"*len(selected)]
    lines += ["| " + label + " | " + " | ".join(value(r) for r in selected) + " |" for label, value in rows]
    readme = ROOT / "README.md"
    updated, count = re.subn(r"<!-- METRICS START -->.*?<!-- METRICS END -->",
        "<!-- METRICS START -->\n\n" + "\n".join(lines) + "\n\n<!-- METRICS END -->", readme.read_text(), flags=re.S)
    if count != 1:
        raise RuntimeError("Expected one README metric block; refusing an ambiguous update")
    readme.write_text(updated)


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
    "LICENSE": "Apache 2.0 license for the lab's original code/documentation; third-party components retain their own licenses.",
    "requirements-docs.txt": "Optional plotting dependency; not required for inference or reading figures.",
    "requirements-demo.txt": "Optional Pillow and pyte dependencies for the real terminal GIF replay.",
    "configs/experiment.json": "Pinned engine/model revisions, model path, context, and resource guards.",
    "configs/model-manifest.json": "Immutable native-model file list, sizes, and content digests.",
    "configs/profiles.json": "Requested CPU/GPU expert budgets and conservative CUDA math setting.",
    "docs/ARCHITECTURE.md": "Inference and measurement diagrams, shared-memory interpretation, evidence flow.",
    "docs/DIAGNOSTICS.md": "Matched CUDA failure/workaround reproduction and corrected runner assumptions.",
    "docs/DEMO.md": "Real Rust Fibonacci chat demo: capture, 30-second acceleration, clock, and reproduction.",
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
    "results/rust-fibonacci.cast": "Actual timestamped terminal I/O, clipped to the ready-prompt-to-answer interval.",
    "results/rust-fibonacci-output.txt": "Final terminal screen of the real Rust Fibonacci response and chat statistics.",
    "results/rust-fibonacci-demo.json": "GIF/source hashes, real duration, 30-second export, speed factor, and capture settings.",
    "docs/assets/rust-fibonacci-30s.gif": "30-second real terminal replay with an accelerated elapsed-time clock at top right.",
    "tests/test_lab.py": "Regression tests for integrity, port reuse, numeric gates, publication, and placement.",
    "workloads/quality.json": "Two strict known-answer smoke cases; not a broad capability benchmark.",
    "workloads/smoke.jsonl": "Exact versioned performance prompts for the HTTP study.",
    "scripts/setup.sh": "Clone pinned engine, build sm_110, run CUDA gate, and record binary identity.",
    "scripts/render_comparison.py": "Rebuild the opening CPU/16/32 GiB chart from separate historical and follow-up public summaries.",
    "docs/assets/benchmark-comparison.json": "Opening comparison chart: both source hashes, plotted values, scope notes, and renderer identity.",
    "docs/assets/figures.json": "Chart input hash, renderer version, interpretation notes, and asset inventory.",
}


def file_description(path):
    if path in DESCRIPTIONS:
        return DESCRIPTIONS[path]
    if path.startswith('Tunning-Enhance-1/'):
        if path.endswith('.py'):
            return (ast.get_docstring(ast.parse((ROOT/path).read_text())) or 'Tuning study tool.').splitlines()[0].replace('|','/')
        return 'Isolated 16/32 GiB tuning study: '+path.split('/',1)[1]+'. See its own linked index for context.'
    if path.startswith("docs/assets/"):
        label = {"benchmark-comparison": "Historical and follow-up CPU/16/32 GiB speed, latency, and memory comparison",
                 "results-overview": "Valid-run throughput and first-output latency",
                 "phase-times": "Separate engine-reported phase timers",
                 "placement-and-reads": "Requested/reported tier capacity and physical read counters",
                 "study-workflow": "Compact study workflow with an opaque white background"}.get(Path(path).stem, "Generated figure")
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
              ("Tests and workloads", lambda p: p.startswith(("tests/", "workloads/"))),
              ("Tunning-Enhance-1 study", lambda p: p.startswith('Tunning-Enhance-1/'))]
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
    workflow()
    readme_metrics(runs)
    provenance = {"source": "results/summary.json", "source_sha256": source_hash,
        "renderer": "scripts/render_docs.py", "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "diagram_source": "scripts/render_docs.py",
        "performance_gate": "Complete run, finite output, and every strict smoke check passed",
        "notes": ["Observed measurements only; default invalid CUDA excluded from speed comparison",
            "HTTP requests within one server start per profile are correlated, not independent trials",
            "Phase timers shown separately without assuming additive timing",
            "TTFT proxy is first nonempty streamed output, not exact first-token timing; steady decode and per-token latency were not measured",
            "SVG IDs and metadata are deterministic in the recorded local renderer; fonts/layout can vary across Matplotlib versions"],
        "figures": [name + suffix for name in ("results-overview", "phase-times", "placement-and-reads", "study-workflow") for suffix in (".svg", ".png")]}
    (ROOT / "docs/assets/figures.json").write_text(json.dumps(provenance, indent=2) + "\n")
    build_index()
    print("Rendered 3 charts and workflow as SVG and PNG; refreshed README metrics, provenance, and file index.")


if __name__ == "__main__":
    main()
