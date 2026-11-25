#!/usr/bin/env python3
"""Publish selected metrics from raw local runs, retaining failed attempts."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def parse_device_tier(log, requested_gib=None):
    """Read the engine's startup capacity; do not guess when telemetry is absent."""
    match = re.search(r"^\[CUDA\] tier glm53: on, .*?, budget ([\d.]+) (GiB|MiB|GB|MB) = (\d+) experts", log, re.M)
    if not match:
        return None
    value, unit, experts = match.groups()
    factor = {"GiB": 2**30, "MiB": 2**20, "GB": 10**9, "MB": 10**6}[unit]
    budget_gib = float(value)*factor/2**30
    return {"reported_budget": float(value), "reported_unit": unit,
        "reported_budget_gib": budget_gib, "expert_capacity": int(experts),
        "requested_budget_gib": requested_gib,
        "clamped_below_requested": None if requested_gib is None else budget_gib+0.02 < requested_gib,
        "evidence": match.group(0),
        "scope": "Rounded startup tier budget and capacity, not measured steady-state allocated bytes"}


def summarize_run(data, raw_dir):
    log_path = raw_dir / "server.log"
    nonfinite = data.get("numerical_failure_detected", False) or (
        log_path.exists() and "[router] non-finite logits" in log_path.read_text(errors="replace"))
    summary = {
        "run_id": raw_dir.name, "status": data["status"], "profile": data["profile"],
        "error": data.get("error"), "profile_settings": data["profile_settings"],
        "cache_state": data["cache_state"], "io_mode": data["io_mode"],
        "workload_sha256": data["workload_sha256"],
        "startup_seconds": data.get("startup_seconds"),
        "measurement_read_bytes": data.get("measurement_read_bytes"),
        "measurement_rchar": data.get("measurement_rchar"),
        "minimum_global_memavailable_bytes": data.get("minimum_global_memavailable_bytes"),
        "peak_process_group_rss_sum_bytes": data.get("peak_process_group_rss_sum_bytes"),
        "rss_caution": data.get("rss_caution"),
        "protected_containers_unchanged": data.get("protected_containers_unchanged"),
        "cuda_routed_expert_execution_verified": data.get("cuda_routed_expert_execution_verified", False),
        "quality_passed": data.get("quality_passed"), "quality_total": data.get("quality_total"),
        "native_tier_cumulative_counters_at_http_stage": data.get("native_tier_cumulative_counters_at_http_stage"),
        "counter_scope": data.get("counter_scope"),
        "numerical_failure_detected": nonfinite,
    }
    if nonfinite:
        summary["status"] = "invalid_output"
        summary["error"] = "Non-finite router logits; repeated invalid output; performance is diagnostic only"
    requested = data["profile_settings"].get("gpu_tier_gib")
    summary["observed_device_expert_tier"] = data.get("observed_device_expert_tier") or (
        parse_device_tier(log_path.read_text(errors="replace"), requested) if log_path.exists() else None)
    if "G53_DIRECT" not in data.get("environment_overrides", {}) and data["profile"] != "fixture":
        summary["io_mode"] = "Native direct-I/O default enabled (G53_DIRECT unset); effective fallback path not traced"
        summary["metadata_correction"] = "Original runner mislabeled the upstream default as buffered; pinned glm53.c defaults G53_DIRECT to 1"
    http = data.get("http_report")
    path = raw_dir / "http.json"
    if not http and path.exists():
        http = json.loads(path.read_text())
    if http:
        summary["http_summary"] = http.get("summary")
        summary["http_attempts"] = http.get("requests")
    profile = raw_dir / "engine-profile.json"
    if profile.exists():
        summary["engine_profile"] = json.loads(profile.read_text())
    return summary


def main():
    runs = [summarize_run(json.loads(p.read_text()), p.parent)
            for p in sorted((ROOT / "results/raw").glob("*/run.json"))]
    campaign_path = ROOT / "runtime/campaign-state.json"
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    status = ("failed" if campaign.get("phase") == "failed" else "pending" if not runs else "measured")
    result = {"status": status, "campaign_error": campaign.get("error"),
        "metric": "HTTP completion-token throughput includes prefill, queueing, and HTTP overhead; it is not steady decode tok/s",
        "limitations": ["One machine and two short synthetic prompts; not a model intelligence benchmark",
            "I/O policy is recorded per attempt; the initial native-default direct-I/O run was mislabeled in its original metadata and is corrected in the curated summary",
            "Initial default CUDA failed numerically; valid comparison order was CPU then conservative CUDA, so cache state and drift can confound comparisons",
            "CPU and GPU profiles differ in device expert capacity as well as compute; no isolated GPU-kernel speedup claim",
            "Quality checks are two strict known-answer smoke tests, not a general quality score"], "runs": runs}
    (ROOT / "results/summary.json").write_text(json.dumps(result, indent=2) + "\n")
    text = ["# Results", "", "Full-model measurements are pending." if not runs else
        "These are initial measurements on one Thor. Failed attempts remain in the dataset.", "",
        "The build and GLM CUDA correctness evidence is in [build-validation.json](../results/build-validation.json).", "",
        "| Run | Status | Profile | Smoke checks | Reported device tier (GiB) | HTTP completion tok/s | First-output p50 (s) | Disk reads (GB) |", "|---|---|---|---|---:|---:|---:|---:|"]
    for run in runs:
        h = run.get("http_summary") or {}
        rate = h.get("successful_completion_tokens_per_second") if run["status"] == "complete" else None
        first = (h.get("successful_first_output_seconds") or {}).get("p50")
        read = run.get("measurement_read_bytes")
        def fmt(v):
            return "unavailable" if v is None else f"{v:.3f}"
        quality = (f"{run['quality_passed']}/{run['quality_total']}" if run.get('quality_total') is not None else "not run")
        observed = run.get("observed_device_expert_tier") or {}
        placement = fmt(observed.get("reported_budget_gib")) if observed else "not applicable" if run['profile_settings'].get('gpu') == 'none' else "unavailable"
        text.append(f"| {run['run_id']} | {run['status']} | {run['profile']} | {quality} | {placement} | {fmt(rate)} | {fmt(first)} | {fmt(None if read is None else read/1e9)} |")
    text += ["", "Invalid-output timings are excluded from the headline throughput column; raw diagnostic rates remain in JSON.", "",
             "HTTP throughput includes the whole measured batch. It must not be compared directly with steady-decode benchmarks.", "",
             "See [summary.json](../results/summary.json) for per-attempt metrics, memory, integrity checks, and limitations.", "",
             "## Measured performance", "",
             "![Measured throughput and first-output latency for quality-checked runs](assets/results-overview.svg)", "",
             "In the initial 2026-11-25 study, CPU measured 0.415 completion tok/s and conservative CUDA measured 0.517. Both passed 2/2 strict smoke checks. First nonempty output p50 was 44.5 s and 28.9 s respectively. These rates make that operating point a slow interactive experiment; they do not establish a general model or hardware ranking.", "",
             "The throughput bars include prefill and HTTP overhead. The latency dots are four requests within one server start per configuration, not independent benchmark trials. Default CUDA produced invalid output and is excluded from the speed comparison.", "",
             "## Reported phase times", "",
             "![Separate engine-reported mean phase times](assets/phase-times.svg)", "",
             "The expert-compute timer is the largest of the four displayed counters in both valid configurations. Conservative CUDA reported lower expert-compute and attention times, while its expert-disk timer was higher. These are whole-configuration observations: residency, cache state, and compute all differ. Timers may overlap and are not added into a wall-time breakdown.", "",
             "## Placement and storage reads", "",
             "![Requested versus reported GPU expert capacity and measured storage reads](assets/placement-and-reads.svg)", "",
             "Both CUDA attempts requested a 16 GiB expert tier. The rejected attempt reported only 1.54 GiB at startup; the valid conservative attempt reported 16.00 GiB. This is rounded engine-reported capacity, not measured allocated bytes. The exact cause of the earlier clamp remains unproven.", "",
             "The valid CPU and CUDA runs read approximately 377.1 GB and 365.9 GB during their four measured HTTP requests. These process-group counters exclude startup and separate quality checks. Repeated reads can exceed the 194.75 GB checkpoint size; the counters do not measure unique bytes or SSD bandwidth.", "",
             "## Inspect or reproduce", "",
             "- [Architecture and evidence flow](ARCHITECTURE.md): how SSD, shared memory, and correctness gates fit together.",
             "- [Reproduction guide](REPRODUCE.md): exact setup and benchmark commands.",
             "- [Methodology](METHODOLOGY.md): metric definitions and comparison limits.",
             "- [Diagnostics](DIAGNOSTICS.md) and [issues/fixes](ISSUES_AND_PRS.md): evidence behind the conservative CUDA workaround.",
             "- [Complete file index](INDEX.md): every public file and its purpose.", "",
             "Charts are generated from [summary.json](../results/summary.json) by [render_docs.py](../scripts/render_docs.py). The [figure manifest](assets/figures.json) records the source SHA-256 and plotting version; SVG and PNG exports are committed. Regenerate from the saved summary with `python3 scripts/render_docs.py` without downloading weights.", "",
             "These interpretation paragraphs describe the saved initial study. Revisit them when publishing a new campaign; the table and charts regenerate from data.", "",
             "## Limits", ""] + ["- " + line for line in result["limitations"]]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(text) + "\n")
    readme = ROOT / "README.md"
    value = ("**Current status:** build and 46 GLM CUDA kernel checks passed. Full-model measurements are pending; see [results](docs/RESULTS.md)."
        if not runs else f"**Current status:** {len(runs)} full-model attempt(s) recorded, "
        f"{sum(row['status'] == 'complete' for row in runs)} completed. See [results](docs/RESULTS.md) for measurements and failures.")
    if campaign.get("phase") == "failed":
        value += " The campaign stopped on a failure; inspect the recorded error before rerunning."
    readme.write_text(re.sub(r"<!-- STATUS START -->.*?<!-- STATUS END -->",
        "<!-- STATUS START -->\n" + value + "\n<!-- STATUS END -->", readme.read_text(), flags=re.S))
    print(f"Summarized {len(runs)} raw runs")


if __name__ == "__main__":
    main()
