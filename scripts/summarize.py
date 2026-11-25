#!/usr/bin/env python3
"""Publish selected metrics from raw local runs, retaining failed attempts."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


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
            "Initial profile order is CUDA then CPU; OS cache state and drift can confound comparisons",
            "CPU and GPU profiles differ in device expert capacity as well as compute; no isolated GPU-kernel speedup claim",
            "Quality checks are two strict known-answer smoke tests, not a general quality score"], "runs": runs}
    (ROOT / "results/summary.json").write_text(json.dumps(result, indent=2) + "\n")
    text = ["# Results", "", "Full-model measurements are pending." if not runs else
        "These are initial measurements on one Thor. Failed attempts remain in the dataset.", "",
        "The build and GLM CUDA correctness evidence is in [build-validation.json](../results/build-validation.json).", "",
        "| Run | Status | Profile | HTTP completion tok/s | First-output p50 (s) | Disk reads (GB) |", "|---|---|---|---:|---:|---:|"]
    for run in runs:
        h = run.get("http_summary") or {}
        rate = h.get("successful_completion_tokens_per_second") if run["status"] == "complete" else None
        first = (h.get("successful_first_output_seconds") or {}).get("p50")
        read = run.get("measurement_read_bytes")
        def fmt(v):
            return "unavailable" if v is None else f"{v:.3f}"
        text.append(f"| {run['run_id']} | {run['status']} | {run['profile']} | {fmt(rate)} | {fmt(first)} | {fmt(None if read is None else read/1e9)} |")
    text += ["", "Invalid-output timings are excluded from the headline throughput column; raw diagnostic rates remain in JSON.", "",
             "HTTP throughput includes the whole measured batch. It must not be compared directly with steady-decode benchmarks.", "",
             "See [summary.json](../results/summary.json) for per-attempt metrics, memory, integrity checks, and limitations.", "",
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
