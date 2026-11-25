#!/usr/bin/env python3
"""Own and measure one Colibri server; never stop an existing user service."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def meminfo():
    return {line.split()[0].rstrip(":"): int(line.split()[1])*1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(line.split()) >= 3 and line.split()[2] == "kB"}


def process_tree(group):
    result = {"rss_bytes_sum": 0, "read_bytes": 0, "rchar": 0, "pids": []}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            result["pids"].append(int(proc.name))
            result["rss_bytes_sum"] += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            io = dict(line.split(": ", 1) for line in (proc / "io").read_text().splitlines())
            for key in ("read_bytes", "rchar"):
                result[key] += int(io[key])
        except (OSError, ValueError, IndexError):
            continue
    return result


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def owned_stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if proc.poll() is None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def check_port_available(port):
    # Match HTTPServer's reuse policy for recently closed TIME_WAIT sockets.
    # A live listener still rejects this bind; no process is displaced.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def container_snapshot():
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True).split()
    if not ids:
        return {}
    items = json.loads(subprocess.check_output(["docker", "inspect", *ids]))
    return {c["Id"]: {"image": c["Image"], "state": c["State"]["Status"],
        "started": c["State"]["StartedAt"], "mounts": c["Mounts"]} for c in items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profiles = json.loads((ROOT / "configs/profiles.json").read_text())
    parser.add_argument("--profile", choices=profiles, default="cuda-16-32")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 256 or not 1 <= args.repeats <= 5:
        parser.error("Use max-tokens 1..256 and repeats 1..5 for this bounded experiment")
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_model import validate
    validation = validate()
    check_path = ROOT / "runtime/cuda-check-passed.json"
    if not check_path.is_file():
        raise RuntimeError("Run the GLM CUDA correctness check before the experiment")
    check = json.loads(check_path.read_text())
    if (check.get("engine_commit") != cfg["engine_commit"] or check.get("binary_sha256") !=
            hashlib.sha256((ROOT / "vendor/colibri/c/glm53").read_bytes()).hexdigest()):
        raise RuntimeError("Engine changed after the recorded CUDA check")
    profile = profiles[args.profile]
    port = cfg["port"]
    check_port_available(port)
    baseline = container_snapshot()
    if any(v["state"] == "running" for key, v in baseline.items()
           if key.startswith("65023526b31b")):
        raise RuntimeError("The protected large Qwen model is running; do not load GLM beside it")
    if meminfo()["MemAvailable"] < 80 * 2**30:
        raise RuntimeError("At least 80 GiB available RAM is required for this initial experiment")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / "results/raw" / f"{stamp}-{args.profile}"
    run.mkdir(parents=True)
    command = ["python3", str(ROOT / "vendor/colibri/c/coli"), "serve",
        "--model", str(ROOT / cfg["model_directory"]), "--ctx", str(cfg["context"]),
        "--gpu", profile["gpu"], "--no-tune-profile", "--temp", "0", "--effort", "low",
        "--ngen", "256", "--kv-slots", "1", "--host", "127.0.0.1", "--port", str(port),
        "--model-id", "glm53-flash-native-int4"]
    settings = {"GLM53_BITS": "4", "GLM53_EXPERT_GB": str(profile["expert_cache_gb"]),
        "COLI_CUDA": "0" if profile["gpu"] == "none" else "1", "COLI_VULKAN": "0",
        "COLI_VK_TIER_GB": str(profile["gpu_tier_gib"]), "COLI_VK_TIER_HOST": "0",
        "COLI_PREFIX_STORE": "0", "USAGE_SAVE": "0", "G53_DIRECT": "0",
        "COLI_USAGE": str(run / "fresh-usage"), "OMP_NUM_THREADS": "14",
        "GLM53_VERBOSE": "1"}
    if profile.get("tensor_core_gemm") is False:
        settings["COLI_CUDA_TC_GEMM"] = "0"
    environment = {k: v for k, v in os.environ.items()
        if not k.startswith(("COLI_", "COLIBRI_", "GLM53_", "CUDA_", "OMP_"))
        and k not in ("RAM_GB", "CAP", "PIN", "PIN_GB", "SNAP", "SEED", "SERVE", "SERVE_BATCH",
                      "KV_SLOTS", "KVSAVE", "USAGE_SAVE", "DIRECT", "PIPE", "URING", "PREFETCH", "PROF")}
    environment.update(settings)
    report = {"status": "starting", "profile": args.profile, "configuration": cfg,
        "profile_settings": profile, "command": command, "environment_overrides": settings,
        "model_validation": validation, "cache_state": "fresh engine/LRU; OS page cache uncontrolled",
        "io_mode": "explicit buffered reads (G53_DIRECT=0); no global cache flush",
        "workload_sha256": hashlib.sha256((ROOT / "workloads/smoke.jsonl").read_bytes()).hexdigest()}
    write_json(run / "run.json", report)
    stop_monitor = threading.Event()
    guard_triggered = threading.Event()
    numerical_failure = threading.Event()
    proc = None
    minimum_available = [meminfo()["MemAvailable"]]
    peak_rss = [0]
    client = None
    thread = None
    started = time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Experiment interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (run / "server.log").open("w") as server_log:
            proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=server_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            write_json(ROOT / "runtime/server.json", {"pid": proc.pid, "port": port,
                "run_directory": str(run), "profile": args.profile})

            def monitor():
                low_count = 0
                with (run / "telemetry.jsonl").open("w") as out:
                    while not stop_monitor.is_set() and proc.poll() is None:
                        memory, process = meminfo(), process_tree(proc.pid)
                        minimum_available[0] = min(minimum_available[0], memory["MemAvailable"])
                        peak_rss[0] = max(peak_rss[0], process["rss_bytes_sum"])
                        out.write(json.dumps({"seconds": time.monotonic()-started,
                            "memory": memory, "process_group": process}) + "\n")
                        out.flush()
                        log_path = run / "server.log"
                        if "[router] non-finite logits" in log_path.read_text(errors="replace"):
                            numerical_failure.set()
                            os.killpg(proc.pid, signal.SIGTERM)
                            return
                        low_count = low_count + 1 if memory["MemAvailable"] < cfg["minimum_available_memory_gib"]*2**30 else 0
                        if low_count >= 3:
                            guard_triggered.set()
                            os.killpg(proc.pid, signal.SIGTERM)
                            return
                        stop_monitor.wait(1)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Owned server exited before readiness; inspect server.log")
                try:
                    health = get_json(base + "/health")
                    if health.get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup deadline exceeded")
                time.sleep(2)
            report["startup_seconds"] = time.monotonic() - started
            from summarize import parse_device_tier
            report["observed_device_expert_tier"] = parse_device_tier(
                (run / "server.log").read_text(errors="replace"), profile["gpu_tier_gib"])
            write_json(run / "health.json", health)
            io_before = process_tree(proc.pid)
            bench = ["python3", str(ROOT / "vendor/colibri/c/tools/benchmark_http_serving.py"),
                "--base-url", base + "/v1", "--model", "glm53-flash-native-int4",
                "--workload", str(ROOT / "workloads/smoke.jsonl"), "--concurrency", "1",
                "--repeats", str(args.repeats), "--max-tokens", str(args.max_tokens),
                "--temperature", "0", "--timeout", "600", "--output", str(run / "http.json")]
            report["benchmark_command"] = bench
            with (run / "client.log").open("w") as log:
                client = subprocess.Popen(bench, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}, start_new_session=True)
                try:
                    client.wait(timeout=2400)
                except subprocess.TimeoutExpired:
                    owned_stop(client)
                    raise TimeoutError("Bounded smoke campaign exceeded 40 minutes")
            io_after = process_tree(proc.pid)
            report["measurement_read_bytes"] = max(0, io_after["read_bytes"] - io_before["read_bytes"])
            report["measurement_rchar"] = max(0, io_after["rchar"] - io_before["rchar"])
            if client.returncode:
                raise RuntimeError("HTTP benchmark failed; failures retained in raw results")
            write_json(run / "engine-profile.json", get_json(base + "/profile"))
            bench_log = (run / "server.log").read_text(errors="replace")
            counters = re.findall(r"\[CUDA\] tier glm53 (?:run|turn): device (\d+) of (\d+).*?CPU RAM hits (\d+), disk loads (\d+)", bench_log)
            report["native_tier_cumulative_counters_at_http_stage"] = [
                dict(zip(("device_experts_served", "routed_experts", "cpu_ram_hits", "disk_loads"), map(int, row)))
                for row in counters]
            report["counter_scope"] = "Native engine cumulative expert counters through each HTTP turn; GPU and CPU-cache denominators differ"
            report["http_report"] = json.loads((run / "http.json").read_text())
            if args.quality:
                quality = []
                for case in json.loads((ROOT / "workloads/quality.json").read_text()):
                    request = urllib.request.Request(base + "/v1/chat/completions",
                        data=json.dumps({"model": "glm53-flash-native-int4", "stream": False,
                            "temperature": 0, "max_tokens": 256,
                            "messages": [{"role": "user", "content": case["prompt"]}]}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=900) as response:
                        reply = json.load(response)
                    answer = reply["choices"][0]["message"].get("content") or ""
                    if "expected_json" in case:
                        try:
                            passed = json.loads(answer.strip()) == case["expected_json"]
                        except ValueError:
                            passed = False
                    else:
                        passed = answer.strip() == case["expected"]
                    quality.append({"id": case["id"], "passed": passed, "response": reply})
                    write_json(run / "quality.json", quality)
                report["quality_passed"] = sum(row["passed"] for row in quality)
                report["quality_total"] = len(quality)
                if report["quality_passed"] != report["quality_total"]:
                    raise RuntimeError("Known-answer quality gate failed; timing data is retained for diagnosis")
            log_text = (run / "server.log").read_text(errors="replace")
            if profile["gpu"] != "none":
                counts = re.findall(r"\[CUDA\] tier glm53 (?:run|turn): device (\d+) of (\d+)", log_text)
                if not counts or not any(int(n) > 0 for n, _ in counts):
                    raise RuntimeError("No measured routed-expert GPU execution found in server logs")
                report["cuda_routed_expert_execution_verified"] = True
            report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if client is not None:
            owned_stop(client)
        stop_monitor.set()
        if thread is not None:
            thread.join(timeout=3)
        if proc is not None:
            owned_stop(proc)
        after = container_snapshot()
        report["protected_containers_unchanged"] = all(after.get(k) == v for k, v in baseline.items())
        report["memory_guard_triggered"] = guard_triggered.is_set()
        report["numerical_failure_detected"] = numerical_failure.is_set()
        report["minimum_global_memavailable_bytes"] = minimum_available[0]
        report["peak_process_group_rss_sum_bytes"] = peak_rss[0]
        report["rss_caution"] = "RSS sums may double-count shared pages; global MemAvailable is recorded separately"
        if not report["protected_containers_unchanged"] or guard_triggered.is_set() or numerical_failure.is_set():
            report["status"] = "failed"
        if numerical_failure.is_set():
            report["error"] = "Non-finite router logits: run stopped and excluded from usable-output performance"
        write_json(run / "run.json", report)
        write_json(ROOT / "runtime/last-run.json", {"run": str(run), "status": report["status"]})
        print(json.dumps({"run": str(run), "status": report["status"], "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
