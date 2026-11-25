#!/usr/bin/env python3
"""Download one immutable native checkpoint using Colibri's verified downloader."""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/colibri/c"))
import setup_download as download


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    (ROOT / "runtime").mkdir(exist_ok=True)
    lease = (ROOT / "runtime/download.lock").open("w")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A model download is already running in this workspace")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    repo, revision = cfg["model_repository"], cfg["model_revision"]
    files = download.list_repo_files(repo, revision)
    if not files or sum(f["path"].endswith(".safetensors") for f in files) != 62:
        raise RuntimeError("Expected exactly 62 native safetensors shards")
    manifest = {"repository": repo, "revision": revision, "files": files,
                "total_bytes": sum(f["size"] for f in files)}
    save_json(ROOT / "configs/model-manifest.json", manifest)
    if args.manifest_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
        return
    model = ROOT / cfg["model_directory"]
    model.mkdir(parents=True, exist_ok=True)
    remaining = manifest["total_bytes"] - download.bytes_present(str(model), files)
    if shutil.disk_usage(model).free < remaining + cfg["minimum_free_disk_gib"] * 2**30:
        raise RuntimeError("Insufficient disk space including the configured free-space reserve")
    lock = threading.Lock()
    started = time.monotonic()
    progress = {f["path"]: 0 for f in files}
    complete = []
    last_save = [0.0]
    state_path = ROOT / "runtime/download-state.json"

    def state(status, error=None, force=False):
        now = time.monotonic()
        if not force and now - last_save[0] < 2:
            return
        last_save[0] = now
        save_json(state_path, {"status": status, "repository": repo, "revision": revision,
            "pid": os.getpid(), "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": now-started, "bytes_accounted": sum(progress.values()),
            "total_bytes": manifest["total_bytes"], "verified_files": len(complete),
            "total_files": len(files), "error": error})

    def fetch(spec):
        def on_bytes(have, total, fresh=0):
            with lock:
                progress[spec["path"]] = have
                state("downloading")
        destination = download.safe_join(str(model), spec["path"])
        result = download.download_file(download.file_url(repo, revision, spec["path"]),
            destination, spec, progress=on_bytes, verify=True, retries=8)
        with lock:
            progress[spec["path"]] = spec["size"]
            complete.append(spec["path"])
            state("downloading", force=True)
            print(json.dumps({"file": spec["path"], "result": result,
                "verified_files": len(complete), "bytes_accounted": sum(progress.values())}), flush=True)

    state("starting", force=True)
    try:
        ordered = sorted(files, key=lambda f: (f["size"] > 64*2**20, f["size"], f["path"]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fetch, ordered))
        save_json(model / ".colibri-download.json", {
            "repo": repo, "revision": revision, "complete": True,
            "files": [{"path": f["path"], "size": f["size"]} for f in files]})
        with lock:
            state("complete", force=True)
        print("All model files verified.", flush=True)
    except Exception as exc:
        with lock:
            state("error", str(exc), force=True)
        raise


if __name__ == "__main__":
    main()
