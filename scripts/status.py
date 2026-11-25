#!/usr/bin/env python3
"""Print local download and experiment state without contacting external services."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for label, filename in [("download", "download-state.json"), ("campaign", "campaign-state.json"),
                        ("last_run", "last-run.json")]:
    path = root / "runtime" / filename
    if path.exists():
        value = json.loads(path.read_text())
        if "total_bytes" in value:
            value["progress_percent"] = round(100*value["bytes_accounted"]/value["total_bytes"], 2)
        print(label + ": " + json.dumps(value, indent=2))
