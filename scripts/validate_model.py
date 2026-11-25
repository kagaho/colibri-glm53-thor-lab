#!/usr/bin/env python3
"""Check downloaded file sizes and native int4 tensor geometry, without loading weights."""
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]


def validate():
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    inventory = json.loads((ROOT / "configs/model-manifest.json").read_text())
    model = ROOT / cfg["model_directory"]
    manifest = json.loads((model / ".colibri-download.json").read_text())
    if manifest.get("revision") != cfg["model_revision"] or not manifest.get("complete"):
        raise RuntimeError("The pinned, checksum-verified download is not complete")
    for spec in inventory["files"]:
        path = model / spec["path"]
        if not path.is_file() or path.stat().st_size != spec["size"]:
            raise RuntimeError(f"Missing or wrong-sized file: {spec['path']}")
    tensors = {}
    shard_count = 0
    for shard in sorted(model.glob("*.safetensors")):
        shard_count += 1
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= size <= 64 * 2**20:
                raise RuntimeError(f"Unreasonable header length: {shard.name}")
            header = json.loads(stream.read(size))
        data_size = shard.stat().st_size - 8 - size
        ranges = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise RuntimeError(f"Duplicate tensor: {name}")
            start, end = spec["data_offsets"]
            if not 0 <= start <= end <= data_size:
                raise RuntimeError(f"Out-of-file tensor: {name}")
            ranges.append((start, end, name))
            tensors[name] = spec
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise RuntimeError(f"Overlapping tensors: {shard.name}")
    text = json.loads((model / "config.json").read_text())["text_config"]
    hidden, intermediate = text["hidden_size"], text["moe_intermediate_size"]
    pairs = 0
    for name, spec in tensors.items():
        match = re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if not match:
            continue
        rows, columns = ((hidden, intermediate) if match[1] == "down_proj" else (intermediate, hidden))
        scale = tensors.get(name + ".qs")
        if (spec["dtype"] != "U8" or math.prod(spec["shape"]) != rows*columns//2
                or not scale or scale["dtype"] != "F32"
                or math.prod(scale["shape"]) != rows*((columns+63)//64)):
            raise RuntimeError(f"Invalid native gs64 tensor pair: {name}")
        pairs += 1
    minimum = ((text["num_hidden_layers"] - text["first_k_dense_replace"])
               * text["n_routed_experts"] * 3)
    if shard_count != 62 or pairs < minimum:
        raise RuntimeError(f"Incomplete native tensor inventory: {shard_count} shards, {pairs} expert matrices")
    return {"status": "passed", "shards": shard_count, "tensors": len(tensors),
            "native_int4_expert_matrix_scale_pairs": pairs,
            "model_repository": cfg["model_repository"], "model_revision": cfg["model_revision"],
            "checksum_basis": "Colibri downloader verified each file against pinned HF SHA-256 or Git blob ID",
            "limitation": "Geometry and integrity checks do not establish model answer quality"}


if __name__ == "__main__":
    result = validate()
    destination = ROOT / "results/model-validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
