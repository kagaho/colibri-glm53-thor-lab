import copy
import importlib.util
import json
from pathlib import Path
import struct
import socket
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


validator = module("validate_model")
summarizer = module("summarize")
runner = module("run_experiment")


class PortPreflightTests(unittest.TestCase):
    def test_rejects_live_listener(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            with self.assertRaises(OSError):
                runner.check_port_available(server.getsockname()[1])

    def test_accepts_recently_closed_server_port(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen()
            with socket.create_connection(("127.0.0.1", port)) as client:
                conn, _ = server.accept()
                with conn:
                    conn.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        runner.check_port_available(port)


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("configs", "models/native"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.model = self.root / "models/native"
        config = {"model_directory": "models/native", "model_repository": "fixture/native",
                  "model_revision": "a"*40}
        (self.root / "configs/experiment.json").write_text(json.dumps(config))
        text = {"hidden_size": 64, "moe_intermediate_size": 64, "num_hidden_layers": 2,
                "first_k_dense_replace": 1, "n_routed_experts": 1}
        (self.model / "config.json").write_text(json.dumps({"text_config": text}))
        self.header = {}
        offset = 0
        for kind in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.language_model.layers.1.mlp.experts.0.{kind}.weight"
            for suffix, dtype, count, bytes_per in (("", "U8", 2048, 1), (".qs", "F32", 64, 4)):
                self.header[name + suffix] = {"dtype": dtype, "shape": [count],
                    "data_offsets": [offset, offset + count*bytes_per]}
                offset += count*bytes_per
        self.payload_size = offset
        self.write_shards()
        (self.model / ".colibri-download.json").write_text(json.dumps({
            "revision": "a"*40, "complete": True}))

    def tearDown(self):
        self.temp.cleanup()

    def write_shards(self, header=None):
        for index in range(1, 63):
            data = json.dumps((self.header if header is None else header) if index == 1 else {}).encode()
            path = self.model / f"model-{index:05}-of-00062.safetensors"
            path.write_bytes(struct.pack("<Q", len(data)) + data + (b"\0"*self.payload_size if index == 1 else b""))
        files = [{"path": p.name, "size": p.stat().st_size} for p in self.model.iterdir() if p.is_file()]
        (self.root / "configs/model-manifest.json").write_text(json.dumps({"files": files}))

    def check(self):
        with patch.object(validator, "ROOT", self.root):
            return validator.validate()

    def test_accepts_complete_native_geometry(self):
        self.assertEqual(self.check()["native_int4_expert_matrix_scale_pairs"], 3)

    def test_rejects_incomplete_download(self):
        (self.model / ".colibri-download.json").write_text(json.dumps({"revision": "a"*40, "complete": False}))
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            self.check()

    def test_rejects_wrong_model_revision(self):
        (self.model / ".colibri-download.json").write_text(json.dumps({"revision": "b"*40, "complete": True}))
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            self.check()

    def test_rejects_missing_shard(self):
        (self.model / "model-00062-of-00062.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "wrong-sized"):
            self.check()

    def test_rejects_out_of_file_offsets(self):
        header = copy.deepcopy(self.header)
        next(iter(header.values()))["data_offsets"][1] = 10**9
        self.write_shards(header)
        with self.assertRaisesRegex(RuntimeError, "Out-of-file"):
            self.check()

    def test_rejects_wrong_scale_geometry(self):
        header = copy.deepcopy(self.header)
        scale = next(v for k, v in header.items() if k.endswith(".qs"))
        scale["shape"] = [63]
        self.write_shards(header)
        with self.assertRaisesRegex(RuntimeError, "Invalid native"):
            self.check()


class PublicSummaryTests(unittest.TestCase):
    def test_reports_clamped_tier_from_observed_log(self):
        text = "[CUDA] tier glm53: on, NVIDIA Thor, budget 1.54 GiB = 56 experts of 13.5 MiB"
        result = summarizer.parse_device_tier(text, 16)
        self.assertEqual(result["reported_budget_gib"], 1.54)
        self.assertEqual(result["expert_capacity"], 56)
        self.assertTrue(result["clamped_below_requested"])

    def test_reports_full_requested_device_budget(self):
        text = "[CUDA] tier glm53: on, NVIDIA Thor, budget 16.00 GiB = 1088 experts of 13.5 MiB"
        result = summarizer.parse_device_tier(text, 16)
        self.assertEqual(result["reported_budget_gib"], 16)
        self.assertFalse(result["clamped_below_requested"])

    def test_missing_placement_is_unknown(self):
        self.assertIsNone(summarizer.parse_device_tier("CPU-only engine", 0))

    def test_protocol_success_with_nonfinite_logits_is_invalid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "server.log").write_text("[router] non-finite logits at layer 3\n")
            row = {"status": "complete", "profile": "fixture", "profile_settings": {},
                "cache_state": "uncontrolled", "io_mode": "buffered", "workload_sha256": "abc",
                "http_report": {"summary": {"succeeded": 4}, "requests": []}}
            result = summarizer.summarize_run(row, folder)
            self.assertEqual(result["status"], "invalid_output")
            self.assertTrue(result["numerical_failure_detected"])
            self.assertEqual(result["http_summary"]["succeeded"], 4)

    def test_failure_is_retained_without_private_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"status": "failed", "profile": "fixture", "error": "HTTP failed",
                "profile_settings": {}, "cache_state": "uncontrolled", "io_mode": "buffered",
                "workload_sha256": "abc", "command": ["/private/home/path"],
                "environment_overrides": {"TOKEN": "secret"},
                "http_report": {"summary": {"failed": 1}, "requests": [{"success": False}]}}
            result = summarizer.summarize_run(row, Path(directory))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["http_summary"]["failed"], 1)
            self.assertFalse(result["http_attempts"][0]["success"])
            self.assertNotIn("secret", json.dumps(result))
            self.assertNotIn("/private/home", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
