from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import e2e_benchmark


class FindBenchmarkBinaryTests(unittest.TestCase):
    def test_finds_unix_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            bench_path = build_dir / "bin" / "llama-bench"
            bench_path.parent.mkdir(parents=True, exist_ok=True)
            bench_path.touch()

            result = e2e_benchmark.find_benchmark_binary(build_dir, system="Linux")
            self.assertEqual(result, bench_path)

    def test_prefers_release_binary_on_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir)
            release_path = build_dir / "bin" / "Release" / "llama-bench.exe"
            release_path.parent.mkdir(parents=True, exist_ok=True)
            release_path.touch()

            result = e2e_benchmark.find_benchmark_binary(build_dir, system="Windows")
            self.assertEqual(result, release_path)

    def test_raises_when_binary_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                e2e_benchmark.find_benchmark_binary(Path(tmpdir), system="Linux")


class RunCommandTests(unittest.TestCase):
    def test_run_command_writes_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            result = e2e_benchmark.run_command(
                [sys.executable, "-c", "print('hello')"],
                log_dir=log_dir,
                log_step="greeting",
            )
            self.assertEqual(result.returncode, 0)
            log_path = log_dir / "greeting.log"
            self.assertTrue(log_path.exists())
            self.assertIn("hello", log_path.read_text())

    def test_run_command_requires_log_dir(self):
        with self.assertRaises(ValueError):
            e2e_benchmark.run_command([sys.executable, "-c", "print('hello')"], log_step="step")


class BuildBenchmarkCommandTests(unittest.TestCase):
    def test_build_benchmark_command_contains_expected_flags(self):
        bench_path = Path("/tmp/llama-bench")
        command = e2e_benchmark.build_benchmark_command(
            bench_path,
            model="model.gguf",
            n_token=128,
            n_prompt=256,
            threads=4,
        )
        self.assertEqual(
            command,
            [
                str(bench_path),
                "-m",
                "model.gguf",
                "-n",
                "128",
                "-ngl",
                "0",
                "-b",
                "1",
                "-t",
                "4",
                "-p",
                "256",
                "-r",
                "5",
            ],
        )


class LlamaBenchEndToEndTests(unittest.TestCase):
    def _prepare_dummy_model(self, base_dir: Path) -> tuple[Path, bool]:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "utils" / "generate-dummy-bitnet-model.py"

        model_dir = base_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "architectures": ["BitnetForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
        }
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model_dir / "tokenizer.model").write_bytes(b"placeholder-tokenizer")
        (model_dir / "pytorch_model.bin").write_bytes(b"")

        outfile = base_dir / "dummy.gguf"
        command = [
            sys.executable,
            str(script),
            "--vocab-only",
            "--model-size",
            "125M",
            "--model-name",
            "dummy",
            "--outfile",
            str(outfile),
            str(model_dir),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return outfile, False
        except (subprocess.CalledProcessError, FileNotFoundError):
            placeholder = base_dir / "placeholder.gguf"
            placeholder.write_text("synthetic gguf", encoding="utf-8")
            return placeholder, True

    def test_llama_bench_smoke_with_dummy_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path, used_fallback = self._prepare_dummy_model(tmp_path)

            build_dir = tmp_path / "build"
            bench_path = build_dir / "bin" / "llama-bench"
            bench_path.parent.mkdir(parents=True, exist_ok=True)
            bench_script = textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import argparse
                import sys
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("-m", dest="model")
                parser.add_argument("-n", dest="tokens", type=int)
                parser.add_argument("-ngl", dest="gpu_layers", type=int)
                parser.add_argument("-b", dest="batch", type=int)
                parser.add_argument("-t", dest="threads", type=int)
                parser.add_argument("-p", dest="prompt", type=int)
                parser.add_argument("-r", dest="runs", type=int)
                args, _ = parser.parse_known_args()

                model_path = Path(args.model)
                if not model_path.exists():
                    print("model missing", file=sys.stderr)
                    sys.exit(1)

                magnitude = max(1, len(model_path.read_bytes()))
                tokens_per_second = magnitude / max(args.threads, 1)
                latency_ms = 1000.0 / tokens_per_second

                print(f"model={model_path}")
                print(f"tok/s={tokens_per_second:.2f}")
                print(f"latency_ms={latency_ms:.2f}")
                """
            )
            bench_path.write_text(bench_script, encoding="utf-8")
            os.chmod(bench_path, 0o755)

            command = e2e_benchmark.build_benchmark_command(
                bench_path,
                model=str(model_path),
                n_token=32,
                n_prompt=64,
                threads=2,
            )

            log_dir = tmp_path / "logs"
            result = e2e_benchmark.run_command(command, log_dir=log_dir, log_step="benchmark")
            self.assertEqual(result.returncode, 0)

            log_text = (log_dir / "benchmark.log").read_text(encoding="utf-8")
            self.assertIn("tok/s=", log_text)
            self.assertIn("latency_ms=", log_text)

            metrics = {}
            for line in log_text.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        continue

            self.assertGreater(metrics.get("tok/s", 0.0), 0.0)
            self.assertGreater(metrics.get("latency_ms", 0.0), 0.0)
            if used_fallback:
                self.assertTrue(model_path.name.endswith(".gguf"))


if __name__ == "__main__":
    unittest.main()
