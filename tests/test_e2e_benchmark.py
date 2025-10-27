import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
