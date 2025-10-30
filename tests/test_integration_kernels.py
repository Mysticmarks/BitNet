import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SOURCE = PROJECT_ROOT / "tests" / "integration" / "cpp"


class MinimalIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.build_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _configure(self, *, enable_gpu: bool = True) -> None:
        command = [
            "cmake",
            "-S",
            str(INTEGRATION_SOURCE),
            "-B",
            str(self.build_dir),
            f"-DBITNET_INTEGRATION_ENABLE_GPU={'ON' if enable_gpu else 'OFF'}",
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    def _build(self, target: str) -> Path:
        subprocess.run([
            "cmake",
            "--build",
            str(self.build_dir),
            "--target",
            target,
            "-j",
            "2",
        ], check=True, cwd=PROJECT_ROOT)
        binary = self.build_dir / target
        if os.name == "nt":
            binary = binary.with_suffix(".exe")
        return binary

    def _run_binary(self, binary: Path) -> dict:
        completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        metrics = {}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
        return metrics

    def test_cpu_integration_model_executes(self):
        self._configure(enable_gpu=False)
        binary = self._build("bitnet_cpu_integration")
        metrics = self._run_binary(binary)
        self.assertGreater(metrics.get("token_per_second", 0.0), 0.0)
        self.assertGreater(metrics.get("latency_ms", 0.0), 0.0)

    def test_gpu_integration_model_executes_when_available(self):
        if shutil.which("nvcc") is None:
            self.skipTest("CUDA compiler not available in environment")
        self._configure(enable_gpu=True)
        binary = self._build("bitnet_gpu_integration")
        metrics = self._run_binary(binary)
        self.assertGreater(metrics.get("token_per_second", 0.0), 0.0)
        self.assertGreater(metrics.get("latency_ms", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
