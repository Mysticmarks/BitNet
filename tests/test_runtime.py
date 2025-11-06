import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet import (
    BitNetRuntime,
    RuntimeConfigurationError,
    RuntimeLaunchError,
    TelemetryEvent,
)  # noqa: E402


class BitNetRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        self.build_dir = root / "build"
        bin_dir = self.build_dir / "bin"
        bin_dir.mkdir(parents=True)
        # Create fake binaries
        for name in ("llama-cli", "llama-server"):
            path = bin_dir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        self.model_path = root / "model.gguf"
        self.model_path.write_text("fake")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_run_inference_builds_expected_command(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        command = runtime.run_inference(
            model=self.model_path,
            prompt="Hello",
            n_predict=32,
            ctx_size=1024,
            temperature=0.5,
            threads=2,
            batch_size=2,
            gpu_layers=4,
            conversation=True,
            dry_run=True,
            extra_args=["--top-k", "20"],
        )
        self.assertIn("-cnv", command)
        self.assertEqual(command[0], str(self.build_dir / "bin" / "llama-cli"))
        self.assertIn("--top-k", command)
        self.assertIn("20", command)
        ngl_index = command.index("-ngl")
        self.assertEqual(command[ngl_index + 1], "4")

    def test_run_server_clamps_thread_count_to_cpu(self):
        with mock.patch("os.cpu_count", return_value=4):
            runtime = BitNetRuntime(build_dir=self.build_dir)
        command = runtime.run_server(
            model=self.model_path,
            host="0.0.0.0",
            port=8000,
            prompt="System",
            threads=16,
            gpu_layers=2,
            dry_run=True,
        )
        t_index = command.index("-t")
        self.assertEqual(command[t_index + 1], "4")
        ngl_index = command.index("-ngl")
        self.assertEqual(command[ngl_index + 1], "2")

    def test_missing_prompt_raises_configuration_error(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        with self.assertRaises(RuntimeConfigurationError):
            runtime.run_inference(
                model=self.model_path,
                prompt="   ",
                dry_run=True,
            )

    def test_negative_gpu_layers_raises(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        with self.assertRaises(RuntimeConfigurationError):
            runtime.run_inference(
                model=self.model_path,
                prompt="Hello",
                gpu_layers=-1,
                dry_run=True,
            )

    def test_diagnostics_reports_binary_presence(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        report = runtime.diagnostics(model=self.model_path)
        info = report.as_dict()
        self.assertTrue(info["binaries"]["llama-cli"])
        self.assertTrue(info["binaries"]["llama-server"])
        self.assertTrue(info["model_present"])
        self.assertIn("probes", info)
        self.assertIn("disk", info["probes"])
        self.assertIn("memory", info["probes"])
        self.assertIn("gpu", info["probes"])
        self.assertIn("sandbox", info["probes"])

    def test_missing_binaries_raise_informative_error(self):
        runtime = BitNetRuntime(build_dir=self.build_dir / "missing")
        with self.assertRaises(RuntimeConfigurationError) as ctx:
            runtime.run_inference(
                model=self.model_path,
                prompt="hi",
                dry_run=True,
            )
        self.assertIn("Unable to locate", str(ctx.exception))

    def test_telemetry_sink_receives_events(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        events: List[TelemetryEvent] = []
        runtime.add_telemetry_sink(events.append)

        script = self.build_dir.parent / "echo.py"
        script.write_text("print('hello')\n")
        runtime._execute([sys.executable, str(script)])

        self.assertTrue(any(evt.name == "runtime.execute.start" for evt in events))
        self.assertTrue(runtime.recent_events())

    def test_execute_retries_and_metrics(self):
        runtime = BitNetRuntime(
            build_dir=self.build_dir,
            max_retries=1,
            retry_backoff_base=0.0,
        )
        script = self.build_dir.parent / "flaky.py"
        state_path = self.build_dir.parent / "flaky_state.txt"
        script.write_text(
            "import pathlib, sys\n"
            "state = pathlib.Path(sys.argv[1])\n"
            "count = int(state.read_text()) if state.exists() else 0\n"
            "count += 1\n"
            "state.write_text(str(count))\n"
            "if count < 2:\n"
            "    print('fail', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "print('ok')\n"
        )

        chunks: List[str] = []

        def consumer(stream: str, data: str) -> None:
            chunks.append(f"{stream}:{data.strip()}")

        returncode = runtime._execute(
            [sys.executable, str(script), str(state_path)],
            stream_consumer=consumer,
        )
        self.assertEqual(returncode, 0)
        metrics = runtime.metrics_snapshot()
        self.assertGreaterEqual(metrics["executions.retry"], 1)
        self.assertIn("stdout:ok", {chunk.replace(" ", "") for chunk in chunks})

    def test_circuit_breaker_blocks_after_failures(self):
        runtime = BitNetRuntime(
            build_dir=self.build_dir,
            max_retries=0,
            retry_backoff_base=0.0,
            circuit_breaker_threshold=1,
            circuit_breaker_reset=5.0,
        )
        script = self.build_dir.parent / "always_fail.py"
        script.write_text("import sys; sys.exit(1)\n")

        with self.assertRaises(RuntimeLaunchError):
            runtime._execute([sys.executable, str(script)])

        with self.assertRaises(RuntimeLaunchError) as ctx:
            runtime._execute([sys.executable, str(script)])
        self.assertIn("circuit", str(ctx.exception).lower())
        metrics = runtime.metrics_snapshot()
        self.assertGreaterEqual(metrics["executions.circuit_open"], 1)

    def test_prometheus_metrics_include_probe_status(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        metrics = runtime.prometheus_metrics()
        self.assertIn("bitnet_runtime_executions_total", metrics)
        self.assertIn("bitnet_runtime_probe_status", metrics)

    def test_opentelemetry_metrics_exports_values(self):
        runtime = BitNetRuntime(build_dir=self.build_dir)
        events = runtime.opentelemetry_metrics()
        names = {entry["name"] for entry in events}
        self.assertIn("bitnet.runtime.executions.total", names)


if __name__ == "__main__":
    unittest.main()
