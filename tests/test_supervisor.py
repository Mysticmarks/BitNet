import asyncio
import sys
import unittest
from pathlib import Path
from typing import Sequence
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet import RuntimeSupervisor, RuntimeTimeoutError


class StubRuntime:
    def __init__(self, inference_command: Sequence[str], server_command: Sequence[str]):
        self._inference_command = tuple(inference_command)
        self._server_command = tuple(server_command)
        self._env = {}
        self._cpu = 2
        self.last_inference_kwargs = None

    @property
    def cpu_count(self) -> int:
        return self._cpu

    @property
    def environment(self):
        return self._env

    @property
    def inference_command(self):
        return self._inference_command

    def run_inference(self, *, dry_run: bool, **kwargs):
        assert dry_run is True
        self.last_inference_kwargs = kwargs
        return self._inference_command

    def run_server(self, *, dry_run: bool, **kwargs):
        assert dry_run is True
        return self._server_command


class FakeProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", delay: float = 0.0, returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self._delay = delay
        self.returncode = returncode
        self.killed = False
        self.wait_called = False

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    async def wait(self):
        self.wait_called = True
        if self._delay:
            await asyncio.sleep(self._delay)
        return self.returncode

    def kill(self):
        self.killed = True


class RuntimeSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_inference_collects_output(self):
        runtime = StubRuntime(("fake", "--run"), ("fake-server",))
        supervisor = RuntimeSupervisor(runtime, concurrency=2)
        expected_command = runtime.inference_command

        async def fake_exec(*args, **kwargs):
            self.assertEqual(tuple(args), expected_command)
            self.assertIn("env", kwargs)
            return FakeProcess(stdout=b"hello", stderr=b"", delay=0.01)

        with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await supervisor.run_inference(model="m", prompt="p")

        self.assertEqual(result.stdout, "hello")
        self.assertGreaterEqual(result.duration, 0.0)

    async def test_concurrency_is_respected(self):
        runtime = StubRuntime(("fake",), ("fake-server",))
        supervisor = RuntimeSupervisor(runtime, concurrency=1)
        call_times = []

        async def fake_exec(*args, **kwargs):
            call_times.append(asyncio.get_running_loop().time())
            return FakeProcess(delay=0.05)

        with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await asyncio.gather(
                supervisor.run_inference(model="m", prompt="p"),
                supervisor.run_inference(model="m", prompt="p"),
            )

        self.assertGreaterEqual(call_times[1] - call_times[0], 0.045)

    async def test_timeout_kills_process(self):
        runtime = StubRuntime(("fake",), ("fake-server",))
        supervisor = RuntimeSupervisor(runtime, concurrency=2)
        processes = []

        async def fake_exec(*args, **kwargs):
            proc = FakeProcess(delay=0.2)
            processes.append(proc)
            return proc

        with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with self.assertRaises(RuntimeTimeoutError):
                await supervisor.run_inference(model="m", prompt="p", timeout=0.05)

        self.assertTrue(processes[0].killed)
        self.assertTrue(processes[0].wait_called)


if __name__ == "__main__":
    unittest.main()
