import asyncio
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet import RuntimeSupervisor

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except Exception as exc:  # pragma: no cover
    raise unittest.SkipTest(f"hypothesis not available: {exc}")


class StubRuntime:
    def __init__(self):
        self._env = {}
        self._cpu = 4

    @property
    def cpu_count(self):
        return self._cpu

    @property
    def environment(self):
        return self._env

    def run_inference(self, *, dry_run: bool, **kwargs):  # pragma: no cover - not used in property tests
        raise AssertionError("unexpected call")

    def run_server(self, *, dry_run: bool, **kwargs):  # pragma: no cover - not used in property tests
        raise AssertionError("unexpected call")


class SupervisorPropertyTests(unittest.TestCase):
    @settings(max_examples=25)
    @given(
        concurrency=st.integers(min_value=1, max_value=3),
        task_durations=st.lists(st.floats(min_value=0.0, max_value=0.05), min_size=1, max_size=5),
    )
    def test_concurrency_never_exceeds_limit(self, concurrency, task_durations):
        runtime = StubRuntime()
        supervisor = RuntimeSupervisor(runtime, concurrency=concurrency)

        active_counts = []
        durations_iter = iter(task_durations)

        async def fake_create(*command, **kwargs):
            try:
                delay = next(durations_iter)
            except StopIteration:
                delay = 0.0
            active_counts.append(concurrency - supervisor._semaphore._value)  # type: ignore[attr-defined]
            return FakeProcess(delay=delay)

        async def run_scenario():
            with unittest.mock.patch("asyncio.create_subprocess_exec", side_effect=fake_create):
                await asyncio.gather(
                    *[
                        supervisor.run_custom(("/bin/echo", "ok"), timeout=None)
                        for _ in task_durations
                    ]
                )

        asyncio.run(run_scenario())
        for value in active_counts:
            self.assertLessEqual(value, concurrency)


class FakeProcess:
    def __init__(self, delay: float = 0.0):
        self.returncode = 0
        self._delay = delay

    async def communicate(self):
        await asyncio.sleep(self._delay)
        return b"ok", b""

    async def wait(self):
        await asyncio.sleep(self._delay)
        return self.returncode

    def kill(self):
        pass


if __name__ == "__main__":
    unittest.main()
