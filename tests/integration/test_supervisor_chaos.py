import asyncio
import random
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bitnet import RuntimeSupervisor, SchedulingPolicy  # noqa: E402


class DummyRuntime:
    def __init__(self) -> None:
        self._env = {}
        self._cpu = 4

    @property
    def cpu_count(self) -> int:
        return self._cpu

    @property
    def environment(self):  # pragma: no cover - trivial mapping
        return dict(self._env)


class FakeProcess:
    def __init__(self, *, delay: float, succeed: bool) -> None:
        self._delay = delay
        self._succeed = succeed
        self.returncode = 0 if succeed else 1
        self._stdout = b"ok\n" if succeed else b""
        self._stderr = b"" if succeed else b"boom\n"
        self._killed = False

    async def communicate(self):
        await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    async def wait(self):
        await asyncio.sleep(self._delay)
        return self.returncode

    def kill(self):
        self._killed = True


@pytest.mark.asyncio
async def test_supervisor_chaos_soak_runs_under_failures():
    random.seed(42)
    runtime = DummyRuntime()
    supervisor = RuntimeSupervisor(
        runtime,
        concurrency=3,
        default_retry=1,
        default_retry_backoff=0.01,
        scheduling_policy=SchedulingPolicy.RESOURCE_AWARE,
        resource_limits={"gpu": 2},
    )

    async def fake_exec(*args, **kwargs):
        delay = random.uniform(0.005, 0.02)
        succeed = random.random() > 0.35
        return FakeProcess(delay=delay, succeed=succeed)

    successes = 0
    failures = 0

    with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await supervisor.start()
        try:
            tasks = []
            for idx in range(20):
                resources = {"gpu": 1 if idx % 3 == 0 else 0}
                coro = supervisor.run_custom(
                    ["cmd", str(idx)],
                    timeout=0.2,
                    resources=resources,
                )
                tasks.append(coro)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for item in results:
                if isinstance(item, Exception):
                    failures += 1
                else:
                    successes += 1
            assert successes > 0
            assert failures > 0

            # run a follow-up batch to ensure supervisor recovered
            follow_up = await asyncio.gather(
                *[
                    supervisor.run_custom(["cmd", "follow"], timeout=0.2, resources={"gpu": 1})
                    for _ in range(2)
                ]
            )
            assert all(result.returncode == 0 for result in follow_up)
        finally:
            await supervisor.shutdown()
