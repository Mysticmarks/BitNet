import asyncio
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bitnet import (  # noqa: E402
    GRPCSupervisorDriver,
    RaySupervisorDriver,
    RuntimeSupervisor,
    SchedulingPolicy,
)


class DummyRuntime:
    def __init__(self) -> None:
        self._env = {}
        self._cpu = 2

    @property
    def cpu_count(self) -> int:
        return self._cpu

    @property
    def environment(self):  # pragma: no cover - simple mapping
        return dict(self._env)


class FakeProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", delay: float = 0.01, returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self._delay = delay
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    async def wait(self):
        await asyncio.sleep(self._delay)
        return self.returncode

    def kill(self):  # pragma: no cover - defensive
        self.killed = True


def test_grpc_driver_with_tls_and_authentication():
    pytest.importorskip("grpc")

    asyncio.run(_run_grpc_test())


async def _run_grpc_test() -> None:
    runtime = DummyRuntime()
    supervisor = RuntimeSupervisor(
        runtime,
        concurrency=1,
        scheduling_policy=SchedulingPolicy.PRIORITY,
    )

    driver = GRPCSupervisorDriver(
        supervisor,
        host="127.0.0.1",
        port=0,
        use_local_credentials=True,
        auth_token="secret",
    )

    async def fake_exec(*args, **kwargs):
        return FakeProcess(stdout=b"hello\n")

    channel = None
    with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await supervisor.start()
        await driver.start()
        try:
            result = await driver.submit_task([sys.executable, "-c", "print('hello')"], options={"timeout": 1})
            assert "hello" in result.stdout

            channel = driver._grpc_aio.secure_channel(driver.address, driver._client_credentials)
            method = channel.unary_unary(driver._SUBMIT_METHOD)
            with pytest.raises(driver._grpc.RpcError) as excinfo:
                await method(json.dumps({"command": ["echo", "unauth"], "options": {}}).encode("utf-8"))
            assert excinfo.value.code() == driver._grpc.StatusCode.UNAUTHENTICATED
        finally:
            await driver.shutdown()
            await supervisor.shutdown()
            if channel is not None:
                await channel.close()


def test_ray_driver_executes_commands():
    ray = pytest.importorskip("ray")

    asyncio.run(_run_ray_test(ray))


async def _run_ray_test(ray_module) -> None:
    try:
        driver = RaySupervisorDriver(
            DummyRuntime,
            num_workers=1,
            shutdown_ray=True,
            tracer=None,
            init_kwargs={"local_mode": True},
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ray not available: {exc}")

    try:
        result = await driver.submit_task([sys.executable, "-c", "print('ray-driver')"], timeout=5)
        assert "ray-driver" in result.stdout
    finally:
        await driver.shutdown()
