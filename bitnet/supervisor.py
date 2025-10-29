"""Asynchronous orchestration helpers for BitNet runtimes.

The synchronous :class:`bitnet.runtime.BitNetRuntime` API is intentionally
minimal so that it can be reused by lightweight CLI wrappers.  Production
deployments often need to coordinate multiple inference requests, apply
timeouts and collect structured telemetry which the synchronous API does not
provide.  This module introduces a supervisor that wraps the existing runtime
and adds:

* asyncio-compatible execution with bounded concurrency
* cooperative cancellation when a timeout is reached
* aggregated execution metadata (timings, standard streams)

The supervisor focuses on orchestration concerns without duplicating the
validation logic already covered by :class:`BitNetRuntime`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

from .runtime import (
    BitNetRuntime,
    RuntimeLaunchError,
    RuntimeTimeoutError,
)

__all__ = ["RuntimeSupervisor", "RuntimeResult"]


@dataclass(frozen=True)
class RuntimeResult:
    """Structured result for a supervised runtime execution."""

    command: Tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at: float
    completed_at: float

    @property
    def duration(self) -> float:
        return self.completed_at - self.started_at


class RuntimeSupervisor:
    """Async orchestrator for launching llama.cpp binaries.

    Parameters
    ----------
    runtime:
        Pre-configured :class:`BitNetRuntime` instance used to perform command
        validation and environment setup.
    concurrency:
        Maximum number of simultaneous process executions.  Defaults to the
        runtime's detected CPU count.
    capture_output:
        When ``True`` (the default) the supervisor captures standard output and
        error for each run.  Setting this to ``False`` allows the child process
        to inherit the parent's file descriptors.
    env:
        Optional environment overrides applied on top of the runtime's default
        mapping for all supervised executions.
    """

    def __init__(
        self,
        runtime: BitNetRuntime,
        *,
        concurrency: Optional[int] = None,
        capture_output: bool = True,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if concurrency is not None and concurrency <= 0:
            raise ValueError("concurrency must be a positive integer when provided")

        self._runtime = runtime
        limit = concurrency or max(1, runtime.cpu_count)
        self._semaphore = asyncio.Semaphore(limit)
        self._capture_output = capture_output
        self._base_env: MutableMapping[str, str] = dict(runtime.environment)
        if env:
            self._base_env.update(env)
        self._closed = False

    async def __aenter__(self) -> "RuntimeSupervisor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        """Prevent new launches from being scheduled."""

        self._closed = True

    # ------------------------------------------------------------------
    # Public orchestration helpers

    async def run_inference(self, *, timeout: Optional[float] = None, **kwargs) -> RuntimeResult:
        params = dict(kwargs)
        params.pop("dry_run", None)
        command = self._runtime.run_inference(dry_run=True, **params)
        return await self._launch(command, timeout=timeout)

    async def run_server(self, *, timeout: Optional[float] = None, **kwargs) -> RuntimeResult:
        params = dict(kwargs)
        params.pop("dry_run", None)
        command = self._runtime.run_server(dry_run=True, **params)
        return await self._launch(command, timeout=timeout)

    async def run_custom(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> RuntimeResult:
        """Execute a fully specified command under supervision."""

        return await self._launch(tuple(command), timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers

    async def _launch(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float],
    ) -> RuntimeResult:
        if self._closed:
            raise RuntimeError("RuntimeSupervisor is closed and cannot schedule new work")

        async with self._semaphore:
            started_at = time.time()
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE if self._capture_output else None,
                stderr=asyncio.subprocess.PIPE if self._capture_output else None,
                env=self._base_env,
            )

            try:
                if self._capture_output:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                else:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                    stdout_bytes = b""
                    stderr_bytes = b""
            except asyncio.TimeoutError as exc:
                process.kill()
                with contextlib.suppress(ProcessLookupError):
                    await process.wait()
                raise RuntimeTimeoutError(
                    f"Command timed out after {timeout} seconds: {' '.join(command)}"
                ) from exc

            completed_at = time.time()
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")

            if process.returncode != 0:
                raise RuntimeLaunchError(
                    f"Command exited with {process.returncode}: {' '.join(command)}"
                )

            return RuntimeResult(
                command=tuple(command),
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                started_at=started_at,
                completed_at=completed_at,
            )

