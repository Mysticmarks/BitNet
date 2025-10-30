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
import heapq
import itertools
import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .runtime import (
    BitNetRuntime,
    RuntimeLaunchError,
    RuntimeTimeoutError,
)

__all__ = [
    "RuntimeSupervisor",
    "RuntimeResult",
    "StreamEvent",
    "GRPCSupervisorDriver",
    "RaySupervisorDriver",
    "CelerySupervisorDriver",
]


@dataclass(frozen=True)
class RuntimeResult:
    """Structured result for a supervised runtime execution."""

    command: Tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at: float
    completed_at: float
    backend: str
    attempts: int = 1
    fallback_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.completed_at - self.started_at


@dataclass(frozen=True)
class StreamEvent:
    """Streaming telemetry emitted while a command is running."""

    task_id: str
    stream: str
    payload: str
    timestamp: float
    final: bool = False
    result: Optional[RuntimeResult] = None


@dataclass
class ScheduledTask:
    """Envelope for a task awaiting execution."""

    task_id: str
    command: Tuple[str, ...]
    priority: int
    weight: int
    timeout: Optional[float]
    backend: str
    retries: int
    retry_backoff: float
    fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]]
    metadata: Mapping[str, Any]
    stream_queue: Optional["asyncio.Queue[StreamEvent]"] = None
    future: Optional["asyncio.Future[RuntimeResult]"] = None


class CircuitBreaker:
    """Simple circuit breaker for backend isolation."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time
        self._failures = 0
        self._opened_at: Optional[float] = None

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = time.time()

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.time() - self._opened_at) >= self._recovery_time:
            # allow a trial execution
            self._failures = 0
            self._opened_at = None
            return False
        return True



class RuntimeSupervisor:
    """Async orchestrator for launching llama.cpp binaries.

    Parameters
    ----------
    runtime:
        Pre-configured :class:`BitNetRuntime` instance used to perform command
        validation and environment setup.
    concurrency:
        Maximum number of simultaneous process executions across all backends.
        Defaults to the runtime's detected CPU count.
    capture_output:
        When ``True`` (the default) the supervisor captures standard output and
        error for each run.  Setting this to ``False`` allows the child process
        to inherit the parent's file descriptors.
    env:
        Optional environment overrides applied on top of the runtime's default
        mapping for all supervised executions.
    backend_limits:
        Optional mapping that defines bulkhead isolation semaphores for each
        backend.  Backends not listed fall back to the global concurrency.
    backend_weights:
        Weighting used by the fair scheduler.  A backend with a higher weight
        receives proportionally more scheduling opportunities.
    default_retry:
        Number of retry attempts applied when a task does not override the
        retry policy.
    """

    def __init__(
        self,
        runtime: BitNetRuntime,
        *,
        concurrency: Optional[int] = None,
        capture_output: bool = True,
        env: Optional[Mapping[str, str]] = None,
        backend_limits: Optional[Mapping[str, int]] = None,
        backend_weights: Optional[Mapping[str, int]] = None,
        default_retry: int = 0,
        default_retry_backoff: float = 0.5,
    ) -> None:
        if concurrency is not None and concurrency <= 0:
            raise ValueError("concurrency must be a positive integer when provided")

        self._runtime = runtime
        limit = concurrency or max(1, runtime.cpu_count)
        self._global_limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self._capture_output = capture_output
        self._base_env: MutableMapping[str, str] = dict(runtime.environment)
        if env:
            self._base_env.update(env)

        self._default_retry = max(0, default_retry)
        self._default_retry_backoff = max(0.0, default_retry_backoff)

        self._backend_limits = dict(backend_limits or {})
        self._backend_weights = {k: max(1, v) for k, v in (backend_weights or {}).items()}
        self._bulkheads: Dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        self._task_queue: List[Tuple[Tuple[int, float, int], ScheduledTask]] = []
        self._counter = itertools.count()
        self._backend_finish: Dict[str, float] = {}
        self._closed = False
        self._active_tasks = 0
        self._queue_event = asyncio.Event()
        self._dispatcher: Optional[asyncio.Task[None]] = None
        self._active_per_backend: Dict[str, int] = defaultdict(int)

        self._startup_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []

        self._default_backend = "default"

    # ------------------------------------------------------------------
    # Lifecycle management

    async def __aenter__(self) -> "RuntimeSupervisor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.shutdown()

    async def start(self) -> None:
        if self._dispatcher is not None:
            return

        for callback in self._startup_callbacks:
            await callback()

        self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def shutdown(self, *, drain: bool = True) -> None:
        self._closed = True
        if drain:
            await self.drain()

        if self._dispatcher is not None:
            self._dispatcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher
            self._dispatcher = None

        for callback in self._shutdown_callbacks:
            await callback()

    def close(self) -> None:
        """Prevent new tasks from being scheduled without awaiting shutdown."""

        self._closed = True

    def register_startup_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._startup_callbacks.append(callback)

    def register_shutdown_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._shutdown_callbacks.append(callback)

    async def drain(self) -> None:
        """Wait for the scheduler queue to empty and active tasks to finish."""

        while True:
            if not self._task_queue and self._active_tasks == 0:
                return
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Introspection helpers

    @property
    def queue_depths(self) -> Mapping[str, Dict[str, int]]:
        depths: Dict[str, Dict[str, int]] = {}
        for _, task in self._task_queue:
            bucket = depths.setdefault(task.backend, {"queued": 0, "running": 0})
            bucket["queued"] += 1
        for backend, count in self._active_per_backend.items():
            bucket = depths.setdefault(backend, {"queued": 0, "running": 0})
            bucket["running"] = count
        return depths

    # ------------------------------------------------------------------
    # Public orchestration helpers

    async def run_inference(
        self,
        *,
        timeout: Optional[float] = None,
        priority: int = 0,
        weight: int = 1,
        backend: Optional[str] = None,
        retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> RuntimeResult:
        params = dict(kwargs)
        params.pop("dry_run", None)
        command = self._runtime.run_inference(dry_run=True, **params)
        return await self._launch(
            tuple(command),
            timeout=timeout,
            priority=priority,
            weight=weight,
            backend=backend,
            retries=retries,
            retry_backoff=retry_backoff,
            fallback=fallback,
            metadata=metadata or {},
        )

    async def run_server(
        self,
        *,
        timeout: Optional[float] = None,
        priority: int = 0,
        weight: int = 1,
        backend: Optional[str] = None,
        retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> RuntimeResult:
        params = dict(kwargs)
        params.pop("dry_run", None)
        command = self._runtime.run_server(dry_run=True, **params)
        return await self._launch(
            tuple(command),
            timeout=timeout,
            priority=priority,
            weight=weight,
            backend=backend,
            retries=retries,
            retry_backoff=retry_backoff,
            fallback=fallback,
            metadata=metadata or {},
        )

    async def run_custom(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float] = None,
        priority: int = 0,
        weight: int = 1,
        backend: Optional[str] = None,
        retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RuntimeResult:
        """Execute a fully specified command under supervision."""

        return await self._launch(
            tuple(command),
            timeout=timeout,
            priority=priority,
            weight=weight,
            backend=backend,
            retries=retries,
            retry_backoff=retry_backoff,
            fallback=fallback,
            metadata=metadata or {},
        )

    async def stream_run_custom(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float] = None,
        priority: int = 0,
        weight: int = 1,
        backend: Optional[str] = None,
        retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a command and yield :class:`StreamEvent` updates."""

        if self._closed:
            raise RuntimeError("RuntimeSupervisor is closed and cannot schedule new work")

        queue: "asyncio.Queue[StreamEvent]" = asyncio.Queue()
        task_id = uuid.uuid4().hex

        await self._enqueue_task(
            ScheduledTask(
                task_id=task_id,
                command=tuple(command),
                priority=priority,
                weight=max(1, weight),
                timeout=timeout,
                backend=backend or self._default_backend,
                retries=self._default_retry if retries is None else max(0, retries),
                retry_backoff=
                    self._default_retry_backoff
                    if retry_backoff is None
                    else max(0.0, retry_backoff),
                fallback=fallback,
                metadata=metadata or {},
                stream_queue=queue,
            )
        )

        async def iterator() -> AsyncIterator[StreamEvent]:
            while True:
                event = await queue.get()
                yield event
                if event.final:
                    return

        return iterator()

    # ------------------------------------------------------------------
    # Internal helpers

    async def _launch(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float],
        priority: int,
        weight: int,
        backend: Optional[str],
        retries: Optional[int],
        retry_backoff: Optional[float],
        fallback: Optional[Union[Sequence[str], Callable[[], Awaitable[RuntimeResult]]]],
        metadata: Mapping[str, Any],
    ) -> RuntimeResult:
        if self._closed:
            raise RuntimeError("RuntimeSupervisor is closed and cannot schedule new work")

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[RuntimeResult]" = loop.create_future()
        task_id = uuid.uuid4().hex

        scheduled = ScheduledTask(
            task_id=task_id,
            command=tuple(command),
            priority=priority,
            weight=max(1, weight),
            timeout=timeout,
            backend=backend or self._default_backend,
            retries=self._default_retry if retries is None else max(0, retries),
            retry_backoff=
                self._default_retry_backoff
                if retry_backoff is None
                else max(0.0, retry_backoff),
            fallback=fallback,
            metadata=metadata,
            future=future,
        )

        await self._enqueue_task(scheduled)
        return await future

    async def _enqueue_task(self, task: ScheduledTask) -> None:
        if self._dispatcher is None:
            await self.start()

        backend = task.backend
        backend_weight = self._backend_weights.get(backend, 1)
        effective_weight = max(1, backend_weight * task.weight)
        start = max(self._backend_finish.get(backend, 0.0), 0.0)
        finish = start + (1.0 / effective_weight)
        self._backend_finish[backend] = finish
        key = (task.priority, finish, next(self._counter))

        heapq.heappush(self._task_queue, (key, task))
        self._queue_event.set()

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                await self._queue_event.wait()

                while self._task_queue:
                    key, task = heapq.heappop(self._task_queue)
                    breaker = self._get_breaker(task.backend)
                    if breaker.is_open():
                        if task.future:
                            task.future.set_exception(
                                RuntimeError(
                                    f"Circuit open for backend '{task.backend}', rejecting task"
                                )
                            )
                        elif task.stream_queue is not None:
                            await task.stream_queue.put(
                                StreamEvent(
                                    task_id=task.task_id,
                                    stream="supervisor",
                                    payload=
                                        f"Circuit open for backend '{task.backend}', rejecting task",
                                    timestamp=time.time(),
                                    final=True,
                                )
                            )
                        continue

                    asyncio.create_task(self._run_task(task))

                self._queue_event.clear()
        except asyncio.CancelledError:
            pass

    def _get_bulkhead(self, backend: str) -> asyncio.Semaphore:
        if backend not in self._bulkheads:
            limit = self._backend_limits.get(backend)
            capacity = limit if limit and limit > 0 else self._global_limit
            self._bulkheads[backend] = asyncio.Semaphore(capacity)
        return self._bulkheads[backend]

    def _get_breaker(self, backend: str) -> CircuitBreaker:
        if backend not in self._circuit_breakers:
            self._circuit_breakers[backend] = CircuitBreaker()
        return self._circuit_breakers[backend]

    async def _run_task(self, task: ScheduledTask) -> None:
        backend = task.backend
        future = task.future
        queue = task.stream_queue

        bulkhead = self._get_bulkhead(backend)
        breaker = self._get_breaker(backend)

        async with self._semaphore:
            async with bulkhead:
                self._active_tasks += 1
                self._active_per_backend[backend] += 1
                try:
                    result = await self._execute_with_retry(task, breaker)
                    if future and not future.done():
                        future.set_result(result)
                    if queue is not None:
                        await queue.put(
                            StreamEvent(
                                task_id=task.task_id,
                                stream="result",
                                payload="",
                                timestamp=time.time(),
                                final=True,
                                result=result,
                            )
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    if future and not future.done():
                        future.set_exception(exc)
                    if queue is not None:
                        await queue.put(
                            StreamEvent(
                                task_id=task.task_id,
                                stream="error",
                                payload=str(exc),
                                timestamp=time.time(),
                                final=True,
                            )
                        )
                finally:
                    self._active_tasks -= 1
                    self._active_per_backend[backend] -= 1
                    if self._active_per_backend[backend] <= 0:
                        self._active_per_backend.pop(backend, None)

    async def _execute_with_retry(
        self, task: ScheduledTask, breaker: CircuitBreaker
    ) -> RuntimeResult:
        attempts = 0
        backoff = task.retry_backoff
        last_exception: Optional[Exception] = None

        while attempts <= task.retries:
            attempts += 1
            try:
                result = await self._execute_command(task)
                breaker.record_success()
                return RuntimeResult(
                    command=result.command,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    started_at=result.started_at,
                    completed_at=result.completed_at,
                    backend=task.backend,
                    attempts=attempts,
                    fallback_used=result.fallback_used,
                    metadata=dict(task.metadata),
                )
            except (RuntimeLaunchError, RuntimeTimeoutError) as exc:
                breaker.record_failure()
                last_exception = exc
                if attempts <= task.retries:
                    await asyncio.sleep(backoff * attempts)
                    continue

                if task.fallback is not None:
                    result = await self._execute_fallback(task, attempts)
                    breaker.record_success()
                    return result
                raise

        if task.fallback is not None:
            result = await self._execute_fallback(task, attempts)
            breaker.record_success()
            return result

        assert last_exception is not None
        raise last_exception

    async def _execute_fallback(self, task: ScheduledTask, attempts: int) -> RuntimeResult:
        fallback = task.fallback
        if fallback is None:
            raise RuntimeError("Fallback requested but not configured")

        if callable(fallback):
            result = await fallback()
            return RuntimeResult(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                started_at=result.started_at,
                    completed_at=result.completed_at,
                    backend=task.backend,
                    attempts=attempts,
                    fallback_used=True,
                    metadata=dict(task.metadata),
                )

        return await self._execute_process(
            tuple(fallback),
            task=task,
            attempts=attempts,
            is_fallback=True,
        )

    async def _execute_command(self, task: ScheduledTask) -> RuntimeResult:
        return await self._execute_process(task.command, task=task, attempts=0)

    async def _execute_process(
        self,
        command: Tuple[str, ...],
        *,
        task: ScheduledTask,
        attempts: int,
        is_fallback: bool = False,
    ) -> RuntimeResult:
        started_at = time.time()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE if self._capture_output else None,
            stderr=asyncio.subprocess.PIPE if self._capture_output else None,
            env=self._base_env,
        )

        stdout_bytes = b""
        stderr_bytes = b""

        async def _pump_stream(reader: Optional[asyncio.StreamReader], name: str) -> bytes:
            if reader is None:
                return b""
            buffer = bytearray()
            while True:
                chunk = await reader.readline()
                if not chunk:
                    break
                buffer.extend(chunk)
                if task.stream_queue is not None:
                    await task.stream_queue.put(
                        StreamEvent(
                            task_id=task.task_id,
                            stream=name,
                            payload=chunk.decode(errors="replace"),
                            timestamp=time.time(),
                        )
                    )
            return bytes(buffer)

        try:
            if self._capture_output and task.stream_queue is not None:
                stdout_task = asyncio.create_task(_pump_stream(process.stdout, "stdout"))
                stderr_task = asyncio.create_task(_pump_stream(process.stderr, "stderr"))
                await asyncio.wait_for(process.wait(), timeout=task.timeout)
                stdout_bytes = await stdout_task
                stderr_bytes = await stderr_task
            elif self._capture_output:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=task.timeout
                )
            else:
                await asyncio.wait_for(process.wait(), timeout=task.timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()
            raise RuntimeTimeoutError(
                f"Command timed out after {task.timeout} seconds: {' '.join(command)}"
            ) from exc

        completed_at = time.time()

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

        if process.returncode != 0:
            raise RuntimeLaunchError(
                f"Command exited with {process.returncode}: {' '.join(command)}"
            )

        return RuntimeResult(
            command=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            completed_at=completed_at,
            backend=task.backend,
            attempts=attempts if attempts else 1,
            fallback_used=is_fallback,
            metadata=dict(task.metadata),
        )


def _result_from_payload(payload: Mapping[str, Any]) -> RuntimeResult:
    """Convert a serialized payload back into a :class:`RuntimeResult`."""

    metadata = dict(payload.get("metadata", {}))
    return RuntimeResult(
        command=tuple(payload.get("command", ())),
        returncode=int(payload["returncode"]),
        stdout=payload.get("stdout", ""),
        stderr=payload.get("stderr", ""),
        started_at=float(payload["started_at"]),
        completed_at=float(payload["completed_at"]),
        backend=payload.get("backend", "distributed"),
        attempts=int(payload.get("attempts", 1)),
        fallback_used=bool(payload.get("fallback_used", False)),
        metadata=metadata,
    )


class GRPCSupervisorDriver:
    """gRPC control plane for multi-host scheduling."""

    _SERVICE_NAME = "bitnet.Supervisor"
    _SUBMIT_METHOD = "/bitnet.Supervisor/Submit"

    def __init__(
        self,
        supervisor: RuntimeSupervisor,
        *,
        host: str = "0.0.0.0",
        port: int = 50051,
    ) -> None:
        try:
            import grpc  # type: ignore
            from grpc import aio as grpc_aio  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("gRPC driver requires the 'grpcio' package") from exc

        self._grpc = grpc
        self._grpc_aio = grpc_aio
        self._supervisor = supervisor
        self._host = host
        self._port = port
        self._server: Optional[Any] = None

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    async def start(self) -> None:
        if self._server is not None:
            return

        handler = self._build_handler()
        self._server = self._grpc_aio.server()
        self._server.add_generic_rpc_handlers((handler,))
        self._server.add_insecure_port(self.address)
        await self._server.start()

    async def shutdown(self) -> None:
        if self._server is None:
            return
        await self._server.stop(0)
        self._server = None

    async def wait_for_termination(self) -> None:
        if self._server is None:
            return
        await self._server.wait_for_termination()

    async def submit_task(
        self,
        command: Sequence[str],
        *,
        options: Optional[Mapping[str, Any]] = None,
        target: Optional[str] = None,
    ) -> RuntimeResult:
        if target is None:
            target = self.address

        channel = self._grpc_aio.insecure_channel(target)
        try:
            method = channel.unary_unary(self._SUBMIT_METHOD)
            payload = json.dumps({
                "command": list(command),
                "options": dict(options or {}),
            }).encode("utf-8")
            response = await method(payload)
            data = json.loads(response.decode("utf-8"))
            return _result_from_payload(data)
        finally:
            await channel.close()

    def _build_handler(self) -> "Any":
        grpc_aio = self._grpc_aio
        supervisor = self._supervisor
        submit_method = self._SUBMIT_METHOD

        handler_factory = getattr(grpc_aio, "unary_unary_rpc_method_handler", None)
        if handler_factory is None:
            handler_factory = getattr(self._grpc, "unary_unary_rpc_method_handler")

        class _Handler(grpc_aio.GenericRpcHandler):
            def service(self, handler_call_details: Any) -> Optional[Any]:
                if handler_call_details.method == submit_method:
                    return handler_factory(self._submit)
                return None

            async def _submit(self, request: bytes, context: Any) -> bytes:
                payload = json.loads(request.decode("utf-8"))
                command = payload.get("command", [])
                options = payload.get("options", {})
                result = await supervisor.run_custom(command, **options)
                return json.dumps(asdict(result)).encode("utf-8")

        return _Handler()


class RaySupervisorDriver:
    """Adapter that uses Ray actors to execute commands remotely."""

    def __init__(
        self,
        runtime_factory: Callable[[], BitNetRuntime],
        *,
        supervisor_options: Optional[Mapping[str, Any]] = None,
        num_workers: int = 1,
        shutdown_ray: bool = False,
    ) -> None:
        try:
            import ray  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Ray driver requires the 'ray' package") from exc

        self._ray = ray
        self._shutdown_ray = shutdown_ray
        if not self._ray.is_initialized():  # pragma: no cover - environment dependent
            self._ray.init(ignore_reinit_error=True)

        options = dict(supervisor_options or {})
        runtime_factory_fn = runtime_factory

        ray = self._ray

        @ray.remote
        class _Worker:
            def __init__(self) -> None:
                self._runtime = runtime_factory_fn()
                self._supervisor = RuntimeSupervisor(self._runtime, **options)

            async def start(self) -> None:
                await self._supervisor.start()

            async def submit(self, command: Sequence[str], opts: Mapping[str, Any]) -> Mapping[str, Any]:
                result = await self._supervisor.run_custom(command, **dict(opts))
                return asdict(result)

            async def shutdown(self) -> None:
                await self._supervisor.shutdown()

        self._workers = [_Worker.remote() for _ in range(max(1, num_workers))]
        self._ray.get([w.start.remote() for w in self._workers])
        self._worker_cycle = itertools.cycle(self._workers)

    async def submit_task(self, command: Sequence[str], **options: Any) -> RuntimeResult:
        worker = next(self._worker_cycle)
        ref = worker.submit.remote(list(command), options)
        payload = await asyncio.to_thread(self._ray.get, ref)
        return _result_from_payload(payload)

    async def shutdown(self) -> None:
        refs = [w.shutdown.remote() for w in self._workers]
        await asyncio.to_thread(self._ray.get, refs)
        if self._shutdown_ray:
            self._ray.shutdown()


class CelerySupervisorDriver:
    """Adapter that uses Celery workers for task execution."""

    def __init__(self, app: Any, *, task_name: str = "bitnet.supervisor.run") -> None:
        try:
            import celery  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Celery driver requires the 'celery' package") from exc

        if not isinstance(app, celery.Celery):  # type: ignore[attr-defined]
            raise TypeError("app must be an instance of celery.Celery")

        self._app = app
        self._task_name = task_name

    def submit_task(self, command: Sequence[str], **options: Any) -> Any:
        return self._app.send_task(self._task_name, args=[list(command)], kwargs=options)

    @staticmethod
    def register_worker(
        app: Any,
        runtime_factory: Callable[[], BitNetRuntime],
        *,
        supervisor_options: Optional[Mapping[str, Any]] = None,
        task_name: str = "bitnet.supervisor.run",
    ) -> None:
        options = dict(supervisor_options or {})

        @app.task(name=task_name)  # type: ignore[attr-defined]
        def _run_command(command: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
            runtime = runtime_factory()
            supervisor = RuntimeSupervisor(runtime, **options)

            async def _execute() -> Mapping[str, Any]:
                async with supervisor:
                    result = await supervisor.run_custom(command, **kwargs)
                    return asdict(result)

            return asyncio.run(_execute())

    async def gather_result(self, async_result: Any) -> RuntimeResult:
        payload = await asyncio.to_thread(async_result.get)
        return _result_from_payload(payload)

