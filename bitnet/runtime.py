"""Runtime orchestration helpers for BitNet Python entry points.

This module centralises the cross-platform logic required to launch the
pre-built llama.cpp binaries that power BitNet inference.  The original
scripts embedded a copy of this logic in multiple places which made it hard to
add validation, diagnostics or better error messages.  The new helper class
keeps that behaviour in one location and introduces a light amount of
instrumentation so that future extensions (e.g. watchdogs or telemetry) have a
clean hook point.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import random
import selectors
import shutil
import subprocess
import time
import contextlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

__all__ = [
    "BitNetRuntime",
    "HealthProbe",
    "ProbeReport",
    "RuntimeConfigurationError",
    "RuntimeDiagnostics",
    "RuntimeLaunchError",
    "RuntimeTimeoutError",
    "TelemetryEvent",
]


class RuntimeConfigurationError(ValueError):
    """Raised when the runtime environment is not configured correctly."""


class RuntimeLaunchError(RuntimeError):
    """Raised when a runtime process exits unsuccessfully."""


class RuntimeTimeoutError(RuntimeLaunchError):
    """Raised when a runtime process exceeds an execution deadline."""


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """A structured summary of the runtime health check."""

    build_dir: Path
    binaries: Dict[str, bool]
    model_present: Optional[bool]
    cpu_count: int
    probes: Mapping[str, ProbeReport]

    def as_dict(self) -> Dict[str, object]:
        return {
            "build_dir": str(self.build_dir),
            "binaries": {k: bool(v) for k, v in self.binaries.items()},
            "model_present": self.model_present,
            "cpu_count": self.cpu_count,
            "probes": {name: report.as_dict() for name, report in self.probes.items()},
        }


class BitNetRuntime:
    """Utility wrapper that prepares and launches llama.cpp executables."""

    def __init__(
        self,
        *,
        build_dir: Path | str = "build",
        env: Optional[Mapping[str, str]] = None,
        log_level: int | str = logging.INFO,
        health_probes: Optional[Sequence[HealthProbe]] = None,
        telemetry_sinks: Optional[Sequence[Callable[[TelemetryEvent], None]]] = None,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
        retry_backoff_cap: float = 8.0,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_reset: float = 30.0,
    ) -> None:
        self.build_dir = Path(build_dir)
        self._env: MutableMapping[str, str] = dict(env or os.environ)
        self._platform = platform.system()
        self._cpu_count = os.cpu_count() or 1
        self._logger = logging.getLogger("bitnet.runtime")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(self._coerce_log_level(log_level))
        self._last_execution: Optional[float] = None
        self._health_probes: Tuple[HealthProbe, ...] = (
            tuple(health_probes) if health_probes is not None else self._default_health_probes()
        )
        self._telemetry_sinks: List[Callable[[TelemetryEvent], None]] = list(telemetry_sinks or [])
        self._recent_events: Deque[TelemetryEvent] = deque(maxlen=256)
        self._metrics: Dict[str, float] = {
            "executions.total": 0,
            "executions.success": 0,
            "executions.failure": 0,
            "executions.retry": 0,
            "executions.timeout": 0,
            "executions.circuit_open": 0,
            "executions.duration_ms_total": 0.0,
        }
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_base = max(0.0, float(retry_backoff_base))
        self._retry_backoff_cap = max(self._retry_backoff_base, float(retry_backoff_cap))
        self._circuit_breaker_threshold = max(1, int(circuit_breaker_threshold))
        self._circuit_breaker_reset = max(1.0, float(circuit_breaker_reset))
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[float] = None
        self._probe_cache: Dict[str, Tuple[float, ProbeReport]] = {}
        self._probe_ttl = 30.0

    # ------------------------------------------------------------------
    # Public API

    @property
    def cpu_count(self) -> int:
        """Number of logical CPU cores detected on initialisation."""

        return self._cpu_count

    @property
    def environment(self) -> Mapping[str, str]:
        """Read-only view of the environment variables used for launches."""

        return MappingProxyType(self._env)

    @property
    def last_execution_timestamp(self) -> Optional[float]:
        """UTC timestamp of the most recent successful process execution."""

        return self._last_execution

    # ------------------------------------------------------------------
    # Telemetry API

    def add_telemetry_sink(self, sink: Callable[[TelemetryEvent], None]) -> None:
        """Register a callback that receives emitted telemetry events."""

        self._telemetry_sinks.append(sink)

    def remove_telemetry_sink(self, sink: Callable[[TelemetryEvent], None]) -> None:
        """Unregister a previously registered telemetry sink."""

        with contextlib.suppress(ValueError):
            self._telemetry_sinks.remove(sink)

    def metrics_snapshot(self) -> Mapping[str, float]:
        """Return a read-only copy of the current runtime metrics."""

        return MappingProxyType(dict(self._metrics))

    def recent_events(self, limit: Optional[int] = None) -> Sequence[TelemetryEvent]:
        """Return the most recent telemetry events."""

        if limit is None or limit >= len(self._recent_events):
            return tuple(self._recent_events)
        # Take a slice from the right-hand side without mutating the deque
        return tuple(list(self._recent_events)[-limit:])

    def run_inference(
        self,
        *,
        model: Path | str,
        prompt: str,
        n_predict: int = 128,
        ctx_size: int = 2048,
        temperature: float = 0.8,
        threads: Optional[int] = None,
        batch_size: int = 1,
        gpu_layers: Optional[int] = None,
        conversation: bool = False,
        dry_run: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        stream_consumer: Optional[Callable[[str, str], None]] = None,
    ) -> Sequence[str] | int:
        """Launch the llama-cli binary with validated arguments."""

        command = self._build_inference_command(
            model=model,
            prompt=prompt,
            n_predict=n_predict,
            ctx_size=ctx_size,
            temperature=temperature,
            threads=threads,
            batch_size=batch_size,
            gpu_layers=gpu_layers,
            conversation=conversation,
            extra_args=extra_args,
        )
        if dry_run:
            return command
        return self._execute(command, stream_consumer=stream_consumer)

    def run_server(
        self,
        *,
        model: Path | str,
        host: str = "127.0.0.1",
        port: int = 8080,
        prompt: Optional[str] = None,
        n_predict: int = 4096,
        ctx_size: int = 2048,
        temperature: float = 0.8,
        threads: Optional[int] = None,
        batch_size: int = 1,
        gpu_layers: Optional[int] = None,
        dry_run: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        stream_consumer: Optional[Callable[[str, str], None]] = None,
    ) -> Sequence[str] | int:
        """Launch the llama-server binary with validated arguments."""

        command = self._build_server_command(
            model=model,
            host=host,
            port=port,
            prompt=prompt,
            n_predict=n_predict,
            ctx_size=ctx_size,
            temperature=temperature,
            threads=threads,
            batch_size=batch_size,
            gpu_layers=gpu_layers,
            extra_args=extra_args,
        )
        if dry_run:
            return command
        return self._execute(command, stream_consumer=stream_consumer)

    def diagnostics(self, *, model: Optional[Path | str] = None) -> RuntimeDiagnostics:
        binaries = {
            "llama-cli": self._resolve_binary("llama-cli", required=False) is not None,
            "llama-server": self._resolve_binary("llama-server", required=False) is not None,
        }
        model_present: Optional[bool]
        if model is None:
            model_present = None
        else:
            model_present = Path(model).expanduser().exists()
        probe_results = {probe.name: self._run_probe(probe, refresh=True) for probe in self._health_probes}
        return RuntimeDiagnostics(
            build_dir=self.build_dir,
            binaries=binaries,
            model_present=model_present,
            cpu_count=self._cpu_count,
            probes=probe_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers

    def _coerce_log_level(self, level: int | str) -> int:
        if isinstance(level, int):
            return level
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, str):
            raise RuntimeConfigurationError(f"Unknown log level: {level}")
        return resolved

    def _execute(
        self,
        command: Sequence[str],
        *,
        stream_consumer: Optional[Callable[[str, str], None]] = None,
    ) -> int:
        command_tuple = tuple(command)
        max_attempts = max(1, self._max_retries + 1)
        attempt = 0

        while True:
            now = time.time()
            if self._circuit_open_until and now < self._circuit_open_until:
                self._metrics["executions.circuit_open"] += 1
                self._emit_event(
                    "runtime.execute.circuit_blocked",
                    {
                        "command": command_tuple,
                        "open_until": self._circuit_open_until,
                    },
                )
                raise RuntimeLaunchError(
                    "Runtime circuit breaker is open after repeated failures."
                )
            if self._circuit_open_until and now >= self._circuit_open_until:
                self._circuit_open_until = None
                self._consecutive_failures = 0

            attempt += 1
            start_time = time.time()
            self._metrics["executions.total"] += 1
            self._emit_event(
                "runtime.execute.start",
                {"command": command_tuple, "attempt": attempt},
            )
            self._logger.debug("Executing command (attempt %s/%s): %s", attempt, max_attempts, " ".join(command_tuple))

            stdout_data = ""
            stderr_data = ""
            returncode = -1

            try:
                returncode, stdout_data, stderr_data = self._stream_process(
                    command_tuple, stream_consumer=stream_consumer
                )
            except OSError as exc:
                stderr_data = str(exc)
                returncode = -1

            duration_ms = (time.time() - start_time) * 1000.0
            self._metrics["executions.duration_ms_total"] += duration_ms

            completion_payload = {
                "command": command_tuple,
                "returncode": returncode,
                "attempt": attempt,
                "duration_ms": duration_ms,
            }
            self._emit_event("runtime.execute.complete", completion_payload)

            if returncode == 0:
                self._metrics["executions.success"] += 1
                self._last_execution = time.time()
                self._consecutive_failures = 0
                self._circuit_open_until = None
                if stdout_data:
                    self._logger.debug(stdout_data.rstrip())
                if stderr_data:
                    self._logger.debug(stderr_data.rstrip())
                return returncode

            self._metrics["executions.failure"] += 1
            self._consecutive_failures += 1
            failure_payload = {
                "command": command_tuple,
                "returncode": returncode,
                "attempt": attempt,
                "stderr": stderr_data,
            }
            self._emit_event("runtime.execute.failure", failure_payload)

            if self._consecutive_failures >= self._circuit_breaker_threshold:
                self._circuit_open_until = time.time() + self._circuit_breaker_reset
                self._emit_event(
                    "runtime.execute.circuit_opened",
                    {
                        "command": command_tuple,
                        "failures": self._consecutive_failures,
                        "open_until": self._circuit_open_until,
                    },
                )

            if attempt >= max_attempts:
                raise RuntimeLaunchError(
                    f"Command exited with {returncode}: {' '.join(command_tuple)}"
                )

            self._metrics["executions.retry"] += 1
            delay = self._compute_backoff(attempt)
            self._emit_event(
                "runtime.execute.retry",
                {"command": command_tuple, "attempt": attempt, "delay": delay},
            )
            if delay > 0:
                time.sleep(delay)

    def _stream_process(
        self,
        command: Sequence[str],
        *,
        stream_consumer: Optional[Callable[[str, str], None]],
    ) -> Tuple[int, str, str]:
        process = self._spawn_process(command)
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        selector = selectors.DefaultSelector()

        try:
            if process.stdout is not None:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            if process.stderr is not None:
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")

            while selector.get_map():
                events = selector.select(timeout=0.1)
                if not events:
                    if process.poll() is not None:
                        break
                    continue
                for key, _ in events:
                    stream_name = key.data
                    try:
                        data = key.fileobj.readline()  # type: ignore[attr-defined]
                    except Exception as exc:  # pragma: no cover - defensive
                        self._logger.debug("Failed to read %s stream: %s", stream_name, exc)
                        selector.unregister(key.fileobj)
                        continue
                    if data == "":
                        selector.unregister(key.fileobj)
                        continue
                    if stream_name == "stdout":
                        stdout_chunks.append(data)
                    else:
                        stderr_chunks.append(data)
                    self._emit_event(
                        f"runtime.execute.{stream_name}",
                        {"command": command, "chunk": data},
                    )
                    if stream_consumer is not None:
                        try:
                            stream_consumer(stream_name, data)
                        except Exception:  # pragma: no cover - user provided callback
                            self._logger.exception("Stream consumer raised an exception")
            process.wait()
        finally:
            selector.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        return process.returncode, stdout_text, stderr_text

    def _spawn_process(self, command: Sequence[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            text=True,
            bufsize=1,
        )

    def _compute_backoff(self, attempt: int) -> float:
        if self._retry_backoff_base <= 0:
            return 0.0
        exponent = max(0, attempt - 1)
        delay = self._retry_backoff_base * (2 ** exponent)
        if delay <= 0:
            return 0.0
        jitter = random.uniform(0, delay * 0.1)
        return min(self._retry_backoff_cap, delay + jitter)

    def _emit_event(self, name: str, attributes: Mapping[str, object]) -> None:
        event = TelemetryEvent(name=name, timestamp=time.time(), attributes=dict(attributes))
        self._recent_events.append(event)
        for sink in list(self._telemetry_sinks):
            try:
                sink(event)
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.exception("Telemetry sink raised an exception")

    def _default_health_probes(self) -> Tuple[HealthProbe, ...]:
        return (
            _DiskHealthProbe(),
            _MemoryHealthProbe(),
            _GpuHealthProbe(),
        )

    def _run_probe(self, probe: HealthProbe, *, refresh: bool = False) -> ProbeReport:
        cached = self._probe_cache.get(probe.name)
        if not refresh and cached and (time.time() - cached[0]) < self._probe_ttl:
            return cached[1]
        try:
            result = probe(self)
        except Exception as exc:  # pragma: no cover - defensive
            result = ProbeReport(name=probe.name, status="error", details={"error": str(exc)})
        self._probe_cache[probe.name] = (time.time(), result)
        return result

    def _get_probe_result(self, name: str, *, refresh: bool = False) -> Optional[ProbeReport]:
        for probe in self._health_probes:
            if probe.name == name:
                return self._run_probe(probe, refresh=refresh)
        return None

    def _auto_tune_threads(self, requested: Optional[int]) -> int:
        if requested is not None:
            return requested
        if self._cpu_count <= 2:
            return self._cpu_count
        if self._cpu_count <= 8:
            return max(1, self._cpu_count - 1)
        return min(self._cpu_count, max(4, self._cpu_count // 2))

    def _auto_tune_gpu_layers(self, requested: Optional[int]) -> Optional[int]:
        if requested is not None:
            return requested
        gpu_snapshot = self._gpu_snapshot()
        count = gpu_snapshot.get("count", 0)
        if count <= 0:
            return 0
        devices = [d for d in gpu_snapshot.get("devices", []) if isinstance(d, Mapping)]
        memory_candidates: List[float] = []
        for device in devices:
            memory_gb = device.get("memory_gb")
            if isinstance(memory_gb, (int, float)) and memory_gb > 0:
                memory_candidates.append(float(memory_gb))
        if not memory_candidates:
            return 0
        min_memory = min(memory_candidates)
        layers = int(max(1.0, min(60.0, math.floor(min_memory * 4.0))))
        return layers

    def _gpu_snapshot(self) -> Dict[str, object]:
        probe = self._get_probe_result("gpu")
        if probe is None:
            return {"count": 0, "devices": []}
        return dict(probe.details)


    def _build_inference_command(
        self,
        *,
        model: Path | str,
        prompt: str,
        n_predict: int,
        ctx_size: int,
        temperature: float,
        threads: Optional[int],
        batch_size: int,
        gpu_layers: Optional[int],
        conversation: bool,
        extra_args: Optional[Sequence[str]],
    ) -> List[str]:
        self._ensure_positive(n_predict, "n_predict")
        self._ensure_positive(ctx_size, "ctx_size")
        self._ensure_positive(batch_size, "batch_size")
        self._ensure_temperature(temperature)
        tuned_gpu_layers = self._auto_tune_gpu_layers(gpu_layers)
        gpu_layers = self._resolve_gpu_layers(tuned_gpu_layers)
        prompt = prompt.strip()
        if not prompt:
            raise RuntimeConfigurationError("Prompt must not be empty.")

        binary = self._require_binary("llama-cli")
        model_path = self._require_model(model)
        thread_count = self._resolve_threads(self._auto_tune_threads(threads))

        command: List[str] = [
            str(binary),
            "-m",
            str(model_path),
            "-n",
            str(n_predict),
            "-t",
            str(thread_count),
            "-p",
            prompt,
            "-c",
            str(ctx_size),
            "--temp",
            f"{temperature}",
            "-b",
            str(batch_size),
        ]
        if gpu_layers is not None:
            command.extend(["-ngl", str(gpu_layers)])
        if conversation:
            command.append("-cnv")
        if extra_args:
            command.extend(extra_args)
        return command

    def _build_server_command(
        self,
        *,
        model: Path | str,
        host: str,
        port: int,
        prompt: Optional[str],
        n_predict: int,
        ctx_size: int,
        temperature: float,
        threads: Optional[int],
        batch_size: int,
        gpu_layers: Optional[int],
        extra_args: Optional[Sequence[str]],
    ) -> List[str]:
        self._ensure_positive(n_predict, "n_predict")
        self._ensure_positive(ctx_size, "ctx_size")
        self._ensure_positive(batch_size, "batch_size")
        self._ensure_temperature(temperature)
        self._ensure_port(port)
        tuned_gpu_layers = self._auto_tune_gpu_layers(gpu_layers)
        gpu_layers = self._resolve_gpu_layers(tuned_gpu_layers)

        binary = self._require_binary("llama-server")
        model_path = self._require_model(model)
        thread_count = self._resolve_threads(self._auto_tune_threads(threads))

        command: List[str] = [
            str(binary),
            "-m",
            str(model_path),
            "-c",
            str(ctx_size),
            "-t",
            str(thread_count),
            "-n",
            str(n_predict),
            "--temp",
            f"{temperature}",
            "--host",
            host,
            "--port",
            str(port),
            "-cb",
            "-b",
            str(batch_size),
        ]
        if gpu_layers is not None:
            command.extend(["-ngl", str(gpu_layers)])
        if prompt:
            command.extend(["-p", prompt])
        if extra_args:
            command.extend(extra_args)
        return command

    def _resolve_threads(self, requested: Optional[int]) -> int:
        if requested is None:
            return max(1, self._cpu_count)
        if requested <= 0:
            raise RuntimeConfigurationError("Thread count must be greater than zero.")
        if requested > self._cpu_count:
            self._logger.warning(
                "Requested %s threads but only %s CPU cores detected. Clamping value.",
                requested,
                self._cpu_count,
            )
            return self._cpu_count
        return requested

    def _require_binary(self, name: str) -> Path:
        path = self._resolve_binary(name, required=True)
        if path is None:
            raise RuntimeConfigurationError(f"Binary {name} not found.")
        return path

    def _resolve_binary(self, name: str, *, required: bool) -> Optional[Path]:
        candidates = list(self._candidate_binaries(name))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if required:
            hint = self._missing_binary_hint(name, candidates)
            raise RuntimeConfigurationError(hint)
        return None

    def _candidate_binaries(self, name: str) -> Iterable[Path]:
        bin_dir = self.build_dir / "bin"
        if self._platform == "Windows":
            yield bin_dir / "Release" / f"{name}.exe"
            yield bin_dir / f"{name}.exe"
            yield bin_dir / name
        else:
            yield bin_dir / name
            yield bin_dir / f"{name}.exe"

    def _missing_binary_hint(self, name: str, candidates: Iterable[Path]) -> str:
        paths = "\n".join(f"  - {candidate}" for candidate in candidates)
        return (
            f"Unable to locate the {name} executable. Expected to find one of:\n"
            f"{paths}\n"
            "Build the project with `cmake --build build -j` before running this command."
        )

    def _require_model(self, model: Path | str) -> Path:
        path = Path(model).expanduser()
        if not path.exists():
            raise RuntimeConfigurationError(
                f"Model file not found: {path}. Provide a valid --model path (GGUF)."
            )
        if not path.is_file():
            raise RuntimeConfigurationError(f"Model path must be a file: {path}")
        return path

    def _resolve_gpu_layers(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 0:
            raise RuntimeConfigurationError("gpu_layers must be zero or a positive integer.")
        return value

    def _ensure_positive(self, value: int, field: str) -> None:
        if value <= 0:
            raise RuntimeConfigurationError(f"{field} must be a positive integer. Got {value}.")

    def _ensure_temperature(self, value: float) -> None:
        if value <= 0:
            raise RuntimeConfigurationError("Temperature must be positive.")

    def _ensure_port(self, value: int) -> None:
        if not (0 < value < 65536):
            raise RuntimeConfigurationError(f"Port must be between 1 and 65535. Got {value}.")


@dataclass
class _DiskHealthProbe:
    name: str = "disk"

    def __call__(self, runtime: BitNetRuntime) -> ProbeReport:
        target = runtime.build_dir
        try:
            if not target.exists():
                target = target.parent if target.parent.exists() else Path(".")
            usage = shutil.disk_usage(target)
        except FileNotFoundError:
            usage = shutil.disk_usage(Path("."))
        except OSError as exc:
            return ProbeReport(
                name=self.name,
                status="error",
                details={"error": str(exc)},
            )
        free_ratio = usage.free / usage.total if usage.total else 0
        status = "ok" if free_ratio > 0.1 else "degraded"
        return ProbeReport(
            name=self.name,
            status=status,
            details={
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "used_bytes": usage.used,
                "free_ratio": free_ratio,
            },
        )


@dataclass
class _MemoryHealthProbe:
    name: str = "memory"

    def __call__(self, runtime: BitNetRuntime) -> ProbeReport:  # noqa: ARG002 - runtime reserved
        total, available = _memory_stats()
        if total <= 0:
            return ProbeReport(
                name=self.name,
                status="unknown",
                details={"total_bytes": total, "available_bytes": available},
            )
        free_ratio = available / total if total else 0
        status = "ok" if free_ratio > 0.1 else "degraded"
        return ProbeReport(
            name=self.name,
            status=status,
            details={
                "total_bytes": total,
                "available_bytes": available,
                "free_ratio": free_ratio,
            },
        )


@dataclass
class _GpuHealthProbe:
    name: str = "gpu"

    def __call__(self, runtime: BitNetRuntime) -> ProbeReport:  # noqa: ARG002 - runtime reserved
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except FileNotFoundError:
            devices = _env_visible_gpus()
            status = "absent" if not devices else "partial"
            return ProbeReport(
                name=self.name,
                status=status,
                details={"count": len(devices), "devices": devices, "reason": "nvidia-smi not available"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ProbeReport(
                name=self.name,
                status="error",
                details={"count": 0, "error": str(exc)},
            )

        if completed.returncode != 0:
            return ProbeReport(
                name=self.name,
                status="unavailable",
                details={
                    "count": 0,
                    "error": completed.stderr.strip() or completed.stdout.strip(),
                },
            )

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        devices = []
        for index, line in enumerate(lines):
            parts = [segment.strip() for segment in line.split(",")]
            name = parts[0] if parts else f"GPU-{index}"
            memory_gb: Optional[float] = None
            if len(parts) > 1:
                memory_gb = _parse_memory(parts[1])
            devices.append({"name": name, "memory_gb": memory_gb})
        status = "ok" if devices else "absent"
        return ProbeReport(
            name=self.name,
            status=status,
            details={"count": len(devices), "devices": devices},
        )


def _memory_stats() -> Tuple[int, int]:
    psutil_module = None
    try:
        import psutil as psutil_module  # type: ignore
    except Exception:  # pragma: no cover - psutil optional
        psutil_module = None

    if psutil_module is not None:
        try:
            stats = psutil_module.virtual_memory()
            return int(stats.total), int(stats.available)
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total = int(page_size * phys_pages)
            available = int(page_size * avail_pages)
            return total, available
        except (ValueError, OSError, AttributeError):  # pragma: no cover - defensive
            pass

    if platform.system() == "Windows":  # pragma: no cover - executed on Windows CI only
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception:
            pass

    return 0, 0


def _parse_memory(value: str) -> Optional[float]:
    tokens = value.split()
    if not tokens:
        return None
    try:
        amount = float(tokens[0])
    except ValueError:
        return None
    unit = tokens[1].lower() if len(tokens) > 1 else "mib"
    if unit.startswith("gi"):
        return amount
    if unit.startswith("mi"):
        return amount / 1024.0
    if unit.startswith("ti"):
        return amount * 1024.0
    return amount


def _env_visible_gpus() -> List[Mapping[str, object]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or visible.strip() in {"", "none"}:
        return []
    devices = []
    for index, token in enumerate(visible.split(",")):
        token = token.strip()
        if token:
            devices.append({"name": f"GPU-{token}", "index": index})
    return devices
@dataclass(frozen=True)
class TelemetryEvent:
    """Structured telemetry payload emitted by :class:`BitNetRuntime`."""

    name: str
    timestamp: float
    attributes: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }


class HealthProbe(Protocol):
    """Callable interface implemented by runtime health probes."""

    name: str

    def __call__(self, runtime: "BitNetRuntime") -> "ProbeReport":
        ...


@dataclass(frozen=True)
class ProbeReport:
    """A health check result produced by a :class:`HealthProbe`."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "details": dict(self.details),
        }


