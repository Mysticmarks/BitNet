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
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

__all__ = [
    "BitNetRuntime",
    "RuntimeConfigurationError",
    "RuntimeLaunchError",
]


class RuntimeConfigurationError(ValueError):
    """Raised when the runtime environment is not configured correctly."""


class RuntimeLaunchError(RuntimeError):
    """Raised when a runtime process exits unsuccessfully."""


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """A structured summary of the runtime health check."""

    build_dir: Path
    binaries: Dict[str, bool]
    model_present: Optional[bool]
    cpu_count: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "build_dir": str(self.build_dir),
            "binaries": {k: bool(v) for k, v in self.binaries.items()},
            "model_present": self.model_present,
            "cpu_count": self.cpu_count,
        }


class BitNetRuntime:
    """Utility wrapper that prepares and launches llama.cpp executables."""

    def __init__(
        self,
        *,
        build_dir: Path | str = "build",
        env: Optional[Mapping[str, str]] = None,
        log_level: int | str = logging.INFO,
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

    # ------------------------------------------------------------------
    # Public API

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
        conversation: bool = False,
        dry_run: bool = False,
        extra_args: Optional[Sequence[str]] = None,
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
            conversation=conversation,
            extra_args=extra_args,
        )
        if dry_run:
            return command
        return self._execute(command)

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
        dry_run: bool = False,
        extra_args: Optional[Sequence[str]] = None,
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
            extra_args=extra_args,
        )
        if dry_run:
            return command
        return self._execute(command)

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
        return RuntimeDiagnostics(
            build_dir=self.build_dir,
            binaries=binaries,
            model_present=model_present,
            cpu_count=self._cpu_count,
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

    def _execute(self, command: Sequence[str]) -> int:
        self._logger.debug("Executing command: %s", " ".join(command))
        try:
            completed = subprocess.run(command, check=True, env=self._env)
        except subprocess.CalledProcessError as exc:
            self._logger.error("Runtime process exited with %s", exc.returncode)
            raise RuntimeLaunchError(str(exc)) from exc
        return completed.returncode

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
        conversation: bool,
        extra_args: Optional[Sequence[str]],
    ) -> List[str]:
        self._ensure_positive(n_predict, "n_predict")
        self._ensure_positive(ctx_size, "ctx_size")
        self._ensure_positive(batch_size, "batch_size")
        self._ensure_temperature(temperature)
        prompt = prompt.strip()
        if not prompt:
            raise RuntimeConfigurationError("Prompt must not be empty.")

        binary = self._require_binary("llama-cli")
        model_path = self._require_model(model)
        thread_count = self._resolve_threads(threads)

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
            "-ngl",
            "0",
            "-c",
            str(ctx_size),
            "--temp",
            f"{temperature}",
            "-b",
            str(batch_size),
        ]
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
        extra_args: Optional[Sequence[str]],
    ) -> List[str]:
        self._ensure_positive(n_predict, "n_predict")
        self._ensure_positive(ctx_size, "ctx_size")
        self._ensure_positive(batch_size, "batch_size")
        self._ensure_temperature(temperature)
        self._ensure_port(port)

        binary = self._require_binary("llama-server")
        model_path = self._require_model(model)
        thread_count = self._resolve_threads(threads)

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
            "-ngl",
            "0",
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

    def _ensure_positive(self, value: int, field: str) -> None:
        if value <= 0:
            raise RuntimeConfigurationError(f"{field} must be a positive integer. Got {value}.")

    def _ensure_temperature(self, value: float) -> None:
        if value <= 0:
            raise RuntimeConfigurationError("Temperature must be positive.")

    def _ensure_port(self, value: int) -> None:
        if not (0 < value < 65536):
            raise RuntimeConfigurationError(f"Port must be between 1 and 65535. Got {value}.")
