"""Cross-platform bootstrapper for BitNet native and Python dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("setup_env")

SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {
        "model_name": "bitnet_b1_58-large",
    },
    "1bitLLM/bitnet_b1_58-3B": {
        "model_name": "bitnet_b1_58-3B",
    },
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {
        "model_name": "Falcon3-7B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-7B-1.58bit": {
        "model_name": "Falcon3-7B-1.58bit",
    },
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {
        "model_name": "Falcon3-10B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-10B-1.58bit": {
        "model_name": "Falcon3-10B-1.58bit",
    },
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {
        "model_name": "Falcon3-3B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-3B-1.58bit": {
        "model_name": "Falcon3-3B-1.58bit",
    },
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {
        "model_name": "Falcon3-1B-Instruct-1.58bit",
    },
    "microsoft/BitNet-b1.58-2B-4T": {
        "model_name": "BitNet-b1.58-2B-4T",
    },
    "tiiuae/Falcon-E-3B-Instruct": {
        "model_name": "Falcon-E-3B-Instruct",
    },
    "tiiuae/Falcon-E-1B-Instruct": {
        "model_name": "Falcon-E-1B-Instruct",
    },
    "tiiuae/Falcon-E-3B-Base": {
        "model_name": "Falcon-E-3B-Base",
    },
    "tiiuae/Falcon-E-1B-Base": {
        "model_name": "Falcon-E-1B-Base",
    },
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"],
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=ON"],
    "x86_64": ["-DBITNET_X86_TL2=ON"],
}

OS_EXTRA_ARGS = {
    "Windows": ["-T", "ClangCL"],
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}

STATE_VERSION = "2"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "bitnet"


class CommandExecutionError(RuntimeError):
    """Raised when an external command fails."""


@dataclass(frozen=True)
class SetupConfig:
    hf_repo: str | None
    model_dir: Path
    log_dir: Path
    quant_type: str
    quant_embd: bool
    use_pretuned: bool
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("BITNET_CACHE_DIR", DEFAULT_CACHE_ROOT))
    )
    skip_python: bool = False
    skip_build: bool = False
    skip_model: bool = False
    force: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))


class BootstrapState:
    """Simple JSON-backed cache for completed bootstrap stages."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._data = {}
            return
        if payload.get("version") != STATE_VERSION:
            self._data = {}
            return
        self._data = payload.get("stages", {})

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        content = {"version": STATE_VERSION, "stages": self._data}
        self._path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")

    def reset(self) -> None:
        self._data.clear()

    def is_complete(self, stage: str, token: str) -> bool:
        record = self._data.get(stage)
        return record is not None and record.get("token") == token

    def mark_complete(self, stage: str, token: str, **metadata: str) -> None:
        self._data[stage] = {"token": token, **metadata, "completed_at": str(int(time.time()))}


# ---------------------------------------------------------------------------
# Platform helpers


def resolve_architecture(machine: str) -> str:
    try:
        return ARCH_ALIAS[machine]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported architecture '{machine}'.") from exc


def system_info() -> tuple[str, str]:
    machine = platform.machine()
    return platform.system(), resolve_architecture(machine)


# ---------------------------------------------------------------------------
# Command runner


def run_command(
    command: Sequence[str],
    *,
    shell: bool = False,
    log_step: str | None = None,
    log_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    log_path: Path | None = None
    if log_step:
        if log_dir is None:
            raise ValueError("log_dir must be provided when log_step is specified")
        log_path = Path(log_dir) / f"{log_step}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if log_path:
            with open(log_path, "w", encoding="utf-8") as stream:
                return subprocess.run(
                    command,
                    shell=shell,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
        return subprocess.run(command, shell=shell, check=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        message = f"Command '{exc.cmd}' failed with return code {exc.returncode}"
        if log_path:
            message += f". See log at {log_path}"
        raise CommandExecutionError(message) from exc


# ---------------------------------------------------------------------------
# Dependency helpers


def ensure_tool_available(tool: str, *, install_hint: str) -> None:
    if shutil.which(tool):
        return
    raise RuntimeError(f"Required tool '{tool}' not found on PATH. {install_hint}")


def detect_compilers(system: str | None = None) -> list[str]:
    system = system or platform.system()
    if system == "Windows":
        c_candidates = ["clang-cl", "clang", "cl", "gcc"]
        cxx_candidates = ["clang-cl", "clang++", "cl", "g++"]
    else:
        c_candidates = ["clang", "gcc"]
        cxx_candidates = ["clang++", "g++"]

    c_compiler = _first_available(c_candidates)
    cxx_compiler = _first_available(cxx_candidates)

    if not c_compiler or not cxx_compiler:
        raise RuntimeError(
            "Unable to locate a supported compiler toolchain. "
            "Install Clang or GCC (or clang-cl on Windows) and retry."
        )

    return [
        f"-DCMAKE_C_COMPILER={c_compiler}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
    ]


def _first_available(candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def compute_requirements_hash() -> str:
    root = Path("requirements.txt")
    if not root.exists():
        return "no-requirements"

    digest = hashlib.sha256()
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        canonical = path.resolve()
        if canonical in seen:
            return
        seen.add(canonical)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return
        digest.update(content)
        for line in content.decode("utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("-r"):
                nested = stripped[2:].strip()
                visit(path.parent / nested)

    visit(root)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Bootstrapper implementation


class EnvironmentBootstrapper:
    def __init__(
        self,
        config: SetupConfig,
        *,
        command_runner=run_command,
    ) -> None:
        self.config = config
        self._runner = lambda cmd, **kwargs: command_runner(cmd, log_dir=config.log_dir, **kwargs)
        self._state = BootstrapState(config.cache_dir / "bootstrap-state.json")
        if config.force:
            self._state.reset()
        system, arch = system_info()
        self.system = system
        self.arch = arch

    # ------------------------------------------------------------------
    # Public orchestration

    def run(self) -> None:
        LOG.info(
            "Starting BitNet environment bootstrap (system=%s, arch=%s)",
            self.system,
            self.arch,
        )
        try:
            if not self.config.skip_python:
                self._ensure_python_dependencies()
            if not self.config.skip_build:
                self._ensure_codegen()
                self._ensure_build()
            if not self.config.skip_model:
                self._ensure_model_assets()
        finally:
            self._state.save()

    # ------------------------------------------------------------------
    # Individual stages

    def _ensure_python_dependencies(self) -> None:
        token = compute_requirements_hash()
        if self._state.is_complete("python", token):
            LOG.info("Python dependencies already satisfied (token=%s)", token)
            return

        ensure_tool_available("python", install_hint="Install Python 3.9 or newer.")
        pip_cache = self.config.cache_dir / "pip"
        env = dict(os.environ)
        env.setdefault("PIP_CACHE_DIR", str(pip_cache))
        pip_cache.mkdir(parents=True, exist_ok=True)

        LOG.info("Installing Python dependencies via pip (cache=%s)", pip_cache)
        self._runner(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            log_step="pip_upgrade",
            env=env,
        )
        if Path("requirements.txt").exists():
            self._runner(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--requirement",
                    "requirements.txt",
                ],
                log_step="pip_requirements",
                env=env,
            )
        self._runner(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "3rdparty/llama.cpp/gguf-py",
            ],
            log_step="pip_install_gguf",
            env=env,
        )
        self._state.mark_complete("python", token)

    def _ensure_codegen(self) -> None:
        token = self._codegen_token()
        if self._state.is_complete("codegen", token):
            LOG.info("Kernel code generation already completed (token=%s)", token)
            return
        self._generate_code()
        self._state.mark_complete("codegen", token)

    def _ensure_build(self) -> None:
        token = f"{self.arch}:{self.config.quant_type}:{self.config.use_pretuned}"
        if self._state.is_complete("cmake", token) and self._binaries_present():
            LOG.info("Native build already available (token=%s)", token)
            return

        ensure_tool_available("cmake", install_hint="Install CMake and ensure it is on PATH.")
        cmake_args = [
            "cmake",
            "-S",
            ".",
            "-B",
            "build",
            *COMPILER_EXTRA_ARGS[self.arch],
            *OS_EXTRA_ARGS.get(self.system, []),
            *detect_compilers(self.system),
        ]
        LOG.info("Configuring CMake with args: %s", cmake_args)
        self._runner(cmake_args, log_step="generate_build_files")
        build_command = ["cmake", "--build", "build", "--config", "Release"]
        if self.system != "Windows":
            build_command.extend(["--", f"-j{os.cpu_count() or 1}"])
        LOG.info("Compiling native targets via cmake --build")
        self._runner(build_command, log_step="compile")
        self._state.mark_complete("cmake", token)

    def _ensure_model_assets(self) -> None:
        token = self._model_token()
        if self._state.is_complete("model", token):
            LOG.info("Model artefacts already prepared (token=%s)", token)
            return
        self._prepare_model()
        self._state.mark_complete("model", token)

    # ------------------------------------------------------------------
    # Stage helpers

    def _codegen_token(self) -> str:
        return ":".join(
            [
                self.arch,
                self.config.quant_type,
                "pretuned" if self.config.use_pretuned else "autotuned",
                self._resolved_model_name(),
            ]
        )

    def _model_token(self) -> str:
        return ":".join(
            [
                self.config.hf_repo or self.config.model_dir.as_posix(),
                self.config.quant_type,
                "embd" if self.config.quant_embd else "no-embd",
            ]
        )

    def _binaries_present(self) -> bool:
        build_dir = Path("build") / "bin"
        if self.system == "Windows":
            return (build_dir / "Release" / "llama-cli.exe").exists()
        return (build_dir / "llama-cli").exists() and (build_dir / "llama-server").exists()

    # ------------------------------------------------------------------
    # Logic migrated from legacy script

    def _generate_code(self) -> None:
        config = self.config
        model_name = self._resolved_model_name()
        runner = self._runner
        arch = self.arch

        llama3_f3_models = {
            model["model_name"]
            for model in SUPPORTED_HF_MODELS.values()
            if model["model_name"].startswith("Falcon")
            or model["model_name"].startswith("Llama")
        }

        if arch == "arm64":
            if config.use_pretuned:
                pretuned_kernels = Path("preset_kernels") / model_name
                if not pretuned_kernels.exists():
                    raise FileNotFoundError(
                        f"Pretuned kernels not found for model {config.hf_repo or model_name}"
                    )
                self._copy_pretuned_kernels(pretuned_kernels, "tl1")
            if model_name == "bitnet_b1_58-large":
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl1.py",
                        "--model",
                        "bitnet_b1_58-large",
                        "--BM",
                        "256,128,256",
                        "--BK",
                        "128,64,128",
                        "--bm",
                        "32,64,32",
                    ],
                    log_step="codegen",
                )
            elif model_name in llama3_f3_models:
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl1.py",
                        "--model",
                        "Llama3-8B-1.58-100B-tokens",
                        "--BM",
                        "256,128,256,128",
                        "--BK",
                        "128,64,128,64",
                        "--bm",
                        "32,64,32,64",
                    ],
                    log_step="codegen",
                )
            elif model_name in {"bitnet_b1_58-3B", "BitNet-b1.58-2B-4T"}:
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl1.py",
                        "--model",
                        "bitnet_b1_58-3B",
                        "--BM",
                        "160,320,320",
                        "--BK",
                        "64,128,64",
                        "--bm",
                        "32,64,32",
                    ],
                    log_step="codegen",
                )
            else:
                raise ValueError(f"No code generation recipe for model '{model_name}' on arm64")
        else:
            if config.use_pretuned:
                pretuned_kernels = Path("preset_kernels") / model_name
                if not pretuned_kernels.exists():
                    raise FileNotFoundError(
                        f"Pretuned kernels not found for model {config.hf_repo or model_name}"
                    )
                self._copy_pretuned_kernels(pretuned_kernels, "tl2")
            if model_name == "bitnet_b1_58-large":
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl2.py",
                        "--model",
                        "bitnet_b1_58-large",
                        "--BM",
                        "256,128,256",
                        "--BK",
                        "96,192,96",
                        "--bm",
                        "32,32,32",
                    ],
                    log_step="codegen",
                )
            elif model_name in llama3_f3_models:
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl2.py",
                        "--model",
                        "Llama3-8B-1.58-100B-tokens",
                        "--BM",
                        "256,128,256,128",
                        "--BK",
                        "96,96,96,96",
                        "--bm",
                        "32,32,32,32",
                    ],
                    log_step="codegen",
                )
            elif model_name in {"bitnet_b1_58-3B", "BitNet-b1.58-2B-4T"}:
                runner(
                    [
                        sys.executable,
                        "utils/codegen_tl2.py",
                        "--model",
                        "bitnet_b1_58-3B",
                        "--BM",
                        "160,320,320",
                        "--BK",
                        "96,96,96",
                        "--bm",
                        "32,32,32",
                    ],
                    log_step="codegen",
                )
            else:
                raise ValueError(
                    f"No code generation recipe for model '{model_name}' on {self.arch}"
                )

    def _copy_pretuned_kernels(self, source: Path, suffix: str) -> None:
        shutil.copyfile(source / f"bitnet-lut-kernels-{suffix}.h", "include/bitnet-lut-kernels.h")
        shutil.copyfile(source / f"kernel_config_{suffix}.ini", "include/kernel_config.ini")

    def _prepare_model(self) -> None:
        config = self.config
        hf_url = config.hf_repo
        quant_type = config.quant_type
        quant_embd = config.quant_embd
        model_root = config.model_dir

        if hf_url is not None:
            ensure_tool_available(
                "huggingface-cli",
                install_hint="Install it via 'pip install huggingface_hub'.",
            )
            model_root = model_root / SUPPORTED_HF_MODELS[hf_url]["model_name"]
            model_root.mkdir(parents=True, exist_ok=True)
            LOG.info("Downloading model %s from HuggingFace to %s...", hf_url, model_root)
            self._runner(
                ["huggingface-cli", "download", hf_url, "--local-dir", str(model_root)],
                log_step="download_model",
            )
        elif not model_root.exists():
            raise FileNotFoundError(f"Model directory {model_root} does not exist.")
        else:
            LOG.info("Loading model from directory %s.", model_root)

        gguf_path = model_root / f"ggml-model-{quant_type}.gguf"
        if gguf_path.exists() and gguf_path.stat().st_size > 0:
            LOG.info("GGUF model already exists at %s", gguf_path)
            return

        LOG.info("Converting model to GGUF format...")
        converter_args = [
            sys.executable,
            "utils/convert-hf-to-gguf-bitnet.py",
            str(model_root),
        ]
        if quant_type.startswith("tl"):
            converter_args.extend(["--outtype", quant_type])
            if quant_embd:
                converter_args.append("--quant-embd")
            self._runner(converter_args, log_step="convert_to_tl")
        else:
            converter_args.extend(["--outtype", "f32"])
            self._runner(converter_args, log_step="convert_to_f32_gguf")

            f32_model = model_root / "ggml-model-f32.gguf"
            i2s_model = model_root / "ggml-model-i2_s.gguf"
            quantize_candidates = [
                Path("build") / "bin" / "llama-quantize",
                Path("build") / "bin" / "Release" / "llama-quantize",
            ]
            quantize_binary = next(
                (candidate for candidate in quantize_candidates if candidate.exists()),
                None,
            )
            if quantize_binary is None:
                raise FileNotFoundError(
                    "llama-quantize binary not found. Run the compile step before quantization."
                )

            quantize_command = [str(quantize_binary), str(f32_model), str(i2s_model), "I2_S", "1"]
            if quant_embd:
                quantize_command = [
                    str(quantize_binary),
                    "--token-embedding-type",
                    "f16",
                    str(f32_model),
                    str(i2s_model),
                    "I2_S",
                    "1",
                    "1",
                ]
            self._runner(quantize_command, log_step="quantize_to_i2s")
        LOG.info("GGUF model saved at %s", gguf_path)

    def _resolved_model_name(self) -> str:
        if self.config.hf_repo:
            return SUPPORTED_HF_MODELS[self.config.hf_repo]["model_name"]
        return self.config.model_dir.name


# ---------------------------------------------------------------------------
# CLI helpers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    _, arch = system_info()
    parser = argparse.ArgumentParser(
        description="Bootstrap the BitNet inference environment"
    )
    parser.add_argument(
        "--hf-repo",
        "-hr",
        type=str,
        help="Model used for inference",
        choices=SUPPORTED_HF_MODELS.keys(),
    )
    parser.add_argument(
        "--model-dir",
        "-md",
        type=Path,
        help="Directory to save/load the model",
        default=Path("models"),
    )
    parser.add_argument(
        "--log-dir",
        "-ld",
        type=Path,
        help="Directory to save logging output",
        default=Path("logs"),
    )
    parser.add_argument(
        "--quant-type",
        "-q",
        type=str,
        help="Quantization type",
        choices=SUPPORTED_QUANT_TYPES[arch],
        default="i2_s",
    )
    parser.add_argument(
        "--quant-embd",
        action="store_true",
        help="Quantize the embeddings to f16",
    )
    parser.add_argument(
        "--use-pretuned",
        "-p",
        action="store_true",
        help="Use the pretuned kernel parameters",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("BITNET_CACHE_DIR", DEFAULT_CACHE_ROOT)),
        help="Directory used for caching bootstrap artefacts",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python dependency installation",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip native compilation steps",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model conversion/quantization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached state and re-run all stages",
    )
    return parser.parse_args(argv)


def validate_configuration(config: SetupConfig, arch: str) -> None:
    if config.hf_repo and config.hf_repo not in SUPPORTED_HF_MODELS:
        raise ValueError(f"Unsupported HuggingFace repository: {config.hf_repo}")
    supported_quant = SUPPORTED_QUANT_TYPES.get(arch, [])
    if config.quant_type not in supported_quant:
        raise ValueError(
            f"Quantization type '{config.quant_type}' is not supported for architecture '{arch}'."
        )


def signal_handler(sig, frame):  # pragma: no cover - interactive convenience only
    LOG.info("Ctrl+C pressed, exiting...")
    sys.exit(0)


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = parse_args(argv)
    config = SetupConfig(
        hf_repo=getattr(cli_args, "hf_repo", None),
        model_dir=cli_args.model_dir,
        log_dir=cli_args.log_dir,
        quant_type=cli_args.quant_type,
        quant_embd=cli_args.quant_embd,
        use_pretuned=cli_args.use_pretuned,
        cache_dir=cli_args.cache_dir,
        skip_python=cli_args.skip_python,
        skip_build=cli_args.skip_build,
        skip_model=cli_args.skip_model,
        force=cli_args.force,
    )

    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        _, arch = system_info()
        validate_configuration(config, arch)
        EnvironmentBootstrapper(config).run()
    except (CommandExecutionError, ValueError, RuntimeError, FileNotFoundError) as exc:
        LOG.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
