import argparse
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("setup_env")

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
    "x86_64": ["i2_s", "tl2"]
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=ON"],
    "x86_64": ["-DBITNET_X86_TL2=ON"]
}

OS_EXTRA_ARGS = {
    "Windows":["-T", "ClangCL"],
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}


class CommandExecutionError(RuntimeError):
    """Raised when an external command fails."""


@dataclass(frozen=True)
class SetupConfig:
    hf_repo: Optional[str]
    model_dir: str
    log_dir: str
    quant_type: str
    quant_embd: bool
    use_pretuned: bool


def resolve_architecture(machine: str) -> str:
    """Resolve the canonical architecture name for the given machine string."""
    try:
        return ARCH_ALIAS[machine]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported architecture '{machine}'.") from exc


def system_info() -> tuple[str, str]:
    machine = platform.machine()
    return platform.system(), resolve_architecture(machine)


def get_model_name(config: SetupConfig) -> str:
    if config.hf_repo:
        return SUPPORTED_HF_MODELS[config.hf_repo]["model_name"]
    return os.path.basename(os.path.normpath(config.model_dir))


def run_command(
    command: Sequence[str],
    *,
    shell: bool = False,
    log_step: Optional[str] = None,
    log_dir: Optional[str | Path] = None,
) -> subprocess.CompletedProcess:
    """Run a system command and ensure it succeeds.

    Parameters
    ----------
    command:
        The command to execute.
    shell:
        Whether to execute the command through the shell.
    log_step:
        When provided, stdout/stderr are redirected to ``{log_step}.log`` under ``log_dir``.
    log_dir:
        Directory used for storing step logs. Required when ``log_step`` is specified.
    """

    log_path: Optional[Path] = None
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
                )
        return subprocess.run(command, shell=shell, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = f"Command '{exc.cmd}' failed with return code {exc.returncode}"
        if log_path:
            message += f". See log at {log_path}"
        raise CommandExecutionError(message) from exc


def detect_compilers(system: Optional[str] = None) -> list[str]:
    """Determine the C and C++ compilers to hand off to CMake.

    Parameters
    ----------
    system:
        Optional override for the host operating system (used for testing).

    Returns
    -------
    list[str]
        The ``-D`` arguments to forward to ``cmake``.
    """

    system = system or platform.system()
    if system == "Windows":
        c_candidates = ["clang-cl", "clang", "cl", "gcc"]
        cxx_candidates = ["clang-cl", "clang++", "cl", "g++"]
    else:
        c_candidates = ["clang", "gcc"]
        cxx_candidates = ["clang++", "g++"]

    c_compiler = None
    for candidate in c_candidates:
        resolved = shutil.which(candidate)
        if resolved:
            c_compiler = resolved
            break

    cxx_compiler = None
    for candidate in cxx_candidates:
        resolved = shutil.which(candidate)
        if resolved:
            cxx_compiler = resolved
            break

    if not c_compiler or not cxx_compiler:
        raise RuntimeError(
            "Unable to locate a supported compiler toolchain. "
            "Install Clang or GCC (or clang-cl on Windows) and retry."
        )

    return [
        f"-DCMAKE_C_COMPILER={c_compiler}",
        f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
    ]


def ensure_tool_available(tool: str, *, install_hint: str) -> None:
    """Ensure the given CLI tool exists on PATH, raising a helpful error otherwise."""

    if shutil.which(tool):
        return
    raise RuntimeError(f"Required tool '{tool}' not found on PATH. {install_hint}")

def prepare_model(config: SetupConfig, command_runner=run_command):
    hf_url = config.hf_repo
    quant_type = config.quant_type
    quant_embd = config.quant_embd

    model_root = Path(config.model_dir)
    if hf_url is not None:
        ensure_tool_available(
            "huggingface-cli",
            install_hint="Install it via 'pip install huggingface_hub'.",
        )
        model_root = model_root / SUPPORTED_HF_MODELS[hf_url]["model_name"]
        model_root.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading model %s from HuggingFace to %s...", hf_url, model_root)
        command_runner(
            ["huggingface-cli", "download", hf_url, "--local-dir", str(model_root)],
            log_step="download_model",
        )
    elif not model_root.exists():
        raise FileNotFoundError(f"Model directory {model_root} does not exist.")
    else:
        logging.info("Loading model from directory %s.", model_root)

    gguf_path = model_root / f"ggml-model-{quant_type}.gguf"
    if not gguf_path.exists() or gguf_path.stat().st_size == 0:
        logging.info("Converting model to GGUF format...")
        converter_args = [
            sys.executable,
            "utils/convert-hf-to-gguf-bitnet.py",
            str(model_root),
        ]

        if quant_type.startswith("tl"):
            converter_args.extend(["--outtype", quant_type])
            if quant_embd:
                converter_args.append("--quant-embd")
            command_runner(converter_args, log_step="convert_to_tl")
        else:
            converter_args.extend(["--outtype", "f32"])
            command_runner(converter_args, log_step="convert_to_f32_gguf")

            f32_model = model_root / "ggml-model-f32.gguf"
            i2s_model = model_root / "ggml-model-i2_s.gguf"
            quantize_candidates = [
                Path("build") / "bin" / "llama-quantize",
                Path("build") / "bin" / "Release" / "llama-quantize",
            ]
            quantize_binary = next((candidate for candidate in quantize_candidates if candidate.exists()), None)
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
            command_runner(quantize_command, log_step="quantize_to_i2s")

        logging.info("GGUF model saved at %s", gguf_path)
    else:
        logging.info("GGUF model already exists at %s", gguf_path)

def setup_gguf(config: SetupConfig, command_runner=run_command):
    # Install the pip package
    command_runner(
        [sys.executable, "-m", "pip", "install", "3rdparty/llama.cpp/gguf-py"],
        log_step="install_gguf",
    )

def gen_code(config: SetupConfig, command_runner=run_command):
    _, arch = system_info()

    llama3_f3_models = set(
        [
            model["model_name"]
            for model in SUPPORTED_HF_MODELS.values()
            if model["model_name"].startswith("Falcon")
            or model["model_name"].startswith("Llama")
        ]
    )

    if arch == "arm64":
        if config.use_pretuned:
            pretuned_kernels = Path("preset_kernels") / get_model_name(config)
            if not pretuned_kernels.exists():
                raise FileNotFoundError(
                    f"Pretuned kernels not found for model {config.hf_repo}"
                )
            if config.quant_type == "tl1":
                shutil.copyfile(
                    pretuned_kernels / "bitnet-lut-kernels-tl1.h",
                    "include/bitnet-lut-kernels.h",
                )
                shutil.copyfile(
                    pretuned_kernels / "kernel_config_tl1.ini",
                    "include/kernel_config.ini",
                )
            elif config.quant_type == "tl2":
                shutil.copyfile(
                    pretuned_kernels / "bitnet-lut-kernels-tl2.h",
                    "include/bitnet-lut-kernels.h",
                )
                shutil.copyfile(
                    pretuned_kernels / "kernel_config_tl2.ini",
                    "include/kernel_config.ini",
                )
        model_name = get_model_name(config)
        if model_name == "bitnet_b1_58-large":
            command_runner(
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
            command_runner(
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
        elif model_name == "bitnet_b1_58-3B":
            command_runner(
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
        elif model_name == "BitNet-b1.58-2B-4T":
            command_runner(
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
            pretuned_kernels = Path("preset_kernels") / get_model_name(config)
            if not pretuned_kernels.exists():
                raise FileNotFoundError(
                    f"Pretuned kernels not found for model {config.hf_repo}"
                )
            shutil.copyfile(
                pretuned_kernels / "bitnet-lut-kernels-tl2.h",
                "include/bitnet-lut-kernels.h",
            )
        model_name = get_model_name(config)
        if model_name == "bitnet_b1_58-large":
            command_runner(
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
            command_runner(
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
        elif model_name == "bitnet_b1_58-3B":
            command_runner(
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
        elif model_name == "BitNet-b1.58-2B-4T":
            command_runner(
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
            raise ValueError(f"No code generation recipe for model '{model_name}' on {arch}")

def compile(config: SetupConfig, command_runner=run_command):
    ensure_tool_available("cmake", install_hint="Install CMake and ensure it is on PATH.")
    _, arch = system_info()
    if arch not in COMPILER_EXTRA_ARGS:
        raise ValueError(f"Architecture '{arch}' is not supported yet")
    logging.info("Compiling the code using CMake.")
    cmake_args = [
        "cmake",
        "-B",
        "build",
        *COMPILER_EXTRA_ARGS[arch],
        *OS_EXTRA_ARGS.get(platform.system(), []),
        *detect_compilers(),
    ]
    command_runner(cmake_args, log_step="generate_build_files")
    command_runner(["cmake", "--build", "build", "--config", "Release"], log_step="compile")


def main(config: SetupConfig, *, command_runner=run_command):
    runner = lambda command, **kwargs: command_runner(command, log_dir=config.log_dir, **kwargs)
    setup_gguf(config, command_runner=runner)
    gen_code(config, command_runner=runner)
    compile(config, command_runner=runner)
    prepare_model(config, command_runner=runner)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    _, arch = system_info()
    parser = argparse.ArgumentParser(description="Setup the environment for running the inference")
    parser.add_argument("--hf-repo", "-hr", type=str, help="Model used for inference", choices=SUPPORTED_HF_MODELS.keys())
    parser.add_argument("--model-dir", "-md", type=str, help="Directory to save/load the model", default="models")
    parser.add_argument("--log-dir", "-ld", type=str, help="Directory to save the logging info", default="logs")
    parser.add_argument(
        "--quant-type",
        "-q",
        type=str,
        help="Quantization type",
        choices=SUPPORTED_QUANT_TYPES[arch],
        default="i2_s",
    )
    parser.add_argument("--quant-embd", action="store_true", help="Quantize the embeddings to f16")
    parser.add_argument("--use-pretuned", "-p", action="store_true", help="Use the pretuned kernel parameters")
    return parser.parse_args(argv)


def validate_configuration(config: SetupConfig, arch: str) -> None:
    if config.hf_repo and config.hf_repo not in SUPPORTED_HF_MODELS:
        raise ValueError(f"Unsupported HuggingFace repository: {config.hf_repo}")
    supported_quant = SUPPORTED_QUANT_TYPES.get(arch, [])
    if config.quant_type not in supported_quant:
        raise ValueError(
            f"Quantization type '{config.quant_type}' is not supported for architecture '{arch}'."
        )


def signal_handler(sig, frame):
    logging.info("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    cli_args = parse_args()
    config = SetupConfig(
        hf_repo=getattr(cli_args, "hf_repo", None),
        model_dir=cli_args.model_dir,
        log_dir=cli_args.log_dir,
        quant_type=cli_args.quant_type,
        quant_embd=cli_args.quant_embd,
        use_pretuned=cli_args.use_pretuned,
    )
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    try:
        _, arch = system_info()
        validate_configuration(config, arch)
        main(config)
    except (CommandExecutionError, ValueError, RuntimeError, FileNotFoundError) as exc:
        logging.error(str(exc))
        sys.exit(1)
