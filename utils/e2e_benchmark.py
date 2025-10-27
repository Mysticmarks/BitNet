"""Utility for running llama.cpp's benchmark with BitNet defaults."""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)


def run_command(
    command: Sequence[str],
    *,
    shell: bool = False,
    log_dir: Path | None = None,
    log_step: str | None = None,
) -> subprocess.CompletedProcess:
    """Execute ``command`` and raise an informative error on failure."""

    log_path: Path | None = None
    if log_step is not None:
        if log_dir is None:
            raise ValueError("log_dir must be provided when log_step is specified")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{log_step}.log"

    try:
        if log_path is not None:
            with log_path.open("w", encoding="utf-8") as stream:
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
        message = f"Error occurred while running command: {exc}"
        if log_path is not None:
            message += f". Check details in {log_path}"
        raise RuntimeError(message) from exc


def find_benchmark_binary(build_dir: Path, *, system: str | None = None) -> Path:
    """Locate the llama.cpp benchmark binary inside ``build_dir``."""

    system = system or platform.system()
    candidates: Iterable[Path]
    if system == "Windows":
        candidates = (
            build_dir / "bin" / "Release" / "llama-bench.exe",
            build_dir / "bin" / "llama-bench.exe",
            build_dir / "bin" / "llama-bench",
        )
    else:
        candidates = (build_dir / "bin" / "llama-bench",)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Benchmark binary not found. Build the project before running the benchmark."
    )


def build_benchmark_command(
    bench_path: Path,
    *,
    model: str,
    n_token: int,
    n_prompt: int,
    threads: int,
) -> list[str]:
    """Construct the llama.cpp benchmark command with sensible defaults."""

    return [
        str(bench_path),
        "-m",
        model,
        "-n",
        str(n_token),
        "-ngl",
        "0",
        "-b",
        "1",
        "-t",
        str(threads),
        "-p",
        str(n_prompt),
        "-r",
        "5",
    ]


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute the benchmark with the provided CLI arguments."""

    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / "build"
    bench_path = find_benchmark_binary(build_dir)

    command = build_benchmark_command(
        bench_path,
        model=args.model,
        n_token=args.n_token,
        n_prompt=args.n_prompt,
        threads=args.threads,
    )

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        LOGGER.info("Writing benchmark logs to %s", log_dir)
        run_command(command, log_dir=log_dir, log_step="benchmark")
    else:
        run_command(command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the llama.cpp benchmark with BitNet defaults",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model file")
    parser.add_argument(
        "-n",
        "--n-token",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "-p",
        "--n-prompt",
        type=int,
        default=512,
        help="Prompt length to warm up the benchmark",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=2,
        help="Number of threads to use",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional directory to store benchmark logs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
