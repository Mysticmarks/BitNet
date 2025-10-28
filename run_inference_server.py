"""HTTP server launcher for BitNet using llama.cpp."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from bitnet import BitNetRuntime, RuntimeConfigurationError, RuntimeLaunchError


def _parse_threads(value: str) -> Optional[int]:
    if value.lower() == "auto":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Thread count must be 'auto' or an integer") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the llama.cpp server with BitNet ergonomics")
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to the GGUF model")
    parser.add_argument("-n", "--n-predict", type=int, default=4096, help="Maximum tokens to sample")
    parser.add_argument("-c", "--ctx-size", type=int, default=2048, help="Context window size")
    parser.add_argument("-t", "--threads", type=_parse_threads, default=None, help="Thread count or 'auto'")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("-p", "--prompt", type=str, help="Optional system prompt")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Prompt batch size")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to the GPU (requires GPU build)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Directory that contains the compiled llama.cpp binaries",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity for runtime diagnostics",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show a health report for the runtime and exit",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional llama.cpp flags appended verbatim",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runtime = BitNetRuntime(build_dir=args.build_dir, log_level=getattr(logging, args.log_level))

    if args.diagnostics:
        report = runtime.diagnostics(model=args.model)
        print("Runtime diagnostics:")
        for key, value in report.as_dict().items():
            print(f"  {key}: {value}")
        return 0

    try:
        result = runtime.run_server(
            model=args.model,
            host=args.host,
            port=args.port,
            prompt=args.prompt,
            n_predict=args.n_predict,
            ctx_size=args.ctx_size,
            temperature=args.temperature,
            threads=args.threads,
            batch_size=args.batch_size,
            gpu_layers=args.gpu_layers,
            dry_run=args.dry_run,
            extra_args=args.extra_args,
        )
    except RuntimeConfigurationError as exc:
        parser.error(str(exc))
    except RuntimeLaunchError as exc:
        print(f"Runtime exited with an error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run and isinstance(result, list):
        print("Dry run command:")
        print(" ".join(result))
        return 0
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    sys.exit(main())
