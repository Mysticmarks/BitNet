"""High-level CLI for launching BitNet inference runs.

The original implementation simply proxied to llama.cpp.  This version layers
runtime validation, diagnostics, and ergonomic defaults (such as automatic
thread detection) on top of the compiled binaries while remaining entirely
backwards compatible with the llama.cpp flags for advanced users.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from bitnet import BitNetRuntime, RuntimeConfigurationError, RuntimeLaunchError
from bitnet.cli_utils import (
    PRESETS,
    explain_flag,
    interactive_prompt,
    list_presets,
    merge_overrides,
    resolve_preset,
    summarise_configuration,
    themed_text,
    validate_configuration,
)


def _parse_threads(value: str) -> Optional[int]:
    if value.lower() == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Thread count must be 'auto' or an integer") from exc
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BitNet inference via llama.cpp")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to the GGUF model. Required unless using --list-presets/--explain.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to generate text from. If omitted, an interactive editor opens.",
    )
    parser.add_argument("-n", "--n-predict", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("-c", "--ctx-size", type=int, default=2048, help="Context window size")
    parser.add_argument("-t", "--threads", type=_parse_threads, default=None, help="Thread count or 'auto'")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Prompt batch size")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to the GPU (requires GPU build)",
    )
    parser.add_argument("-cnv", "--conversation", action="store_true", help="Enable chat mode")
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
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Load a preset configuration (balanced, latency, quality)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available presets and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display the resolved configuration before execution",
    )
    parser.add_argument(
        "--explain",
        type=str,
        metavar="FLAG",
        help="Show contextual help for a specific flag (e.g. --explain n-predict)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive prompt capture even if --prompt is provided",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print(list_presets())
        return 0

    if args.explain:
        try:
            print(explain_flag(args.explain))
        except KeyError as exc:
            parser.error(str(exc))
        return 0

    if not args.model:
        parser.error("--model is required for inference runs")

    runtime = BitNetRuntime(build_dir=args.build_dir, log_level=getattr(logging, args.log_level))

    if args.diagnostics:
        report = runtime.diagnostics(model=args.model)
        print("Runtime diagnostics:")
        for key, value in report.as_dict().items():
            print(f"  {key}: {value}")
        return 0

    overrides = merge_overrides(
        vars(args),
        ["n_predict", "ctx_size", "temperature", "threads", "batch_size", "gpu_layers"],
    )
    try:
        resolved = resolve_preset(args.preset, overrides)
        validate_configuration(resolved)
    except ValueError as exc:
        parser.error(str(exc))

    if args.show_config:
        print(themed_text(summarise_configuration(resolved)))

    prompt = args.prompt
    if args.interactive or not prompt:
        try:
            prompt = interactive_prompt(existing=prompt)
        except KeyboardInterrupt:
            print("Prompt capture cancelled.")
            return 1
        if not prompt:
            parser.error("Prompt cannot be empty.")

    try:
        result = runtime.run_inference(
            model=args.model,
            prompt=prompt,
            n_predict=int(resolved.values.get("n_predict", args.n_predict)),
            ctx_size=int(resolved.values.get("ctx_size", args.ctx_size)),
            temperature=float(resolved.values.get("temperature", args.temperature)),
            threads=resolved.values.get("threads", args.threads),
            batch_size=int(resolved.values.get("batch_size", args.batch_size)),
            gpu_layers=resolved.values.get("gpu_layers", args.gpu_layers),
            conversation=args.conversation,
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


if __name__ == "__main__":  # pragma: no cover - exercised in tests via main()
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    sys.exit(main())
