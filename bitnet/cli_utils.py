"""Shared helpers for BitNet command line entrypoints."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

PRESETS: Mapping[str, Mapping[str, int | float | bool | None]] = {
    "balanced": {
        "n_predict": 256,
        "ctx_size": 2048,
        "temperature": 0.8,
        "batch_size": 2,
    },
    "latency": {
        "n_predict": 128,
        "ctx_size": 1024,
        "temperature": 0.7,
        "batch_size": 4,
    },
    "quality": {
        "n_predict": 512,
        "ctx_size": 4096,
        "temperature": 0.9,
        "batch_size": 1,
    },
}

EXPLANATIONS: Mapping[str, str] = {
    "model": "Path to the GGUF file to load. The diagnostics flag verifies existence before launch.",
    "n-predict": "Controls how many tokens to generate. Higher values increase cost and latency.",
    "ctx-size": "Context window in tokens. Keep within hardware limits to avoid OOM.",
    "threads": "Thread count. Use 'auto' to match logical CPUs detected by the runtime.",
    "temperature": "Sampling diversity. Values above 1.0 lead to more creative but unstable output.",
    "batch-size": "Prompt batching for throughput. Larger values require more memory.",
    "gpu-layers": "Offload layers to GPU when compiled with CUDA/Metal builds.",
    "conversation": "Chat-style formatting with history awareness.",
}


@dataclass(slots=True)
class ResolvedConfiguration:
    """Effective configuration derived from preset and CLI overrides."""

    preset: Optional[str]
    values: MutableMapping[str, int | float | bool | str | None]
    adjusted: Dict[str, str]


def list_presets() -> str:
    """Return a formatted table of presets."""

    lines = ["Available presets:"]
    for name, preset in PRESETS.items():
        details = ", ".join(f"{key}={value}" for key, value in preset.items())
        lines.append(f"  - {name}: {details}")
    return "\n".join(lines)


def resolve_preset(preset: Optional[str], overrides: Mapping[str, object]) -> ResolvedConfiguration:
    """Merge preset defaults with explicit overrides."""

    values: MutableMapping[str, int | float | bool | str | None] = dict(overrides)
    adjusted: Dict[str, str] = {}
    if preset:
        base = PRESETS.get(preset)
        if not base:
            raise ValueError(f"Unknown preset '{preset}'. Use --list-presets to inspect options.")
        for key, value in base.items():
            values.setdefault(key, value)
        values["preset"] = preset
    else:
        values["preset"] = None
    return ResolvedConfiguration(preset=preset, values=values, adjusted=adjusted)


def explain_flag(flag: str) -> str:
    """Return a contextual explanation for a CLI flag."""

    normalised = flag.lstrip("-")
    text = EXPLANATIONS.get(normalised)
    if not text:
        raise KeyError(f"No contextual help registered for '{flag}'.")
    return dedent(text)


def themed_text(text: str) -> str:
    """Apply light theming based on BITNET_CLI_THEME."""

    theme = os.environ.get("BITNET_CLI_THEME", "mono").lower()
    if theme in {"dark", "light"} and sys.stdout.isatty():
        # Simple accent colour; avoid third-party dependencies.
        colour = "\033[95m" if theme == "dark" else "\033[94m"
        reset = "\033[0m"
        return f"{colour}{text}{reset}"
    return text


def validate_configuration(config: ResolvedConfiguration) -> None:
    """Guard against invalid runtime values."""

    values = config.values
    n_predict = int(values.get("n_predict", 0) or 0)
    ctx_size = int(values.get("ctx_size", 0) or 0)
    if n_predict <= 0:
        raise ValueError("n_predict must be positive.")
    if ctx_size <= 0:
        raise ValueError("ctx_size must be positive.")
    if n_predict > ctx_size * 2:
        config.adjusted["n_predict"] = "Capped to 2x context window for stability."
        values["n_predict"] = ctx_size * 2
    temperature = float(values.get("temperature", 0.0) or 0.0)
    if not 0.0 < temperature <= 1.5:
        raise ValueError("temperature must be between 0 and 1.5.")
    batch_size = int(values.get("batch_size", 1) or 1)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    gpu_layers = values.get("gpu_layers")
    if gpu_layers is not None and int(gpu_layers) < 0:
        raise ValueError("gpu_layers cannot be negative.")


def interactive_prompt(existing: Optional[str] = None) -> str:
    """Capture a multi-line prompt with keyboard shortcuts guidance."""

    instructions = themed_text(
        "Enter your prompt. Press Ctrl+D to submit, Ctrl+C to abort."
    )
    print(instructions)
    buffer: list[str] = []
    if existing:
        buffer.append(existing)
    try:
        while True:
            line = input()
            buffer.append(line)
    except EOFError:
        print()  # newline after Ctrl+D
    except KeyboardInterrupt:
        print()  # newline after Ctrl+C
        raise
    return "\n".join(buffer).strip()


def summarise_configuration(config: ResolvedConfiguration) -> str:
    """Generate a human-friendly summary of the configuration."""

    lines = ["Resolved configuration:"]
    if config.preset:
        lines.append(f"  preset: {config.preset}")
    for key in ("n_predict", "ctx_size", "temperature", "batch_size", "gpu_layers", "threads"):
        value = config.values.get(key)
        if value is not None:
            lines.append(f"  {key}: {value}")
    if config.adjusted:
        lines.append("  safety adjustments applied:")
        for key, reason in config.adjusted.items():
            lines.append(f"    - {key}: {reason}")
    return "\n".join(lines)


def merge_overrides(args: Mapping[str, object], fields: Iterable[str]) -> Dict[str, object]:
    """Extract a subset of argparse.Namespace into a dict."""

    return {field: args[field] for field in fields if args.get(field) is not None}
