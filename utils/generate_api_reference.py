#!/usr/bin/env python3
"""Generate the Markdown API reference from the Python sources."""

from __future__ import annotations

import argparse
from importlib import import_module
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "api-reference.md"
MODULE_NAMES = [
    "bitnet.runtime",
    "bitnet.supervisor",
    "run_inference",
    "run_inference_server",
]


def _load_module(name: str) -> ModuleType:
    return import_module(name)


def _format_signature(obj) -> str:
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = "()"
    return signature


def _summarise(docstring: str | None) -> str:
    if not docstring:
        return "No docstring provided."
    summary = docstring.strip().splitlines()[0]
    return summary or "No docstring provided."


def _gather_classes(module: ModuleType, path: Path) -> List[Tuple[str, str, List[Tuple[str, str, str]]]]:
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        methods = []
        for method_name, method in inspect.getmembers(obj, inspect.isfunction):
            if method_name.startswith("__"):
                continue
            if method.__qualname__.split(".")[0] != obj.__name__:
                continue
            if getattr(method, "__module__", None) != module.__name__:
                continue
            methods.append((method_name, _format_signature(method), _summarise(method.__doc__)))
        classes.append((name, _summarise(obj.__doc__), methods))
    return classes


def _gather_functions(module: ModuleType, path: Path) -> List[Tuple[str, str, str]]:
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module.__name__:
            continue
        functions.append((name, _format_signature(obj), _summarise(obj.__doc__)))
    return functions


def _render_module(module: ModuleType) -> str:
    path = Path(inspect.getsourcefile(module) or module.__file__).resolve()
    rel_path = path.relative_to(REPO_ROOT)
    parts: List[str] = [f"## Module: `{rel_path}`\n", f"{_summarise(module.__doc__)}\n\n"]

    classes = _gather_classes(module, path)
    if classes:
        for name, summary, methods in classes:
            parts.append(f"### Class `{name}`\n")
            parts.append(f"{summary}\n\n")
            if methods:
                parts.append("#### Methods\n")
                for method_name, signature, method_summary in methods:
                    parts.append(f"- `{method_name}{signature}` — {method_summary}\n")
                parts.append("\n")

    functions = _gather_functions(module, path)
    if functions:
        parts.append("### Module Functions\n")
        for func_name, signature, func_summary in functions:
            parts.append(f"- `{func_name}{signature}` — {func_summary}\n")
        parts.append("\n")

    return "".join(parts)


def _write_document(sections: List[str]) -> None:
    header = "# API Reference\n\n"
    intro = "_Generated from source. Run `python utils/generate_api_reference.py` after interface changes._\n\n"
    DOC_PATH.write_text(header + intro + "".join(sections), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    sections: List[str] = []
    sys.path.insert(0, str(REPO_ROOT))
    for name in MODULE_NAMES:
        module = _load_module(name)
        sections.append(_render_module(module))
    _write_document(sections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
