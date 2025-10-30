#!/usr/bin/env python3
"""Generate the architecture overview documentation from the source tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "architecture.md"

EXPECTED_PATHS = [
    REPO_ROOT / "run_inference.py",
    REPO_ROOT / "run_inference_server.py",
    REPO_ROOT / "bitnet" / "runtime.py",
    REPO_ROOT / "bitnet" / "supervisor.py",
    REPO_ROOT / "src",
    REPO_ROOT / "gpu",
]

HEADER = "# Architecture Overview\n\n"

INTRO = (
    "_This document is generated from the repository source tree. Regenerate with_\n"
    "```\n"
    "python utils/generate_architecture_docs.py\n"
    "```\n"
    "_(see instructions below).\n\n"
    "The BitNet runtime is intentionally thin: Python orchestration glues together\n"
    "compiled llama.cpp binaries and BitNet-specific kernels. The diagrams below are\n"
    "kept minimal so they match the present module layout.\n\n"
)

STACK_DIAGRAM = """## High-Level Stack\n\n```mermaid\ngraph TD\n    subgraph CLI\n        A[run_inference.py]\n        B[run_inference_server.py]\n    end\n    subgraph PythonRuntime\n        C[bitnet.runtime\\nBitNetRuntime]\n        D[bitnet.supervisor\\nRuntimeSupervisor]\n    end\n    subgraph NativeKernels\n        E[src/ggml-bitnet-*.cpp]\n        F[gpu/]\n    end\n    G[llama.cpp binaries\\n(llama-cli / llama-server)]\n\n    A --> C\n    B --> C\n    C --> G\n    C --> E\n    C --> F\n    D --> C\n```\n\n"""

INTERACTION_DIAGRAM = """## Runtime Interactions\n\n```mermaid\ngraph LR\n    User --> CLI\n    CLI --> |validates args| BitNetRuntime\n    BitNetRuntime --> |dry run / diagnostics| User\n    BitNetRuntime --> |launch| llama-cli\n    BitNetRuntime --> |launch| llama-server\n    RuntimeSupervisor --> |uses| BitNetRuntime\n    RuntimeSupervisor --> |enforces| Concurrency[Concurrency limits]\n    RuntimeSupervisor --> |propagates| Timeouts\n```\n\n"""

FOOTER = (
    "## Generation Instructions\n\n"
    "Run the helper to refresh this file:\n\n"
    "```\n"
    "python utils/generate_architecture_docs.py\n"
    "```\n\n"
    "The script verifies that the modules referenced above still exist and then\n"
    "re-emits the diagrams. Update the script before the architecture diverges.\n"
)


def _verify_paths() -> list[str]:
    missing = [str(path.relative_to(REPO_ROOT)) for path in EXPECTED_PATHS if not path.exists()]
    return missing


def _write_document() -> None:
    DOC_PATH.write_text(HEADER + INTRO + STACK_DIAGRAM + INTERACTION_DIAGRAM + FOOTER, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    missing = _verify_paths()
    if missing:
        print("Error: expected paths are missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    _write_document()
    return 0


if __name__ == "__main__":
    sys.exit(main())
