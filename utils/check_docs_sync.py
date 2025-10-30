#!/usr/bin/env python3
"""Fail the build if interfaces change without documentation updates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = ("README.md", "docs/")
INTERFACE_PREFIXES = (
    "bitnet/",
    "run_inference.py",
    "run_inference_server.py",
    "src/",
    "gpu/",
)


def _rev_exists(revision: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", revision],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _determine_base_revision() -> str | None:
    if base := os.environ.get("DOCS_CHECK_BASE"):
        if _rev_exists(base):
            return base
    if pr_base := os.environ.get("GITHUB_BASE_REF"):
        candidate = f"origin/{pr_base}"
        if _rev_exists(candidate):
            return candidate
    if _rev_exists("origin/main"):
        return "origin/main"
    # Fallback to previous commit if history is shallow
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _git_diff(base: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _touched(paths: Iterable[str], prefixes: Sequence[str]) -> list[str]:
    touched: list[str] = []
    for path in paths:
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix):
                touched.append(path)
                break
    return touched


def main() -> int:
    base = _determine_base_revision()
    if base is None:
        print("warning: unable to determine base revision; skipping docs sync check", file=sys.stderr)
        return 0

    changed_files = _git_diff(base)
    if not changed_files:
        return 0

    interface_changes = _touched(changed_files, INTERFACE_PREFIXES)
    doc_changes = _touched(changed_files, DOC_PATHS)

    if interface_changes and not doc_changes:
        print("\nDocumentation update required:\n", file=sys.stderr)
        for path in interface_changes:
            print(f"  • {path}", file=sys.stderr)
        print(
            "Interfaces changed but README/docs were not updated. Run the documentation checklist in docs/docs-review-checklist.md.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
