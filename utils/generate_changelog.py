#!/usr/bin/env python3
"""Generate the Markdown changelog from Git history."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "CHANGELOG.md"
DEFAULT_TARGETS = (
    lambda: os.environ.get("CHANGELOG_REF"),
    lambda: "origin/main",
    lambda: "HEAD",
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


def _resolve_target() -> str:
    for candidate_fn in DEFAULT_TARGETS:
        candidate = candidate_fn()
        if not candidate:
            continue
        if _rev_exists(candidate):
            return candidate
    raise RuntimeError("Unable to resolve a git revision for changelog generation")


def _git_log(limit: int) -> str:
    target = _resolve_target()
    result = subprocess.run(
        [
            "git",
            "log",
            f"-{limit}",
            "--date=short",
            "--pretty=format:%ad %h %s",
            target,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_document(log_output: str) -> None:
    header = "# Changelog\n\n"
    if log_output:
        first_line = log_output.splitlines()[0]
        cutoff = first_line.split()[0]
        intro = (
            f"_Generated from `git log` as of {cutoff}. Run `python utils/generate_changelog.py` to refresh._\n\n"
        )
        lines = "\n".join(f"- {line}" for line in log_output.splitlines())
        body = lines + "\n"
    else:
        intro = "_Generated from `git log`. Run `python utils/generate_changelog.py` to refresh._\n\n"
        body = "No commits found.\n"
    DOC_PATH.write_text(header + intro + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="Number of commits to include")
    args = parser.parse_args(argv)

    log_output = _git_log(args.limit)
    _write_document(log_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
