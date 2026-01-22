"""Minimal telemetry TUI skeleton for BitNet JSONL streams."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable


def _iter_lines(path: Path) -> Iterable[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _tail_events(path: Path, *, limit: int) -> list[dict[str, object]]:
    lines = _iter_lines(path)
    events = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render BitNet telemetry JSONL in the terminal.")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL telemetry file")
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="Refresh cadence in seconds",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Number of recent events to display",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="dark",
        help="Theme name (placeholder)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    last_snapshot: list[dict[str, object]] = []

    try:
        while True:
            snapshot = _tail_events(args.input, limit=args.tail)
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                print("\033c", end="")
                print(f"BitNet Telemetry TUI (theme={args.theme})")
                print(f"Source: {args.input}")
                print("-" * 60)
                for event in snapshot:
                    print(json.dumps(event, ensure_ascii=False))
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
