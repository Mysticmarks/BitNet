"""Validate that high level documentation remains synchronized."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Missing required documentation file: {path}", file=sys.stderr)
        raise SystemExit(1)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        print(message, file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    srd_path = ROOT / "docs" / "system-requirements.md"
    readme_path = ROOT / "README.md"
    deployment_path = ROOT / "docs" / "deployment.md"

    srd_text = read_text(srd_path)
    readme_text = read_text(readme_path)
    deployment_text = read_text(deployment_path)

    match = re.search(r"_Last updated:\s*(?P<date>[^_]+)_", srd_text)
    ensure(match is not None, "SRD is missing the 'Last updated' marker")
    srd_date = match.group("date").strip()

    readme_match = re.search(r"SRD last updated:\s*(?P<date>[^\n]+)", readme_text)
    ensure(readme_match is not None, "README must echo the SRD last updated date")
    ensure(
        readme_match.group("date").strip() == srd_date,
        "README SRD date does not match docs/system-requirements.md",
    )

    ensure(
        "docs/deployment.md" in readme_text,
        "README should link to docs/deployment.md in the documentation map",
    )

    ensure(
        "Turnkey deployments" in deployment_text,
        "docs/deployment.md must describe turnkey deployment paths",
    )

    ensure(
        "Kubernetes" in deployment_text and "edge" in deployment_text.lower(),
        "docs/deployment.md should cover Kubernetes and edge targets",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
