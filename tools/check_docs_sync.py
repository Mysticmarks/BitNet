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
    roadmap_path = ROOT / "docs" / "system-roadmap.md"
    changelog_path = ROOT / "CHANGELOG.md"
    checklist_path = ROOT / "docs" / "documentation-review-checklist.md"

    srd_text = read_text(srd_path)
    readme_text = read_text(readme_path)
    deployment_text = read_text(deployment_path)
    roadmap_text = read_text(roadmap_path)
    changelog_text = read_text(changelog_path)
    checklist_text = read_text(checklist_path)

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

    roadmap_match = re.search(r"_Last updated:\s*(?P<date>[^_]+)_", roadmap_text)
    ensure(roadmap_match is not None, "Roadmap is missing the 'Last updated' marker")
    roadmap_date = roadmap_match.group("date").strip()

    roadmap_readme_match = re.search(r"Roadmap last updated:\s*(?P<date>[^\n]+)", readme_text)
    ensure(roadmap_readme_match is not None, "README must echo the roadmap last updated date")
    ensure(
        roadmap_readme_match.group("date").strip() == roadmap_date,
        "README roadmap date does not match docs/system-roadmap.md",
    )

    required_components = [
        "src/ggml-bitnet-*.cpp",
        "bitnet.runtime.BitNetRuntime",
        "bitnet.supervisor.RuntimeSupervisor",
        "run_inference.py",
        "run_inference_server.py",
        "setup_env.py",
        "gpu/",
    ]
    for component in required_components:
        ensure(
            component in srd_text,
            f"SRD must list component '{component}' in the architecture overview",
        )

    ensure("## [Unreleased]" in changelog_text, "CHANGELOG must contain an [Unreleased] section")
    ensure("### Documentation" in changelog_text, "CHANGELOG must document documentation updates")
    ensure(
        re.search(r"## \[\d{4}-\d{2}-\d{2}\]", changelog_text),
        "CHANGELOG must include at least one dated release entry",
    )

    checklist_expectations = [
        "docs/system-requirements.md",
        "docs/system-roadmap.md",
        "CHANGELOG.md",
        "docs/deployment.md",
        "docs/telemetry-dashboards.md",
        "python tools/check_docs_sync.py",
    ]
    for item in checklist_expectations:
        ensure(
            item in checklist_text,
            f"Documentation checklist must reference '{item}'",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
