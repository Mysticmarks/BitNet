# Documentation Review Checklist

Use this checklist before merging documentation-affecting changes or promoting a
release. Each item is enforced by `python tools/check_docs_sync.py` and maps
back to automation that prevents drift between the SRD, roadmap, and changelog.

## 1. Source of truth alignment

- [ ] Update the `_Last updated:` marker in `docs/system-requirements.md` and
      verify the README mirrors the value under "SRD last updated".
- [ ] Confirm the "High-Level Components" table in the SRD still lists all
      required modules (`src/ggml-bitnet-*.cpp`, `bitnet.runtime.BitNetRuntime`,
      `bitnet.supervisor.RuntimeSupervisor`, `run_inference.py`,
      `run_inference_server.py`, `setup_env.py`, `gpu/`).
- [ ] Run `python tools/check_docs_sync.py` to ensure SRD diagrams and module
      references remain discoverable.

## 2. Roadmap hygiene

- [ ] Refresh the `_Last updated:` marker in `docs/system-roadmap.md` and update
      the README "Roadmap last updated" line to match.
- [ ] Record scope or milestone adjustments inside the roadmap table before
      merging code that relies on them.

## 3. Change communication

- [ ] Append a bullet under `## [Unreleased]` → `### Documentation` in
      `CHANGELOG.md` summarising the change.
- [ ] Ensure deployment updates reflect in `docs/deployment.md`, including CI
      workflow references and operational runbooks.

## 4. Telemetry and dashboards

- [ ] Validate that `docs/telemetry-dashboards.md` references any new telemetry
      fields or themes introduced by the change.
- [ ] Capture screenshots or recordings for significant dashboard UX changes and
      link them from `docs/iteration-log.md`.

Once all boxes are checked, re-run `python tools/check_docs_sync.py` as the final
pre-merge gate.
