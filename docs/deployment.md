# Deployment and Runtime Guide

This document explains how to prepare a repeatable BitNet runtime environment
that is closer to a production deployment than the ad-hoc steps referenced in
the legacy README instructions.

## 1. Build once, run anywhere

1. Configure and build the C++ sources:
   ```bash
   cmake -S . -B build -DLLAMA_BUILD_EXAMPLES=OFF
   cmake --build build -j
   ```
2. Copy the `build/bin` directory to the machines that will host inference.
3. Confirm the binaries with the health check:
   ```bash
   python run_inference.py --model /path/to/model.gguf --prompt "ping" --dry-run --diagnostics
   ```

The `--diagnostics` flag verifies that the llama.cpp executables exist in the
selected build directory, reports the detected CPU core count, and confirms the
presence of your model file.

## 2. Runtime ergonomics

Both `run_inference.py` and `run_inference_server.py` now use the shared
`bitnet.runtime.BitNetRuntime` helper which provides:

- Automatic thread detection (`--threads auto` by default).
- Input validation with actionable error messages for missing binaries, models
  or invalid numeric ranges.
- Dry-run support to surface the exact command that will be executed, making it
  easy to integrate with supervisors such as `systemd` or Kubernetes Jobs.
- Structured diagnostics that can be exported to JSON if you need to plug the
  scripts into monitoring dashboards.
- An asyncio-based `RuntimeSupervisor` that can enforce concurrency limits and
  kill runaway processes when paired with API gateways or job queues.  See
  [`runtime_supervisor.md`](runtime_supervisor.md) for orchestration patterns.

## 3. Server hardening checklist

Before exposing the llama.cpp server to real users:

- Always pin the bind address via `--host` and, if reachable from outside your
  network, place the process behind a TLS-terminating proxy.
- Use the new `--batch-size` switch to keep request bursts under control.
- Wrap the script inside a process supervisor (e.g. `systemd`, `supervisord` or
  Nomad) and set restart limits according to your SLOs.  Failures propagate as
  non-zero exit codes, making it easy to trigger automated restarts.
- Export metrics by tailing the server stdout and pushing it to your
  observability stack.  The shared runtime logs every launch attempt, so you can
  instrument the scripts without modifying llama.cpp itself.

## 4. Testing the deployment pipeline

The `tests/test_runtime.py` module exercises the runtime helper directly.  The
suite creates ephemeral binaries and models to validate command construction,
thread clamping logic, diagnostics, and the error paths triggered when
pre-requisites are missing.  `tests/test_supervisor.py` adds asyncio-driven
coverage for concurrency guarantees and timeout enforcement so you can wire the
supervisor into production control planes with confidence.  Extend these tests
with integration checks that spawn actual llama.cpp processes when you have the
binaries and GGUF artefacts available in CI.

## 5. Future extensions

- Emit JSON diagnostics via structured logging for richer dashboards.
- Introduce GPU-aware runtime helpers that detect the available accelerators and
  pick the optimal binary automatically.
- Extend reproducible infrastructure definitions to cover GPU node pools and
  service meshes.

## 6. Automation and Release Alignment

The CI workflow defined in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
promotes build artefacts in a staircase pattern so that downstream deployment
stays reproducible:

1. **Lint gate (`lint`)** installs Python tooling, runs `ruff`, and validates
   bytecode compilation to catch packaging regressions early.
2. **Unit and integration tests (`python-tests`)** execute `pytest` with the
   same dependency set as local development to confirm runtime behaviours.
3. **Native build (`build-native`)** configures and compiles the llama.cpp
   derivatives, uploading the resulting tarball for release packaging.
4. **Python wheel (`python-wheel`)** builds the distribution artefact used by
   CLI wrappers and dashboards.
5. **Container images (`container-images`)** produce reproducible `bitnet-runtime`
   and `bitnet-edge` images that can be promoted to registries.
6. **Docs synchronisation (`docs-sync`)** executes `python tools/check_docs_sync.py`
   to verify timestamps, SRD component coverage, the roadmap log, and the
   documentation checklist.
7. **Terraform validation (`terraform`)** ensures infrastructure definitions
   remain deployable.

### 6.1 Release promotion steps

1. Trigger a tagged build (`git tag vX.Y.Z && git push origin vX.Y.Z`) once the
   CI pipeline turns green.
2. Download the `bitnet-binaries` and `bitnet-wheel` artefacts from the run and
   attach them to the GitHub Release draft.
3. Publish container images by loading the `bitnet-runtime.tar` and
   `bitnet-edge.tar` archives locally and pushing to the target registry using
   `docker load` / `docker push`.
4. Update `CHANGELOG.md` under the dated section with deployment highlights,
   referencing any SRD or roadmap amendments.
5. Announce the release in `docs/iteration-log.md`, linking to dashboard themes
   and operational runbooks as appropriate.

### 6.2 Operational runbooks

| Scenario | First response | Escalation path | Telemetry hook |
| --- | --- | --- | --- |
| **Hotfix rollback** | Redeploy previous container artefact using the stored `bitnet-runtime.tar`. | Notify release manager, file incident log in `docs/iteration-log.md`. | Compare telemetry deltas in dashboards using the "Release" theme. |
| **Model asset missing** | Run `python run_inference.py --dry-run --diagnostics` to confirm path resolution. | Loop in storage owner, restore object store permissions. | Observe `missing_artifact` events in telemetry stream. |
| **Kubernetes pod crash** | Check `kubectl logs` and `kubectl describe` for readiness probe failures. | Page on-call for kernel/runtime subsystem, attach CI artefact references. | Inspect `RuntimeSupervisor` failure summaries in dashboards. |
| **Edge device drift** | Re-run `setup_env.py` with `--clean` on affected device, verify bootstrap cache. | Engage fleet operations to reimage or resync environment. | Use TUI dashboard offline snapshot mode to compare checksums. |

Operators should treat `docs/documentation-review-checklist.md` as the
pre-flight gate before approving changes that impact the pipeline or runbooks.

## 7. Turnkey deployments

BitNet now ships repeatable deployment assets that reduce the amount of custom
plumbing required to move from local experiments to managed infrastructure.  CI
guarantees that these assets stay in sync with the runtime codebase.

### 6.1 Container images

- `infra/docker/bitnet-runtime.Dockerfile` produces a multi-stage image that
  compiles the llama.cpp binaries, builds the Python wheel via `python -m
  build`, and copies the artefacts into a slim runtime layer.  The pinned base
  image (`python:3.11.9-slim-bookworm`) and explicit labels make the build
  reproducible and traceable.
- `infra/docker/bitnet-edge.Dockerfile` targets edge or offline devices.  It
  retains a Python virtual environment in the final layer so administrators can
  upgrade packages without rebuilding the container.  The Dockerfile works with
  `docker buildx build --platform linux/amd64,linux/arm64` for multi-arch
  publishing.

Both Dockerfiles expose the `run_inference.py` entrypoint by default.  Override
the command to launch the HTTP server instead:

```bash
docker run --rm -p 8080:8080 bitnet-runtime \
  run_inference_server.py --model /models/bitnet.gguf --host 0.0.0.0
```

### 6.2 Kubernetes deployment (Terraform)

The Terraform module under `infra/terraform/` provisions:

1. A namespace to isolate BitNet resources.
2. A `Secret` that stores the model location (typically a signed URL or object
   store reference).
3. A `ConfigMap` holding CLI arguments so operators can tweak runtime flags
   without rebuilding images.
4. A `Deployment` + `Service` pairing that exposes the runtime on port 80 by
   default, including liveness and readiness probes that hit the `/health`
   endpoint.

Usage pattern:

```bash
cd infra/terraform
terraform init -backend=false
terraform apply -var="model_url=https://example.com/models/bitnet.gguf"
```

The deployment mounts the container image declared via `var.image` and runs
`run_inference_server.py --host 0.0.0.0 --model $(MODEL_URL)`, ensuring the same
command path as local usage.  Update `var.resource_limits` and
`var.resource_requests` to fit the cluster sizing.

### 6.3 Edge device bootstrap

For devices that cannot run containers, the refactored `setup_env.py` offers an
idempotent bootstrapper:

```bash
python setup_env.py --model-dir /models --hf-repo microsoft/BitNet-b1.58-2B-4T \
  --cache-dir /var/cache/bitnet
```

The script caches pip downloads under the supplied cache directory, tracks
completed stages in `.cache/bitnet/bootstrap-state.json`, and can be invoked
multiple times without repeating heavy work.  Combine it with the
`infra/docker/bitnet-edge.Dockerfile` build instructions to pre-provision
virtual environments that can be rsynced onto constrained devices.
