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

## 6. Turnkey deployments

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
