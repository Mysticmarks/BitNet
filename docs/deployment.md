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
pre-requisites are missing.  Extend these tests with integration checks that
spawn actual llama.cpp processes when you have the binaries and GGUF artefacts
available in CI.

## 5. Future extensions

- Add watchdog processes that monitor token throughput and restart the binary
  on stalls.
- Emit JSON diagnostics via structured logging for richer dashboards.
- Introduce GPU-aware runtime helpers that detect the available accelerators and
  pick the optimal binary automatically.
