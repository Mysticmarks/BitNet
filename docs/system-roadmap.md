# System Roadmap & Scope Register

_Last updated: 2025-05-23_

## Vision
Deliver a production-grade BitNet inference stack that runs efficiently across CPU, GPU, and future accelerator backends with cohesive tooling, documentation, and automation.

## Scope Register
| Capability | Current Status | Gaps / Follow-up | Owner(s) |
| --- | --- | --- | --- |
| CPU inference kernels (`src/ggml-bitnet-*.cpp`) | Stable | Continue profiling on AVX-512 and Apple Silicon targets | Kernels WG (L. Chen) |
| GPU inference (CUDA/ROCm) | Beta | Align feature flags with CPU runtime, publish perf baselines | GPU WG (M. Rivera) |
| Python runtime orchestration | Stable | Expand diagnostics with memory / disk pre-flight checks | Tooling WG (S. Patel) |
| Async supervisor (`RuntimeSupervisor`) | Experimental | Add retry policies and structured tracing exporters | Tooling WG (S. Patel) |
| Model conversion utilities | Stable | Support streaming conversion for very large checkpoints | Tooling WG (R. Nguyen) |
| Deployment workflows (`docs/deployment.md`) | Draft | Validate Kubernetes + container recipes | DevOps WG (A. Silva) |
| Testing & CI | Partial | Introduce GitHub Actions for lint + unit tests | DevOps WG (A. Silva) |
| Documentation | Growing | Integrate architecture diagrams, keep README & SRD synced | Docs WG (K. Morgan) |

## Current Limitations

- GPU diagnostics are absent from the runtime health checks; CLI users must confirm CUDA/ROCm readiness manually.
- There is no automated deployment pipeline. Container images and IaC samples are tracked as future work.
- Async supervisor tracing hooks are placeholders until the observability milestone lands.

## Near-Term Milestones (0-3 months)
1. **Docs Review Guardrail** (Owner: K. Morgan) – Enforce documentation sync via CI (landed in this iteration, maintain going forward).
2. **Diagnostics Enhancements** (Owner: S. Patel) – Extend `RuntimeDiagnostics` with GPU capability checks and disk space validation.
3. **Automation Baseline** (Owner: A. Silva) – Add GitHub Actions workflow for Python tests and a CMake configure smoke test.

## Mid-Term Milestones (3-6 months)
1. **GPU Parity Push** (Owner: M. Rivera) – Finalise CUDA kernels for TL2 layout, ship reproducible benchmarks.
2. **Observability** (Owner: S. Patel) – Expose structured telemetry hooks for inference latency, throughput, and energy metrics.
3. **Extensibility** (Owner: S. Patel) – Define plugin API for scheduler policies in `RuntimeSupervisor`.

## Long-Term Milestones (6-12 months)
1. **Docs Refresh** (Owner: K. Morgan) – Keep architecture diagrams and API references generated each quarter to reflect code changes.
2. **Multi-node Scaling** (Owner: L. Chen) – Investigate distributed runtime for sharded models using message-passing fabric.
3. **Adaptive Optimisation** (Owner: L. Chen) – Integrate profile-guided tuning and caching for hot prompts.
4. **NPU Support** (Owner: M. Rivera) – Prototype kernels for Windows Copilot+ NPUs and Apple ANE.

## Iteration Notes
- This iteration established the authoritative SRD and roadmap scaffolding.
- Follow-up work should record progress in `docs/iteration-log.md` and refresh timestamps here when milestones shift.
