# System Roadmap & Scope Register

_Last updated: 2026-01-22_

## Vision
Deliver a production-grade BitNet inference stack that runs efficiently across CPU, GPU, and future accelerator backends with cohesive tooling, documentation, and automation.

## Scope Register
| Capability | Current Status | Gaps / Follow-up | Owner(s) |
| --- | --- | --- | --- |
| CPU inference kernels (`src/ggml-bitnet-*.cpp`) | Stable | Continue profiling on AVX-512 and Apple Silicon targets | Kernels WG |
| GPU inference (CUDA/ROCm) | Beta | Align feature flags with CPU runtime, publish perf baselines | GPU WG |
| Python runtime orchestration | Stable | Expand diagnostics with memory / disk pre-flight checks | Tooling WG |
| Async supervisor (`RuntimeSupervisor`) | Experimental | Add retry policies and structured tracing exporters | Tooling WG |
| Model conversion utilities | Stable | Support streaming conversion for very large checkpoints | Tooling WG |
| Deployment workflows (`docs/deployment.md`) | Draft | Validate Kubernetes + container recipes | DevOps WG |
| Testing & CI | Partial | Introduce GitHub Actions for lint + unit tests | DevOps WG |
| Documentation | Growing | Integrate architecture diagrams, keep README & SRD synced | Docs WG |
| Omnimodal runtime (text/image/audio/video) | Planned | Define modality schema, adapters, and codec validation paths | Runtime WG |
| 4K UHD 30fps video output | Planned | Add frame pacing controls, telemetry, and GPU capability checks | Runtime WG |

## Near-Term Milestones (0-3 months)
1. **Automation Baseline**: Add GitHub Actions workflow for Python tests and CMake configure step.
2. **Diagnostics Enhancements**: Extend `RuntimeDiagnostics` with GPU capability checks and disk space validation.
3. **GPU Parity Push**: Finalise CUDA kernels for TL2 layout, ship reproducible benchmarks.
4. **Omnimodal Schema Draft**: Publish media payload schema and adapter contracts for text/image/audio/video input.
5. **4K Output Telemetry**: Add FPS and frame drop telemetry fields to runtime diagnostics.

## Mid-Term Milestones (3-6 months)
1. **Observability**: Expose structured telemetry hooks for inference latency, throughput, and energy metrics.
2. **Extensibility**: Define plugin API for scheduler policies in `RuntimeSupervisor`.
3. **Docs Refresh**: Publish architecture diagram set in `/docs/architecture/`.
4. **Omnimodal Adapters**: Deliver reference adapters for image/audio ingestion and video frame assembly.

## Long-Term Milestones (6-12 months)
1. **Multi-node Scaling**: Investigate distributed runtime for sharded models using message-passing fabric.
2. **Adaptive Optimisation**: Integrate profile-guided tuning and caching for hot prompts.
3. **NPU Support**: Prototype kernels for Windows Copilot+ NPUs and Apple ANE.
4. **4K UHD Delivery**: Validate end-to-end 4K UHD 30fps delivery on supported GPU tiers.

## Iteration Notes
- This iteration established the authoritative SRD and roadmap scaffolding.
- Follow-up work should record progress in `docs/iteration-log.md` and refresh timestamps here when milestones shift.
