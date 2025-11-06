# Telemetry Dashboards

This guide captures the optional observability surfaces that consume the
BitNet runtime telemetry stream. Both dashboards subscribe to the
`RuntimeDiagnostics` payloads emitted by `BitNetRuntime` and
`RuntimeSupervisor`, allowing operators to pivot between local development and
production orchestration without code changes.

## 1. Telemetry Stream Contract

| Field | Source | Description |
| --- | --- | --- |
| `timestamp` | Runtime helpers | UTC timestamp when the event was recorded. |
| `component` | CLI or supervisor | Module name (`run_inference`, `RuntimeSupervisor`, etc.). |
| `event` | Runtime helpers | Phase marker such as `launch`, `completion`, `diagnostic`. |
| `metrics.tokens_per_second` | llama.cpp binaries | Smoothed tokens/sec for the most recent window. |
| `metrics.power_watts` | Optional sensor integration | Instantaneous power draw when available. |
| `diagnostics.missing_artifact` | Runtime helpers | Populated when prerequisite assets are missing. |
| `diagnostics.kernel_variant` | Kernel registry | Kernel family in use (`cpu-avx512`, `gpu-cuda`). |
| `context.correlation_id` | Supervisor | Stable identifier for batched runs. |

Events are published in JSON Lines format. Exporters can be toggled with the
`--telemetry-out` flag on the CLIs or by providing a file-like object to
`RuntimeSupervisor`.

## 2. CLI TUI Dashboard

The text-based dashboard is implemented as an optional curses application that
renders the telemetry stream in real time.

### 2.1 Configuration

```bash
python tools/telemetry_tui.py --input telemetry.jsonl --theme dark --refresh 200ms
```

- `--input` accepts a FIFO, file, or `-` for stdin.
- `--refresh` controls the redraw cadence; values under 150 ms can overwhelm slow
  terminals.
- `--columns` (optional) allows users to prune the layout for smaller displays.

### 2.2 Theming guidelines

| Theme | Palette | Usage |
| --- | --- | --- |
| `light` | Solarized light | Daytime reviews, low-contrast environments. |
| `dark` | Solarized dark | Default choice for terminals. |
| `release` | High-contrast amber/teal | Deploy-day monitoring; highlights regressions. |

Themes follow the repository-wide accessibility guidelines: minimum contrast
ratio 4.5:1 and never rely solely on colour to encode severity.

## 3. Web UI Dashboard

A lightweight web dashboard can be launched via:

```bash
python tools/telemetry_web.py --input telemetry.jsonl --theme release --listen 0.0.0.0:8081
```

- Serves a static SPA that connects to a WebSocket pushing telemetry updates.
- Aggregates per-run metrics (latency p50/p95, success rate) and overlays them
  against the configured baseline.
- Supports exporting snapshots in PNG or CSV form for incident reviews.

### 3.1 Integration with orchestration topology

The web UI expects the same telemetry stream consumed by the supervisor. When
running on Kubernetes, expose the stream via a sidecar that tails the JSONL file
into a socket. For edge deployments, the dashboard can run in polling mode using
HTTP long-poll requests.

## 4. Extensibility hooks

- **Custom themes**: add a JSON definition under `assets/dashboards/themes/` and
  reference it via `--theme custom-name`.
- **Metric enrichers**: implement a callable registered through
  `bitnet.telemetry.registry.register_enricher` to append domain-specific fields.
- **Alert routing**: pair the dashboards with Prometheus Alertmanager or a
  chat-ops webhook by enabling the optional exporter in
  `RuntimeSupervisor.configure_alerts`.

## 5. Validation checklist

1. Point both dashboards at the same telemetry stream and verify that refresh
   cadence stays below the 500 ms SLA.
2. Confirm that kernel variant tags match the binaries built in CI.
3. Capture a snapshot during release dry runs and attach it to the iteration log
   entry for traceability.

Consult `docs/documentation-review-checklist.md` before modifying dashboards to
ensure SRD and roadmap references remain aligned.
