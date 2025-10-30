# Interface Requirements and Interaction Flows

This document captures the target user experience for the BitNet control surfaces across command-line, text user interface, graphical dashboard, and service APIs. The requirements emphasise three pillars: **monitoring**, **configuration**, and **theming**.

## Command-Line Interfaces (CLI)

### Monitoring
- Provide `--diagnostics` and `--show-config` flags for quick runtime health checks and inspection of the resolved preset values.
- Emit human-readable summaries (CPU detection, binary availability, active preset, safety adjustments) before execution.
- Support `--dry-run` to preview the llama.cpp invocation without launching it.

### Configuration
- Offer ergonomic presets (`balanced`, `latency`, `quality`) that seed inference parameters.
- Allow contextual help via `--explain <flag>` to describe advanced arguments and surface safety guidance.
- Enable interactive prompt capture when `--prompt` is omitted; users can enter multi-line prompts and submit with <kbd>Ctrl</kbd>+<kbd>D</kbd>.
- Apply validation guards (token limits, context bounds, GPU layer compatibility) before invoking the runtime.

### Theming
- Respect ANSI-aware output using colour-neutral formatting compatible with dark and light terminals.
- Provide an environment switch `BITNET_CLI_THEME` for future colour themes without breaking monochrome defaults.

### Interaction Flow
1. User optionally lists presets with `--list-presets`.
2. User invokes a preset or custom values; parser merges preset with overrides.
3. CLI performs diagnostics and validation; if issues arise, contextual help is suggested.
4. If no `--prompt` is given, an interactive editor opens. Submit with <kbd>Ctrl</kbd>+<kbd>D</kbd>, cancel with <kbd>Ctrl</kbd>+<kbd>C</kbd>.
5. Runtime executes (or previews) and streams progress to stdout.

## Text User Interface (TUI)

### Monitoring
- Surface rolling telemetry (tokens/sec, queue depth) polled from the REST API.
- Provide shortcut (<kbd>g</kbd>) to toggle between gauges (latency, utilisation) and logs.

### Configuration
- Embed preset selector with arrow keys and <kbd>Enter</kbd> to apply.
- Inline validation communicates errors beside the offending field.

### Theming
- Mirror dashboard palettes; allow live switching with <kbd>t</kbd> cycling through `system`, `dark`, and `light`.

### Interaction Flow
1. User launches TUI via `bitnet monitor` (future extension) which authenticates against API.
2. Landing pane shows active session summary and telemetry gauges.
3. Press <kbd>Tab</kbd> to move to configuration form, adjust values, confirm with <kbd>Ctrl</kbd>+<kbd>S</kbd>.
4. Changes propagate through API; success/failure toast displayed.
5. Theme toggles persist via API preference endpoint.

## Graphical Dashboard (Web GUI)

### Monitoring
- Real-time charts stream via Server-Sent Events (SSE) showing tokens/sec, queue depth, and temperature headroom.
- Status badges for model availability, preset name, and backend health.

### Configuration
- Modular panels for model settings, safety controls, and runtime toggles.
- Forms post to `/api/config` with optimistic UI updates and rollback on validation errors.
- Keyboard shortcuts: <kbd>Shift</kbd>+<kbd>L</kbd> toggles dark mode, <kbd>Shift</kbd>+<kbd>P</kbd> opens preset palette, <kbd>?</kbd> reveals contextual tips.

### Theming
- Default to dark mode with procedural accent palette computed from a hue seed.
- Support prefers-color-scheme detection with manual override persisted via `/api/theme`.

### Interaction Flow
1. Dashboard loads `/api/theme` to initialise theme state.
2. EventSource connection to `/api/metrics/stream` populates charts.
3. User modifies configuration; form submits to `/api/config` (PATCH semantics) and updates cards.
4. Theme or preset changes broadcast to other clients via SSE updates.

## REST / gRPC APIs

### Monitoring
- `GET /api/metrics` returns the latest snapshot; `/api/metrics/stream` provides SSE.
- Future gRPC service `StreamMetrics` replicates SSE contract for binary clients.

### Configuration
- `GET /api/config` yields the resolved configuration including preset metadata.
- `POST /api/config` accepts deltas with strict validation (range checks, GPU safety).
- Responses include actionable error messages tied to field names.

### Theming
- `GET /api/theme`/`POST /api/theme` manage theme preferences per session.
- Theme payload validated against supported modes (`system`, `dark`, `light`) and accent seeds (0–360 degrees).

### Interaction Flow
1. Client fetches configuration and theme to bootstrap UI.
2. Client optionally subscribes to metric stream for live updates.
3. Configuration changes post to API; invalid input yields 422 with detail payload.
4. Server broadcasts change events on SSE channel; clients reconcile local state.

---

These requirements align CLI, TUI, GUI, and programmatic interfaces around a consistent set of capabilities, ensuring BitNet operators can monitor, configure, and theme the system through their preferred modality.
