# API Reference

_Generated from source. Run `python utils/generate_api_reference.py` after interface changes._

## Module: `bitnet/runtime.py`
Runtime orchestration helpers for BitNet Python entry points.

### Class `BitNetRuntime`
Utility wrapper that prepares and launches llama.cpp executables.

#### Methods
- `_build_inference_command(self, *, model: 'Path | str', prompt: 'str', n_predict: 'int', ctx_size: 'int', temperature: 'float', threads: 'Optional[int]', batch_size: 'int', gpu_layers: 'Optional[int]', conversation: 'bool', extra_args: 'Optional[Sequence[str]]') -> 'List[str]'` — No docstring provided.
- `_build_server_command(self, *, model: 'Path | str', host: 'str', port: 'int', prompt: 'Optional[str]', n_predict: 'int', ctx_size: 'int', temperature: 'float', threads: 'Optional[int]', batch_size: 'int', gpu_layers: 'Optional[int]', extra_args: 'Optional[Sequence[str]]') -> 'List[str]'` — No docstring provided.
- `_candidate_binaries(self, name: 'str') -> 'Iterable[Path]'` — No docstring provided.
- `_coerce_log_level(self, level: 'int | str') -> 'int'` — No docstring provided.
- `_ensure_port(self, value: 'int') -> 'None'` — No docstring provided.
- `_ensure_positive(self, value: 'int', field: 'str') -> 'None'` — No docstring provided.
- `_ensure_temperature(self, value: 'float') -> 'None'` — No docstring provided.
- `_execute(self, command: 'Sequence[str]') -> 'int'` — No docstring provided.
- `_missing_binary_hint(self, name: 'str', candidates: 'Iterable[Path]') -> 'str'` — No docstring provided.
- `_require_binary(self, name: 'str') -> 'Path'` — No docstring provided.
- `_require_model(self, model: 'Path | str') -> 'Path'` — No docstring provided.
- `_resolve_binary(self, name: 'str', *, required: 'bool') -> 'Optional[Path]'` — No docstring provided.
- `_resolve_gpu_layers(self, value: 'Optional[int]') -> 'Optional[int]'` — No docstring provided.
- `_resolve_threads(self, requested: 'Optional[int]') -> 'int'` — No docstring provided.
- `diagnostics(self, *, model: 'Optional[Path | str]' = None) -> 'RuntimeDiagnostics'` — No docstring provided.
- `run_inference(self, *, model: 'Path | str', prompt: 'str', n_predict: 'int' = 128, ctx_size: 'int' = 2048, temperature: 'float' = 0.8, threads: 'Optional[int]' = None, batch_size: 'int' = 1, gpu_layers: 'Optional[int]' = 0, conversation: 'bool' = False, dry_run: 'bool' = False, extra_args: 'Optional[Sequence[str]]' = None) -> 'Sequence[str] | int'` — Launch the llama-cli binary with validated arguments.
- `run_server(self, *, model: 'Path | str', host: 'str' = '127.0.0.1', port: 'int' = 8080, prompt: 'Optional[str]' = None, n_predict: 'int' = 4096, ctx_size: 'int' = 2048, temperature: 'float' = 0.8, threads: 'Optional[int]' = None, batch_size: 'int' = 1, gpu_layers: 'Optional[int]' = 0, dry_run: 'bool' = False, extra_args: 'Optional[Sequence[str]]' = None) -> 'Sequence[str] | int'` — Launch the llama-server binary with validated arguments.

### Class `RuntimeConfigurationError`
Raised when the runtime environment is not configured correctly.

### Class `RuntimeDiagnostics`
A structured summary of the runtime health check.

#### Methods
- `as_dict(self) -> 'Dict[str, object]'` — No docstring provided.

### Class `RuntimeLaunchError`
Raised when a runtime process exits unsuccessfully.

### Class `RuntimeTimeoutError`
Raised when a runtime process exceeds an execution deadline.

## Module: `bitnet/supervisor.py`
Asynchronous orchestration helpers for BitNet runtimes.

### Class `RuntimeResult`
Structured result for a supervised runtime execution.

### Class `RuntimeSupervisor`
Async orchestrator for launching llama.cpp binaries.

#### Methods
- `_launch(self, command: 'Sequence[str]', *, timeout: 'Optional[float]') -> 'RuntimeResult'` — No docstring provided.
- `close(self) -> 'None'` — Prevent new launches from being scheduled.
- `run_custom(self, command: 'Sequence[str]', *, timeout: 'Optional[float]' = None) -> 'RuntimeResult'` — Execute a fully specified command under supervision.
- `run_inference(self, *, timeout: 'Optional[float]' = None, **kwargs) -> 'RuntimeResult'` — No docstring provided.
- `run_server(self, *, timeout: 'Optional[float]' = None, **kwargs) -> 'RuntimeResult'` — No docstring provided.

## Module: `run_inference.py`
High-level CLI for launching BitNet inference runs.

### Module Functions
- `_build_parser() -> 'argparse.ArgumentParser'` — No docstring provided.
- `_parse_threads(value: 'str') -> 'Optional[int]'` — No docstring provided.
- `main(argv: 'Optional[list[str]]' = None) -> 'int'` — No docstring provided.

## Module: `run_inference_server.py`
HTTP server launcher for BitNet using llama.cpp.

### Module Functions
- `_build_parser() -> 'argparse.ArgumentParser'` — No docstring provided.
- `_parse_threads(value: 'str') -> 'Optional[int]'` — No docstring provided.
- `main(argv: 'Optional[list[str]]' = None) -> 'int'` — No docstring provided.

