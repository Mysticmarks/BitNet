# Runtime Supervisor

The Python wrapper around the BitNet llama.cpp executables now includes an
asynchronous orchestration layer that targets production-style workloads.  The
`RuntimeSupervisor` wraps the existing `BitNetRuntime` validation logic and adds
bounded concurrency, timeout control, and structured telemetry for each process
launch.

## Why use the supervisor?

The original helpers exposed only blocking `subprocess.run` calls.  This was
sufficient for the CLI entry points, but it prevented services from scheduling
multiple inference jobs, enforcing deadlines, or aggregating output for
monitoring dashboards.  The supervisor solves those problems by:

* limiting the number of concurrent launches via an asyncio semaphore
* capturing stdout and stderr so they can be pushed to logging or tracing
  backends
* cancelling and killing child processes that overrun a configured timeout
* exposing start/completion timestamps for latency calculations

## Basic usage

```python
import asyncio
from bitnet import BitNetRuntime, RuntimeSupervisor


async def main():
    runtime = BitNetRuntime(build_dir="build")
    async with RuntimeSupervisor(runtime, concurrency=runtime.cpu_count) as sup:
        result = await sup.run_inference(
            model="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
            prompt="Summarise the release notes in one paragraph",
            timeout=30,
        )
    print(result.stdout)


asyncio.run(main())
```

## Handling timeouts

Every orchestration method accepts a `timeout` argument.  When the deadline is
reached the supervisor automatically kills the child process, awaits its exit,
and raises `RuntimeTimeoutError`.  This makes it straightforward to integrate
BitNet inference into task queues or API gateways that require deterministic
latency bounds.

```python
from bitnet import RuntimeTimeoutError


try:
    await supervisor.run_inference(..., timeout=10)
except RuntimeTimeoutError:
    logger.warning("bitnet job exceeded the 10s budget")
```

## Custom commands

Although the supervisor leans on `BitNetRuntime` to validate inference and
server invocations, you can also schedule arbitrary commands that still benefit
from concurrency limits and timeout handling:

```python
await supervisor.run_custom(["python", "-m", "http.server", "8080"], timeout=5)
```

This is useful when chaining preprocessing or postprocessing steps alongside
BitNet inference.

## Dynamic scheduling

Production clusters rarely treat every inference request equally.  The
`RuntimeSupervisor` now understands multiple scheduling policies and optional
resource reservations:

* ``weighted-fair`` (default) balances launches across logical backends using
  the configured weights.
* ``priority`` executes the lowest priority value first which is useful for
  latency-sensitive, mission-critical jobs.
* ``resource-aware`` honours logical resource pools such as GPU slots.  Tasks
  may declare the resources they need and will not be dispatched until the pool
  can satisfy the request.

```python
supervisor = RuntimeSupervisor(
    runtime,
    scheduling_policy="resource-aware",
    resource_limits={"gpu": 2},
)

# Blocks until at least one GPU slot is available
await supervisor.run_inference(
    model=model_path,
    prompt="Describe the architecture",
    resources={"gpu": 1},
)
```

The scheduler automatically releases resources even when retries, fallbacks, or
errors occur which makes it safe to layer chaos testing on top.

## Distributed drivers and tracing

The supervisor ships with drivers that expose scheduling over gRPC and Ray.
Both are suitable for production thanks to authentication, TLS, and tracing
hooks:

```python
runtime = BitNetRuntime(build_dir="build")
supervisor = RuntimeSupervisor(runtime)
grpc_driver = GRPCSupervisorDriver(
    supervisor,
    host="0.0.0.0",
    port=50051,
    tls_config={"server_cert": "cert.pem", "server_key": "key.pem"},
    auth_token="shared-secret",
)
await grpc_driver.start()

# Client side
result = await grpc_driver.submit_task(["python", "-c", "print('hello')"], options={"timeout": 5})
```

Ray users can spin up lightweight workers that reuse the same scheduling policy
while emitting OpenTelemetry spans for each execution:

```python
ray_driver = RaySupervisorDriver(
    lambda: BitNetRuntime(build_dir="build"),
    num_workers=4,
    shutdown_ray=True,
    init_kwargs={"local_mode": True},
)
result = await ray_driver.submit_task(["python", "-c", "print('distributed')"], timeout=5)
```

## Metrics and observability

`BitNetRuntime` exposes metrics that can be scraped by Prometheus or forwarded
through OpenTelemetry exporters.  The samples include execution counters as
well as the outcome of the extended health probes (disk, memory, GPU pressure,
NUMA topology, and sandboxing):

```python
runtime = BitNetRuntime(build_dir="build")
text = runtime.prometheus_metrics()
otel_payload = runtime.opentelemetry_metrics()
```

The Prometheus endpoint emits stable metric names prefixed with
``bitnet_runtime_`` while the OTLP payload is ready to serialize to JSON and
forward to collector agents.
