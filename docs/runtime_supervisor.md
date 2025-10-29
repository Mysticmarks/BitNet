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
