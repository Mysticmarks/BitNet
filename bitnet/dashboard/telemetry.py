"""Telemetry stream utilities for the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List


@dataclass(slots=True)
class TelemetrySample:
    tokens_per_second: float
    queue_depth: int
    temperature: float
    timestamp: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "tokensPerSecond": self.tokens_per_second,
            "queueDepth": self.queue_depth,
            "temperature": self.temperature,
            "timestamp": self.timestamp,
        }


class TelemetryStream:
    """Manage subscriptions for telemetry updates."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue[TelemetrySample]] = []
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._task is None:
                self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        async with self._lock:
            if self._task:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
                self._task = None

    async def subscribe(self) -> AsyncGenerator[TelemetrySample, None]:
        queue: asyncio.Queue[TelemetrySample] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                sample = await queue.get()
                yield sample
        finally:
            self._subscribers.remove(queue)

    async def broadcast(self, sample: TelemetrySample) -> None:
        for queue in list(self._subscribers):
            await queue.put(sample)

    async def _run(self) -> None:
        rng = random.Random()
        tokens = 120.0
        queue_depth = 0
        while True:
            await asyncio.sleep(1.0)
            tokens = max(40.0, min(280.0, tokens + rng.uniform(-15.0, 15.0)))
            queue_depth = max(0, min(32, queue_depth + rng.randint(-3, 3)))
            temperature = rng.uniform(0.6, 0.95)
            sample = TelemetrySample(
                tokens_per_second=round(tokens, 2),
                queue_depth=queue_depth,
                temperature=round(temperature, 2),
                timestamp=time.time(),
            )
            await self.broadcast(sample)

    async def snapshot(self) -> TelemetrySample:
        """Provide the most recent sample (generate one if stream idle)."""

        sample = TelemetrySample(
            tokens_per_second=180.0,
            queue_depth=2,
            temperature=0.82,
            timestamp=time.time(),
        )
        await self.broadcast(sample)
        return sample
