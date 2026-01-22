"""Telemetry helpers for BitNet runtime and CLI entrypoints."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, Mapping, Optional

from .runtime import TelemetryEvent

__all__ = [
    "TelemetryRecord",
    "TelemetryWriter",
    "build_cli_telemetry_sink",
]


@dataclass(frozen=True)
class TelemetryRecord:
    """Normalized telemetry payload written by CLI helpers."""

    timestamp: float
    component: str
    event: str
    attributes: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "event": self.event,
            "attributes": dict(self.attributes),
        }


class TelemetryWriter:
    """Thread-safe JSONL writer for telemetry records."""

    def __init__(self, handle: IO[str], *, component: str) -> None:
        self._handle = handle
        self._component = component
        self._lock = threading.Lock()

    @property
    def component(self) -> str:
        return self._component

    def write_record(self, record: TelemetryRecord) -> None:
        payload = json.dumps(record.as_dict(), ensure_ascii=False)
        with self._lock:
            self._handle.write(payload + "\n")
            self._handle.flush()

    def __call__(self, event: TelemetryEvent) -> None:
        record = TelemetryRecord(
            timestamp=event.timestamp,
            component=self._component,
            event=event.name,
            attributes=event.attributes,
        )
        self.write_record(record)

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def build_cli_telemetry_sink(
    *,
    output_path: Path,
    component: str,
    header: Optional[Mapping[str, object]] = None,
) -> TelemetryWriter:
    """Create a TelemetryWriter and optionally emit a header record."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("a", encoding="utf-8")
    writer = TelemetryWriter(handle, component=component)
    if header is not None:
        writer.write_record(
            TelemetryRecord(
                timestamp=time.time(),
                component=component,
                event="telemetry.start",
                attributes=dict(header),
            )
        )
    return writer
