from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import pytest


@dataclass
class RecordedMetric:
    name: str
    value: float
    unit: str


class PerformanceState:
    def __init__(self) -> None:
        self.metrics: List[RecordedMetric] = []
        self.thresholds: Dict[str, float] = {}
        self.log_path: Path | None = None

    def record(self, name: str, value: float, unit: str) -> None:
        self.metrics.append(RecordedMetric(name=name, value=float(value), unit=unit))


class PerformanceRecorder:
    def __init__(self, state: PerformanceState) -> None:
        self._state = state

    def record(self, name: str, value: float, unit: str = "value") -> None:
        self._state.record(name, value, unit)


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover - pytest hook
    group = parser.getgroup("bitnet-perf")
    group.addoption(
        "--perf-log",
        action="store",
        default=None,
        help="Write captured performance metrics to the specified JSON file",
    )
    group.addoption(
        "--perf-threshold",
        action="append",
        default=[],
        help="Set minimum thresholds for metrics (format: name=value)",
    )


def _parse_thresholds(raw: list[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for entry in raw:
        if "=" not in entry:
            raise pytest.UsageError(f"Invalid --perf-threshold value: {entry!r}")
        name, value = entry.split("=", 1)
        thresholds[name.strip()] = float(value.strip())
    return thresholds


def pytest_configure(config: pytest.Config) -> None:  # pragma: no cover - pytest hook
    state = PerformanceState()
    log_path = config.getoption("perf_log")
    if log_path:
        state.log_path = Path(log_path)
    state.thresholds = _parse_thresholds(config.getoption("perf_threshold"))
    config._bitnet_perf_state = state  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # pragma: no cover - pytest hook
    state: PerformanceState | None = getattr(session.config, "_bitnet_perf_state", None)  # type: ignore[attr-defined]
    if state is None:
        return

    if state.log_path:
        state.log_path.parent.mkdir(parents=True, exist_ok=True)
        state.log_path.write_text(json.dumps([asdict(metric) for metric in state.metrics], indent=2), encoding="utf-8")

    failing: list[str] = []
    for metric in state.metrics:
        minimum = state.thresholds.get(metric.name)
        if minimum is not None and metric.value < minimum:
            failing.append(f"{metric.name}={metric.value:.2f} < {minimum:.2f}")
    if failing:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("Performance regressions detected:")
            for item in failing:
                reporter.write_line(f"  - {item}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def perf_recorder(request: pytest.FixtureRequest) -> PerformanceRecorder:
    state: PerformanceState | None = getattr(request.config, "_bitnet_perf_state", None)  # type: ignore[attr-defined]
    if state is None:
        state = PerformanceState()
        request.config._bitnet_perf_state = state  # type: ignore[attr-defined]
    return PerformanceRecorder(state)
