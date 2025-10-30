"""Theme utilities for the BitNet dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThemeMode = Literal["system", "dark", "light"]


@dataclass(slots=True)
class ThemePreferences:
    """Persisted theme settings shared across clients."""

    mode: ThemeMode = "dark"
    accent_seed: int = 264

    def update(self, *, mode: ThemeMode | None = None, accent_seed: int | None = None) -> None:
        if mode is not None:
            self.mode = mode
        if accent_seed is not None:
            if not 0 <= accent_seed <= 360:
                raise ValueError("accent_seed must be between 0 and 360")
            self.accent_seed = accent_seed

    def as_dict(self) -> dict[str, int | str]:
        return {"mode": self.mode, "accentSeed": self.accent_seed}


def validate_theme_mode(mode: str) -> ThemeMode:
    if mode not in {"system", "dark", "light"}:
        raise ValueError("Unsupported theme mode")
    return mode  # type: ignore[return-value]
