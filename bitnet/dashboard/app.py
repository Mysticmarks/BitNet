"""FastAPI application for the BitNet dashboard."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import EventSourceResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from ..cli_utils import PRESETS, resolve_preset, validate_configuration
from .telemetry import TelemetryStream
from .theme import ThemePreferences, ThemeMode, validate_theme_mode

STATIC_DIR = Path(__file__).parent / "static"


class ConfigState(BaseModel):
    model: str | None = None
    n_predict: int = Field(256, ge=1, le=8192)
    ctx_size: int = Field(2048, ge=256, le=16384)
    temperature: float = Field(0.8, gt=0.0, le=1.5)
    batch_size: int = Field(2, ge=1, le=32)
    gpu_layers: int = Field(0, ge=0, le=512)
    preset: Optional[str] = Field("balanced")

    class Config:
        validate_assignment = True


class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    n_predict: Optional[int] = Field(None, ge=1, le=8192)
    ctx_size: Optional[int] = Field(None, ge=256, le=16384)
    temperature: Optional[float] = Field(None, gt=0.0, le=1.5)
    batch_size: Optional[int] = Field(None, ge=1, le=32)
    gpu_layers: Optional[int] = Field(None, ge=0, le=512)
    preset: Optional[str] = None

    class Config:
        extra = "forbid"

    @validator("preset")
    def validate_preset(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in PRESETS:
            raise ValueError("Unknown preset")
        return value


class ThemeUpdate(BaseModel):
    mode: Optional[ThemeMode] = Field(None)
    accent_seed: Optional[int] = Field(None, ge=0, le=360)

    class Config:
        extra = "forbid"


class DashboardState:
    def __init__(self) -> None:
        self.config = ConfigState()
        self.theme = ThemePreferences()
        self.telemetry = TelemetryStream()
        self._lock = asyncio.Lock()

    async def update_config(self, payload: ConfigUpdate) -> Dict[str, Any]:
        updates = payload.dict(exclude_unset=True)
        resolved_preset = updates.get("preset", self.config.preset)
        apply_existing_defaults = resolved_preset == self.config.preset or resolved_preset is None
        overrides: Dict[str, Any] = {}
        for field in ("n_predict", "ctx_size", "temperature", "batch_size", "gpu_layers"):
            if field in updates:
                overrides[field] = updates[field]
            elif apply_existing_defaults:
                overrides[field] = getattr(self.config, field)
        resolved = resolve_preset(resolved_preset, overrides)
        try:
            validate_configuration(resolved)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if resolved.values.get("gpu_layers") is None:
            resolved.values["gpu_layers"] = self.config.gpu_layers
        resolved.values.pop("preset", None)
        for key, value in resolved.values.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config.preset = resolved.preset
        if "model" in updates:
            self.config.model = updates["model"]
        return {
            "config": self.config.dict(),
            "adjusted": resolved.adjusted,
        }

    async def update_theme(self, payload: ThemeUpdate) -> Dict[str, Any]:
        updates = payload.dict(exclude_unset=True)
        try:
            if "mode" in updates and updates["mode"] is not None:
                validate_theme_mode(updates["mode"])
            self.theme.update(
                mode=updates.get("mode", None),
                accent_seed=updates.get("accent_seed", None),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"theme": self.theme.as_dict()}


def create_app() -> FastAPI:
    app = FastAPI(title="BitNet Dashboard", version="1.0.0")
    state = DashboardState()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await state.telemetry.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await state.telemetry.stop()

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/metrics")
    async def metrics() -> Dict[str, Any]:
        sample = await state.telemetry.snapshot()
        return {"metrics": sample.as_dict()}

    @app.get("/api/metrics/stream")
    async def metrics_stream() -> EventSourceResponse:
        async def event_generator() -> AsyncGenerator[dict[str, str], None]:
            async for sample in state.telemetry.subscribe():
                payload = json.dumps(sample.as_dict())
                yield {"event": "telemetry", "data": payload}

        return EventSourceResponse(event_generator())

    @app.get("/api/config")
    async def config() -> Dict[str, Any]:
        return {"config": state.config.dict()}

    @app.post("/api/config")
    async def update_config(payload: ConfigUpdate) -> Dict[str, Any]:
        async with state._lock:
            return await state.update_config(payload)

    @app.get("/api/theme")
    async def theme() -> Dict[str, Any]:
        return {"theme": state.theme.as_dict()}

    @app.post("/api/theme")
    async def update_theme(payload: ThemeUpdate) -> Dict[str, Any]:
        async with state._lock:
            return await state.update_theme(payload)

    return app
