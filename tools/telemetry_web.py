"""Minimal web dashboard skeleton for BitNet telemetry JSONL streams."""

from __future__ import annotations

import argparse
import json
import socketserver
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterable


def _iter_lines(path: Path) -> Iterable[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _tail_events(path: Path, *, limit: int) -> list[dict[str, object]]:
    lines = _iter_lines(path)
    events = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


HTML_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <title>BitNet Telemetry Dashboard (Skeleton)</title>
    <style>
      body { font-family: sans-serif; background: #0f1115; color: #e6e6e6; padding: 20px; }
      .event { background: #1c1f26; padding: 10px; margin-bottom: 10px; border-radius: 6px; }
      .muted { color: #8c8c8c; }
    </style>
  </head>
  <body>
    <h1>BitNet Telemetry Dashboard (Skeleton)</h1>
    <p class="muted">Auto-refreshing JSONL view.</p>
    <div id="events"></div>
    <script>
      async function refresh() {
        const response = await fetch('/data');
        const payload = await response.json();
        const container = document.getElementById('events');
        container.innerHTML = '';
        payload.events.forEach((event) => {
          const div = document.createElement('div');
          div.className = 'event';
          div.textContent = JSON.stringify(event);
          container.appendChild(div);
        });
      }
      refresh();
      setInterval(refresh, 1000);
    </script>
  </body>
</html>
"""


class TelemetryHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._write_html()
        elif self.path.startswith("/data"):
            self._write_data()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _write_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_data(self) -> None:
        events = _tail_events(self.server.telemetry_path, limit=self.server.tail)  # type: ignore[attr-defined]
        payload = json.dumps({"events": events}, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class TelemetryServer(HTTPServer):
    def __init__(self, server_address: tuple[str, int], handler, telemetry_path: Path, tail: int) -> None:
        super().__init__(server_address, handler)
        self.telemetry_path = telemetry_path
        self.tail = tail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a minimal telemetry dashboard over HTTP.")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL telemetry file")
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1:8081",
        help="Bind address in host:port form",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Number of events to expose",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="dark",
        help="Theme name (placeholder)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    host, _, port_str = args.listen.rpartition(":")
    host = host or "127.0.0.1"
    port = int(port_str or "8081")
    server = TelemetryServer((host, port), TelemetryHandler, args.input, args.tail)
    print(f"Serving telemetry dashboard on http://{host}:{port} (theme={args.theme})")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
