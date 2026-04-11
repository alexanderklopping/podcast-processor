"""Long-lived worker mode for Render-style deployments."""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .pipeline import execute_pipeline_once

logger = logging.getLogger("mediaverwerker")

DEFAULT_INTERVAL_SECONDS = 3600


def parse_positive_int(value: Optional[str], default: int) -> int:
    """Parse a positive integer from config, falling back to the default."""
    if value is None or value == "":
        return default

    try:
        parsed = int(value)
    except ValueError:
        logger.warning(f"Invalid integer value '{value}', using default {default}")
        return default

    if parsed <= 0:
        logger.warning(f"Non-positive integer value '{value}', using default {default}")
        return default

    return parsed


def iso_now() -> str:
    """Return a UTC ISO timestamp for status reporting."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ServiceState:
    """In-memory status exposed through the health endpoint."""

    started_at: str = field(default_factory=iso_now)
    last_run_started_at: Optional[str] = None
    last_run_finished_at: Optional[str] = None
    last_exit_code: Optional[int] = None
    run_count: int = 0
    running: bool = False

    def begin_run(self) -> None:
        self.run_count += 1
        self.running = True
        self.last_run_started_at = iso_now()

    def finish_run(self, exit_code: int) -> None:
        self.running = False
        self.last_exit_code = exit_code
        self.last_run_finished_at = iso_now()

    def snapshot(self) -> dict:
        return {
            "started_at": self.started_at,
            "running": self.running,
            "run_count": self.run_count,
            "last_run_started_at": self.last_run_started_at,
            "last_run_finished_at": self.last_run_finished_at,
            "last_exit_code": self.last_exit_code,
            "healthy": self.last_exit_code in (None, 0),
        }


class HealthcheckHandler(BaseHTTPRequestHandler):
    """Minimal JSON status endpoint for platform health checks."""

    state: ServiceState

    def do_GET(self):
        if self.path not in ("/", "/health", "/healthz"):
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(self.state.snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def start_health_server(state: ServiceState, port: int) -> ThreadingHTTPServer:
    """Start a small HTTP server so Render can see a live process."""

    class Handler(HealthcheckHandler):
        pass

    Handler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server listening on port {port}")
    return server


def run_daemon(interval_seconds: Optional[int] = None, port: Optional[int] = None) -> None:
    """Run the pipeline forever with a fixed delay between runs."""
    resolved_interval = interval_seconds or parse_positive_int(
        os.getenv("PROCESS_INTERVAL_SECONDS"),
        DEFAULT_INTERVAL_SECONDS,
    )
    resolved_port = port
    if resolved_port is None:
        resolved_port = parse_positive_int(os.getenv("PORT"), 0)

    state = ServiceState()
    if resolved_port:
        start_health_server(state, resolved_port)

    logger.info(f"Daemon mode enabled, running every {resolved_interval} seconds")

    while True:
        state.begin_run()
        exit_code = execute_pipeline_once()
        state.finish_run(exit_code)

        if exit_code == 0:
            logger.info(f"Pipeline run completed, sleeping for {resolved_interval} seconds")
        else:
            logger.warning(f"Pipeline run failed with exit code {exit_code}, retrying in {resolved_interval} seconds")

        time.sleep(resolved_interval)
