"""Run the SoulForge API server: ``python -m app.server``.

Mirrors the TUI/CLI bootstrap, loads the model up front (blocking, like the
CLI), then serves the FastAPI app with uvicorn. Intended to run inside WSL; the
Windows GUI connects over localhost.
"""

from __future__ import annotations

import argparse

import uvicorn

from app.core.chat_controller import ChatController
from app.main import bootstrap, _log_startup_report, _print_startup_issues
from app.server.api import create_app
from app.utils.guards import format_startup_error
from app.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="SoulForge local API server")
    parser.add_argument("--host", default=None, help="Override server.host")
    parser.add_argument("--port", type=int, default=None, help="Override server.port")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config, report = bootstrap(args.config)
        _print_startup_issues(report)
        _log_startup_report(report)

        controller = ChatController(config)
        print("Loading model...")
        controller.load()

        app = create_app(controller)
        host = args.host or config.server.host
        port = args.port or config.server.port
        get_logger("server").info("Serving SoulForge API on %s:%s", host, port)
        print(f"SoulForge API listening on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as error:  # noqa: BLE001
        print(format_startup_error(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
