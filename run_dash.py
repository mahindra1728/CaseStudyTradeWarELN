"""Convenience runner for the Plotly Dash app.

Usage (from repo root):
  python run_dash.py

You can adjust host/port via env vars:
  DASH_HOST=0.0.0.0 DASH_PORT=8050 python run_dash.py
"""
from __future__ import annotations

import os

import threading
import webbrowser

from trade_war_eln.dashboard.dash_app import build_app


def main() -> None:
    host = os.getenv("DASH_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("DASH_PORT", "8050"))
    except ValueError:
        port = 8050
    app = build_app()

    # Open default browser shortly after startup
    def _open_browser() -> None:
        open_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        url = f"http://{open_host}:{port}/"
        try:
            webbrowser.open_new(url)
        except Exception:
            pass

    threading.Timer(1.0, _open_browser).start()

    # Dash >= 3 uses app.run; keep backward compatibility with run_server
    run = getattr(app, "run", None)
    if callable(run):
        run(debug=True, host=host, port=port)
    else:  # pragma: no cover - older Dash
        app.run_server(debug=True, host=host, port=port)


if __name__ == "__main__":
    main()
