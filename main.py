"""
Vercel entrypoint.

Vercel looks for a Python entrypoint at the repository root (app.py,
index.py, server.py, main.py, wsgi.py or asgi.py) and loads the top-level
`app` variable from it, then routes every request to that application.

The FastAPI application itself lives in api/server.py so that the same file
still runs locally with:

    uvicorn api.server:app --host 0.0.0.0 --port 8000

This module only re-exports it. A root-level entrypoint is required here
because the Streamlit UI at app/app.py would otherwise be picked up first by
Vercel's entrypoint search.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.server import app  # noqa: E402,F401

__all__ = ["app"]
