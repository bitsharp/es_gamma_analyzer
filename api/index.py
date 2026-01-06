import os
import sys

# Ensure project root is importable when running as a Vercel Serverless Function.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app as _flask_app  # noqa: E402

# Vercel's Python runtime will serve this WSGI app.
app = _flask_app
