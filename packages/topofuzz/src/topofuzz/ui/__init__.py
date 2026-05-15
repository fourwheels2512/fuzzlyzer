"""topofuzz.ui — 5th-grader-runnable web UI over the topofuzz CLI.

Local-only FastAPI + Jinja2 + HTMX app that wraps `topofuzz run` as a
subprocess. Non-technical users pick a domain workflow, write scenarios
in plain English, and get an embedded report.html. Real spend is
displayed (divided by the 2.75x topofuzz-meter inflation), not raw
topofuzz units.

Entry point: `python -m topofuzz ui` (see topofuzz.cli:ui).
"""

from topofuzz.ui.app import build_app

__all__ = ["build_app"]
