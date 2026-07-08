"""CUDAlpha dashboard — Plotly Dash multi-page app (Python-first).

Foundation: Dash + dash-bootstrap-components (responsive Bootstrap layouts in
pure Python). Visual direction is an intentional Day-5 pass — ADAPT open-source
dashboard patterns (Dash Flightdeck / Volt Bootstrap 5) with attribution, and
check each project's license before copying any code or assets. Do not paste
template code without checking its license.

Pages live in dashboard/pages/ and self-register (Overview, Benchmark Explorer,
CPU vs GPU, Profiling, Job Runs, Validation). Run:  python dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on the path so pages can import `dashboard.data` and the harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server  # for gunicorn / container deployment

_sidebar = dbc.Nav(
    [dbc.NavLink(p["name"], href=p["relative_path"], active="exact")
     for p in sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))],
    vertical=True, pills=True,
)

app.layout = dbc.Container(
    [
        html.H2("CUDAlpha", className="mt-3"),
        html.P("GPU performance-engineering testbed", className="text-muted"),
        dbc.Row([
            dbc.Col(_sidebar, width=3, lg=2),
            dbc.Col(dash.page_container, width=9, lg=10),
        ]),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
