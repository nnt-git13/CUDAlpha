"""Benchmark Explorer — a filterable table of every raw result artifact."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:
    from data import empty_state, load_raw

dash.register_page(__name__, path="/explorer", name="Benchmark Explorer", order=2)

COLUMNS = ["workload", "device", "backend", "size_label", "median_ms", "p95_ms",
           "std_ms", "throughput", "peak_mem_mb", "gpu_util_pct",
           "speedup_vs_cpu", "passed_validation", "job_id", "run_id"]


def _options(df, col):
    if col not in df:
        return [{"label": "all", "value": "__all__"}]
    return [{"label": "all", "value": "__all__"}] + [
        {"label": str(v), "value": str(v)} for v in sorted(df[col].dropna().unique())
    ]


def layout(**kwargs):
    df = load_raw()
    if df.empty:
        return dbc.Container([html.H3("Benchmark Explorer", className="mb-3"),
                              empty_state("No benchmark artifacts found.")], fluid=True)
    return dbc.Container([
        html.H3("Benchmark Explorer", className="mb-3"),
        dbc.Row([
            dbc.Col([html.Label("Workload"),
                     dcc.Dropdown(id="ex-workload", options=_options(df, "workload"),
                                  value="__all__", clearable=False)], md=4),
            dbc.Col([html.Label("Device"),
                     dcc.Dropdown(id="ex-device", options=_options(df, "device"),
                                  value="__all__", clearable=False)], md=4),
            dbc.Col([html.Label("Backend"),
                     dcc.Dropdown(id="ex-backend", options=_options(df, "backend"),
                                  value="__all__", clearable=False)], md=4),
        ], className="mb-3"),
        dash_table.DataTable(
            id="ex-table",
            columns=[{"name": c, "id": c} for c in COLUMNS],
            page_size=20, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": 12, "padding": "4px 8px"},
            style_header={"fontWeight": "bold"},
        ),
    ], fluid=True)


@callback(
    Output("ex-table", "data"),
    Input("ex-workload", "value"),
    Input("ex-device", "value"),
    Input("ex-backend", "value"),
)
def _filter(workload, device, backend):
    df = load_raw()
    if df.empty:
        return []
    for col, val in [("workload", workload), ("device", device), ("backend", backend)]:
        if val and val != "__all__" and col in df:
            df = df[df[col].astype(str) == val]
    cols = [c for c in COLUMNS if c in df.columns]
    df = df[cols].copy()
    for c in ("median_ms", "p95_ms", "std_ms", "throughput", "peak_mem_mb",
              "gpu_util_pct", "speedup_vs_cpu"):
        if c in df:
            df[c] = df[c].round(3)
    return df.to_dict("records")
