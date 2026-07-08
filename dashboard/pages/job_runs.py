"""Job Runs — one row per Slurm array task (job_id), with counts and status."""
import datetime as dt

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:
    from data import empty_state, load_raw

dash.register_page(__name__, path="/job-runs", name="Job Runs", order=5)


def layout(**kwargs):
    df = load_raw()
    if df.empty:
        return dbc.Container([html.H3("Job Runs", className="mb-3"),
                              empty_state("No benchmark artifacts found.")], fluid=True)

    g = df.copy()
    if "job_id" not in g:
        g["job_id"] = "local"

    def _agg(sub):
        gpu = sub[sub.get("device") == "gpu"] if "device" in sub else sub.iloc[0:0]
        validated = gpu["passed_validation"].dropna() if "passed_validation" in gpu else []
        n_fail = int((~validated.astype(bool)).sum()) if len(validated) else 0
        ts = sub["timestamp"].max() if "timestamp" in sub else None
        when = dt.datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds") if ts else "—"
        return {
            "job_id": sub.name,
            "artifacts": len(sub),
            "workloads": ", ".join(sorted(sub["workload"].dropna().unique())) if "workload" in sub else "",
            "gpu_runs": len(gpu),
            "validation_failures": n_fail,
            "status": "OK" if n_fail == 0 else f"{n_fail} FAILED",
            "last_run": when,
        }

    table = (g.groupby("job_id", group_keys=False).apply(_agg)).tolist()
    table = sorted(table, key=lambda r: str(r["job_id"]))
    cols = ["job_id", "artifacts", "workloads", "gpu_runs",
            "validation_failures", "status", "last_run"]
    return dbc.Container([
        html.H3("Job Runs", className="mb-3"),
        html.P("Each row is one job (a Slurm --array task writes its own artifacts). "
               "Failures here are validation failures, not scheduler failures — check "
               "sacct for the latter.", className="text-muted"),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in cols],
            data=table, page_size=25, sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": 12, "padding": "4px 8px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": "{validation_failures} > 0", "column_id": "status"},
                 "backgroundColor": "#f8d7da", "color": "#842029"},
            ],
        ),
    ], fluid=True)
