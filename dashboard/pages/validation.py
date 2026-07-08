"""Validation — CPU-vs-GPU correctness, with the error margins that back it up."""
import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:
    from data import empty_state, load_raw

dash.register_page(__name__, path="/validation", name="Validation", order=6)


def _detail(row, key):
    """Pull a nested validation_detail field flattened by json_normalize, else None."""
    col = f"validation_detail.{key}"
    return row[col] if col in row and row[col] == row[col] else None  # NaN check


def layout(**kwargs):
    df = load_raw()
    if df.empty:
        return dbc.Container([html.H3("Validation", className="mb-3"),
                              empty_state("No benchmark artifacts found.")], fluid=True)

    gpu = df[df.get("device") == "gpu"].copy() if "device" in df else df.iloc[0:0]
    if gpu.empty:
        return dbc.Container([html.H3("Validation", className="mb-3"),
                              empty_state("No GPU runs to validate yet.")], fluid=True)

    rows = []
    for _, r in gpu.iterrows():
        passed = r.get("passed_validation")
        rows.append({
            "workload": r.get("workload"),
            "backend": r.get("backend"),
            "size": r.get("size_label"),
            "passed": "✔" if passed else "✘" if passed is not None else "—",
            "fp16": bool(_detail(r, "fp16")) if _detail(r, "fp16") is not None else "",
            "max_abs_err": _detail(r, "max_abs_err"),
            "max_rel_err": _detail(r, "max_rel_err"),
            "obj_rel_err": _detail(r, "obj_rel_err"),
            "note": _detail(r, "note") or "",
        })

    total = len(rows)
    n_pass = sum(1 for x in rows if x["passed"] == "✔")
    banner_color = "success" if n_pass == total else "warning"
    banner = dbc.Alert(
        f"{n_pass}/{total} GPU runs validated against the CPU reference within tolerance. "
        "fp16 rows are compared at documented looser tolerances — expected deviation, not a failure.",
        color=banner_color, className="mb-3")

    cols = ["workload", "backend", "size", "passed", "fp16",
            "max_abs_err", "max_rel_err", "obj_rel_err", "note"]
    return dbc.Container([
        html.H3("Validation", className="mb-3"),
        banner,
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in cols],
            data=rows, page_size=25, sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "monospace", "fontSize": 12, "padding": "4px 8px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": '{passed} = "✘"', "column_id": "passed"},
                 "backgroundColor": "#f8d7da", "color": "#842029"},
                {"if": {"filter_query": '{passed} = "✔"', "column_id": "passed"},
                 "backgroundColor": "#d1e7dd", "color": "#0f5132"},
            ],
        ),
    ], fluid=True)
