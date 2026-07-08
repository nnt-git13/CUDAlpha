"""CPU vs GPU — runtime, speedup, throughput, and peak-memory charts by size."""
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input, Output, callback, dcc, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:
    from data import empty_state, load_raw

dash.register_page(__name__, path="/cpu-vs-gpu", name="CPU vs GPU", order=3)


def layout(**kwargs):
    df = load_raw()
    if df.empty:
        return dbc.Container([html.H3("CPU vs GPU", className="mb-3"),
                              empty_state("No benchmark artifacts found.")], fluid=True)
    # Only offer workloads that have a CPU baseline to compare against — the
    # GPU-only kernel study lives on the Profiling page, not here.
    workloads = sorted(df["workload"].dropna().unique()) if "workload" in df else []
    if "device" in df:
        cpu_workloads = set(df[df["device"] == "cpu"]["workload"].dropna())
        workloads = [w for w in workloads if w in cpu_workloads]
    return dbc.Container([
        html.H3("CPU vs GPU", className="mb-3"),
        dcc.Dropdown(id="cg-workload", options=[{"label": w, "value": w} for w in workloads],
                     value=workloads[0] if workloads else None, clearable=False,
                     className="mb-3", style={"maxWidth": 320}),
        dbc.Row([dbc.Col(dcc.Graph(id="cg-runtime"), lg=6),
                 dbc.Col(dcc.Graph(id="cg-speedup"), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id="cg-throughput"), lg=6),
                 dbc.Col(dcc.Graph(id="cg-memory"), lg=6)]),
    ], fluid=True)


@callback(
    Output("cg-runtime", "figure"), Output("cg-speedup", "figure"),
    Output("cg-throughput", "figure"), Output("cg-memory", "figure"),
    Input("cg-workload", "value"),
)
def _charts(workload):
    df = load_raw()
    d = df[df["workload"] == workload].copy() if workload and "workload" in df else df.iloc[0:0]
    d = d.sort_values("size_value") if "size_value" in d else d

    def _empty(title):
        f = px.scatter(title=title)
        f.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        return f

    if d.empty:
        return _empty("runtime"), _empty("speedup"), _empty("throughput"), _empty("peak memory")

    runtime = px.bar(d, x="size_label", y="median_ms", color="backend", barmode="group",
                     error_y="std_ms" if "std_ms" in d else None,
                     title=f"{workload}: median runtime by size (± std)")
    runtime.update_layout(yaxis_title="median (ms)", margin=dict(t=50, l=10, r=10, b=10))

    gpu = d[d.get("device") == "gpu"].dropna(subset=["speedup_vs_cpu"]) if "device" in d else d.iloc[0:0]
    if len(gpu):
        speedup = px.line(gpu, x="size_value", y="speedup_vs_cpu", color="backend",
                          markers=True, title=f"{workload}: GPU speedup vs size")
        speedup.add_hline(y=1.0, line_dash="dot", annotation_text="parity")
        speedup.update_layout(xaxis_title="size", yaxis_title="speedup (×)",
                              margin=dict(t=50, l=10, r=10, b=10))
    else:
        speedup = _empty(f"{workload}: speedup (no GPU runs)")

    if "throughput" in d and d["throughput"].notna().any():
        throughput = px.line(d.dropna(subset=["throughput"]), x="size_value", y="throughput",
                             color="backend", markers=True,
                             title=f"{workload}: throughput (items/s)")
        throughput.update_layout(xaxis_title="size", margin=dict(t=50, l=10, r=10, b=10))
    else:
        throughput = _empty(f"{workload}: throughput (n/a)")

    # peak_mem_mb is torch-only; cupy workloads report 0, so treat 0 as no-data
    # rather than drawing empty bars.
    mem = gpu[gpu["peak_mem_mb"] > 0] if "peak_mem_mb" in gpu else gpu.iloc[0:0]
    if len(mem):
        memory = px.bar(mem, x="size_label", y="peak_mem_mb", color="backend", barmode="group",
                        title=f"{workload}: peak GPU memory (MB, torch)")
        memory.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    else:
        memory = _empty(f"{workload}: peak GPU memory (torch-only; n/a for CuPy)")

    return runtime, speedup, throughput, memory
