"""Overview page — headline KPIs plus a best-speedup-per-workload chart."""
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:  # launched as `python dashboard/app.py`
    from data import empty_state, load_raw

dash.register_page(__name__, path="/", name="Overview", order=1)


def _kpi(title: str, value: str, sub: str = "", color: str = "primary"):
    return dbc.Card(dbc.CardBody([
        html.Div(title, className="text-muted small text-uppercase"),
        html.H3(value, className=f"text-{color} mb-0"),
        html.Div(sub, className="text-muted small"),
    ]), className="shadow-sm h-100")


def layout(**kwargs):
    df = load_raw()
    if df.empty:
        return dbc.Container([html.H3("Overview", className="mb-3"),
                              empty_state("No benchmark artifacts found.")], fluid=True)

    gpu = df[df.get("device") == "gpu"] if "device" in df else df.iloc[0:0]
    n_runs = len(df)
    n_workloads = df["workload"].nunique() if "workload" in df else 0
    best_speedup = gpu["speedup_vs_cpu"].max() if "speedup_vs_cpu" in gpu and len(gpu) else None
    mean_util = gpu["gpu_util_pct"].mean() if "gpu_util_pct" in gpu and len(gpu) else None
    if "passed_validation" in gpu and len(gpu):
        pass_rate = 100.0 * gpu["passed_validation"].fillna(False).mean()
    else:
        pass_rate = None

    cards = dbc.Row([
        dbc.Col(_kpi("Artifacts", str(n_runs), f"{n_workloads} workloads"), md=3),
        dbc.Col(_kpi("Best speedup", f"{best_speedup:.1f}×" if best_speedup else "—",
                     "GPU vs CPU median", "success"), md=3),
        dbc.Col(_kpi("Mean GPU util", f"{mean_util:.0f}%" if mean_util is not None else "—",
                     "sampled during runs", "info"), md=3),
        dbc.Col(_kpi("Validation", f"{pass_rate:.0f}%" if pass_rate is not None else "—",
                     "CPU-vs-GPU pass rate",
                     "success" if (pass_rate or 0) >= 99 else "warning"), md=3),
    ], className="g-3 mb-4")

    children = [html.H3("Overview", className="mb-3"), cards]

    if len(gpu) and "speedup_vs_cpu" in gpu:
        best = (gpu.dropna(subset=["speedup_vs_cpu"])
                   .sort_values("speedup_vs_cpu", ascending=False)
                   .groupby("workload", as_index=False).first())
        if len(best):
            fig = px.bar(best, x="workload", y="speedup_vs_cpu", color="backend",
                         text="speedup_vs_cpu", title="Best GPU speedup per workload")
            fig.update_traces(texttemplate="%{text:.1f}×", textposition="outside")
            fig.update_layout(yaxis_title="speedup (×)", margin=dict(t=50, l=10, r=10, b=10))
            children.append(dcc.Graph(figure=fig))

    return dbc.Container(children, fluid=True)
