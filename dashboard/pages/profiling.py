"""Profiling — the two before/after optimization stories, driven by results."""
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html

try:
    from dashboard.data import empty_state, load_raw
except ImportError:
    from data import empty_state, load_raw

dash.register_page(__name__, path="/profiling", name="Profiling", order=4)

BOTTLENECKS = [
    ("#1 Forecaster inference GPU-idle",
     "host-bound: small batches + synchronous pageable H2D copies",
     "pinned memory + non_blocking copy + AMP + torch.compile + CUDA graphs"),
    ("#2 Rolling-mean kernel scales with window",
     "naive kernel re-sums the whole window per output (O(window) global reads)",
     "prefix-sum rolling_mean_fast (O(1) per output) — wins past the crossover"),
]


def _crossover_chart(df):
    """H2: naive vs prefix-sum kernel median vs MA window, from the kernel sweep."""
    if "workload" not in df or "size.window" not in df:
        return None
    d = df[df["workload"] == "backtester-kernel"].copy()
    if d.empty:
        return None
    d["window"] = df["size.window"]
    d = d.dropna(subset=["window", "median_ms"]).sort_values("window")
    d["kernel"] = d["backend"].str.replace("cupy-rawkernel-", "", regex=False)
    fig = px.line(d, x="window", y="median_ms", color="kernel", markers=True,
                  log_x=True, title="H2: kernel runtime vs MA window (crossover)")
    fig.update_layout(xaxis_title="moving-average window (log)",
                      yaxis_title="median (ms) — lower is better",
                      margin=dict(t=50, l=10, r=10, b=10))
    return dcc.Graph(figure=fig)


def layout(**kwargs):
    df = load_raw()
    rows = [html.Tr([html.Td(name), html.Td(cause), html.Td(fix)])
            for name, cause, fix in BOTTLENECKS]
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Bottleneck"), html.Th("Cause"), html.Th("Fix")])),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, className="mb-4")

    children = [html.H3("Profiling", className="mb-3"),
                html.P("The two bottlenecks and their fixes. Bottleneck #1's lever-by-lever "
                       "deltas come from `make profile-forecaster` (torch.profiler); "
                       "bottleneck #2's kernel crossover is charted below from "
                       "`make bench-kernel-sweep`.", className="text-muted"),
                table]

    if df.empty:
        children.append(empty_state("No benchmark artifacts yet — the crossover chart fills in after a run."))
        return dbc.Container(children, fluid=True)

    # The H2 kernel crossover is the honest before/after: at the default small
    # window the naive kernel is actually faster, so a single "before/after" bar
    # would be misleading — the whole point is *where* the prefix-sum overtakes it.
    crossover = _crossover_chart(df)
    if crossover is not None:
        children.append(html.P("H2 (make bench-kernel-sweep): the naive re-sum kernel scales "
                               "O(window); the prefix-sum kernel stays flat O(1)-per-output. "
                               "They cross at window ≥ 200 — an O(1) algorithm only pays off "
                               "once its fixed scan is amortized.", className="text-muted"))
        children.append(crossover)
    return dbc.Container(children, fluid=True)
