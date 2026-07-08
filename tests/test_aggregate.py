"""The aggregator: raw artifacts -> tidy CPU/GPU summary -> markdown."""
from bench.aggregate import build_frame, summary_table, to_markdown
from cudalpha.metrics import BenchmarkResult


def _seed_results(d):
    BenchmarkResult(workload="backtester", device="cpu", backend="numpy",
                    size={"series_len": 100000}, median_ms=10.0, p95_ms=11.0, std_ms=0.5).save(d)
    BenchmarkResult(workload="backtester", device="gpu", backend="cupy-rawkernel-fast",
                    size={"series_len": 100000}, median_ms=1.0, p95_ms=1.2, std_ms=0.1,
                    gpu_util_pct=88.0, speedup_vs_cpu=10.0, passed_validation=True).save(d)


def test_build_frame_flattens_size_label(tmp_path):
    _seed_results(tmp_path)
    df = build_frame(tmp_path)
    assert len(df) == 2
    assert "series_len=100000" in set(df["size_label"])


def test_summary_pairs_cpu_and_gpu(tmp_path):
    _seed_results(tmp_path)
    summary = summary_table(build_frame(tmp_path))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["cpu_ms"] == 10.0 and row["gpu_ms"] == 1.0
    assert row["speedup_vs_cpu"] == 10.0


def test_markdown_renders_without_error(tmp_path):
    _seed_results(tmp_path)
    md = to_markdown(summary_table(build_frame(tmp_path)))
    assert "backtester" in md and "Speedup" in md


def test_empty_dir_is_graceful(tmp_path):
    assert build_frame(tmp_path).empty
    assert "No results" in to_markdown(summary_table(build_frame(tmp_path)))
