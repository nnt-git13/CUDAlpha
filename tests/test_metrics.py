"""Results schema: env capture, JSON round-trip, and filename provenance."""
from cudalpha.metrics import BenchmarkResult, capture_environment, load_results


def _make(**kw):
    base = dict(workload="backtester", device="gpu", backend="cupy-rawkernel-fast",
                size={"series_len": 100000}, median_ms=1.2, p95_ms=1.5, std_ms=0.1)
    base.update(kw)
    return BenchmarkResult(**base)


def test_capture_environment_has_core_keys():
    env = capture_environment()
    for k in ("python", "platform", "git_sha", "nvidia_driver"):
        assert k in env


def test_save_and_load_round_trip(tmp_path):
    r = _make(speedup_vs_cpu=8.0, passed_validation=True)
    path = r.save(tmp_path)
    assert path.exists()
    loaded = load_results(tmp_path)
    assert len(loaded) == 1
    row = loaded[0]
    assert row["workload"] == "backtester"
    assert row["backend"] == "cupy-rawkernel-fast"
    assert row["size"] == {"series_len": 100000}
    assert row["speedup_vs_cpu"] == 8.0
    assert row["passed_validation"] is True
    assert "env" in row and "run_id" in row


def test_filename_encodes_workload_backend_device_size(tmp_path):
    path = _make().save(tmp_path)
    name = path.name
    assert name.startswith("backtester_cupy-rawkernel-fast_gpu_")
    assert "series_len100000" in name


def test_load_results_ignores_non_result_json(tmp_path):
    (tmp_path / "summary.parquet").write_bytes(b"not json")
    (tmp_path / "bad.json").write_text("{ not valid json")
    _make().save(tmp_path)
    assert len(load_results(tmp_path)) == 1
