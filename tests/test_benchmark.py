"""The statistical timing harness: warmup discard, distribution, synchronize."""
from cudalpha.benchmark import time_callable


def test_reports_median_p95_std():
    stats = time_callable(lambda: sum(range(1000)), warmup=2, trials=10)
    assert set(stats) == {"median_ms", "p95_ms", "std_ms"}
    assert stats["p95_ms"] >= stats["median_ms"]  # p95 >= median by construction
    assert stats["std_ms"] >= 0.0


def test_warmup_iterations_are_not_timed():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    time_callable(fn, warmup=5, trials=7)
    assert calls["n"] == 12, "should call warmup + trials times total"


def test_synchronize_called_each_trial():
    syncs = {"n": 0}

    def sync():
        syncs["n"] += 1

    time_callable(lambda: None, warmup=3, trials=4, synchronize=sync)
    # once after warmup + once per trial
    assert syncs["n"] == 1 + 4


def test_single_trial_has_zero_std():
    stats = time_callable(lambda: None, warmup=0, trials=1)
    assert stats["std_ms"] == 0.0
