import pytest

from app.metrics import Metrics


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_percentiles_and_throughput():
    clock = Clock()
    metrics = Metrics(window_s=300, clock=clock)
    clock.now += 10
    for latency in range(1, 101):  # 1..100 ms
        metrics.observe_request(float(latency), 200)
    summary = metrics.summary(window_s=10)
    assert summary["requests"] == 100
    assert summary["latency_ms"]["p50"] == pytest.approx(50.5)
    assert summary["latency_ms"]["p95"] == pytest.approx(95.05)
    assert summary["latency_ms"]["p99"] == pytest.approx(99.01)
    assert summary["throughput_rps"] == pytest.approx(10.0)


def test_error_rate_counts_only_5xx():
    metrics = Metrics(clock=Clock())
    for status in (200, 200, 422, 500, 503, 200, 200, 200, 200, 200):
        metrics.observe_request(5.0, status)
    summary = metrics.summary()
    assert summary["error_rate"] == pytest.approx(0.2)
    assert summary["rejected_rate"] == pytest.approx(0.1)
    assert summary["totals"]["errors"] == 2


def test_cache_hit_rate_is_per_text():
    metrics = Metrics(clock=Clock())
    metrics.observe_request(1.0, 200, items=4, cache_hits=3)
    metrics.observe_request(1.0, 200, items=1, cache_hits=0)
    assert metrics.summary()["cache_hit_rate"] == pytest.approx(0.6)


def test_average_batch_size():
    metrics = Metrics(clock=Clock())
    metrics.observe_batch(4, 10.0)
    metrics.observe_batch(8, 12.0)
    summary = metrics.summary()
    assert summary["avg_batch_size"] == pytest.approx(6.0)
    assert summary["inference_ms"]["mean"] == pytest.approx(11.0)


def test_old_observations_leave_the_window():
    clock = Clock()
    metrics = Metrics(window_s=60, clock=clock)
    metrics.observe_request(500.0, 500)
    clock.now += 120
    metrics.observe_request(5.0, 200)
    summary = metrics.summary(window_s=60)
    assert summary["requests"] == 1
    assert summary["error_rate"] == 0.0
    assert summary["totals"]["requests"] == 2  # lifetime totals keep everything


def test_empty_window_reports_none_not_zero():
    summary = Metrics(clock=Clock()).summary()
    assert summary["latency_ms"]["p95"] is None
    assert summary["error_rate"] is None
    assert summary["throughput_rps"] == 0.0


def test_series_has_fixed_length_and_places_data():
    clock = Clock(now=1_000.0)
    metrics = Metrics(window_s=60, clock=clock)
    metrics.observe_request(10.0, 200)
    series = metrics.summary(step_s=5)["series"]
    assert len(series) == 12
    assert [p["t"] for p in series] == sorted(p["t"] for p in series)
    filled = [p for p in series if p["p95"] is not None]
    assert len(filled) == 1
    assert filled[0]["t"] <= 1_000 < filled[0]["t"] + 5
    assert filled[0]["rps"] == pytest.approx(0.2)


def test_prometheus_exposition():
    metrics = Metrics(clock=Clock())
    metrics.observe_request(3.0, 200)  # 0.003 s -> le=0.005 bucket
    metrics.observe_request(9_000.0, 500)  # 9 s -> only +Inf
    text = metrics.prometheus({"inference_queue_depth": 2})
    assert "inference_requests_total 2" in text
    assert "inference_request_errors_total 1" in text
    assert 'inference_request_duration_seconds_bucket{le="0.0025"} 0' in text
    assert 'inference_request_duration_seconds_bucket{le="0.005"} 1' in text
    assert 'inference_request_duration_seconds_bucket{le="5.0"} 1' in text
    assert 'inference_request_duration_seconds_bucket{le="+Inf"} 2' in text
    assert "inference_request_duration_seconds_count 2" in text
    assert "inference_queue_depth 2" in text
