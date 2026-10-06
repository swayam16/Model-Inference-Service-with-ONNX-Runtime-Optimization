import json


def test_predict_returns_label_and_trace_headers(client):
    response = client.post("/predict", json={"text": "a good film"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "POSITIVE"
    assert body["cached"] is False
    assert body["model"] == "fake-model"
    assert response.headers["X-Request-ID"] == body["request_id"]
    timing = response.headers["Server-Timing"]
    for stage in ("cache", "queue", "tokenize", "inference", "total"):
        assert f"{stage};dur=" in timing


def test_repeat_request_is_served_from_cache(client, engine):
    first = client.post("/predict", json={"text": "a good film"}).json()
    calls = len(engine.batch_sizes)
    second = client.post("/predict", json={"text": "a good film"}).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["label"] == first["label"]
    assert len(engine.batch_sizes) == calls  # the model was not called again


def test_caller_request_id_is_propagated(client):
    response = client.post(
        "/predict", json={"text": "good"}, headers={"X-Request-ID": "upstream-123"}
    )
    assert response.headers["X-Request-ID"] == "upstream-123"
    assert response.json()["request_id"] == "upstream-123"


def test_malformed_request_id_is_replaced(client):
    response = client.post("/predict", json={"text": "good"}, headers={"X-Request-ID": "<script>"})
    assert response.headers["X-Request-ID"] != "<script>"


def test_batch_endpoint_preserves_order(client):
    texts = ["good", "bad", "very good", "bad"]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["label"] for r in results] == ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]


def test_validation_errors(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 5001}).status_code == 422
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["a"] * 65}).status_code == 422


def test_model_failure_is_a_traced_500_and_counts_as_an_error(client):
    response = client.post("/predict", json={"text": "boom"})
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["X-Request-ID"]

    client.post("/predict", json={"text": "good"})
    summary = client.get("/metrics/summary").json()
    assert summary["requests"] == 2
    assert summary["error_rate"] == 0.5
    assert summary["totals"]["errors"] == 1


def test_metrics_summary_tracks_requests_cache_and_batches(client):
    for text in ("good", "good", "bad"):
        client.post("/predict", json={"text": text})
    summary = client.get("/metrics/summary?window=60&step=5").json()
    assert summary["requests"] == 3
    assert summary["latency_ms"]["p95"] > 0
    assert summary["throughput_rps"] > 0
    assert summary["cache_hit_rate"] == 1 / 3
    assert summary["avg_batch_size"] == 1.0
    assert summary["error_rate"] == 0.0
    assert len(summary["series"]) == 60


def test_only_prediction_requests_are_measured(client):
    client.get("/healthz")
    client.get("/metrics/summary")
    assert client.get("/metrics/summary").json()["requests"] == 0


def test_traces_expose_stage_timings(client):
    client.post("/predict", json={"text": "good"})
    client.post("/predict", json={"text": "good"})
    traces = client.get("/traces?limit=10").json()
    assert len(traces) == 2
    newest, oldest = traces
    assert newest["cache_hits"] == 1 and "inference" not in newest["spans_ms"]
    assert oldest["cache_hits"] == 0 and oldest["spans_ms"]["inference"] >= 0
    assert oldest["batch_size"] == 1
    assert oldest["status"] == 200


def test_prometheus_endpoint(client):
    client.post("/predict", json={"text": "good"})
    text = client.get("/metrics").text
    assert "inference_requests_total 1" in text
    assert "inference_queue_depth 0" in text


def test_health_info_and_dashboard(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    info = client.get("/info").json()
    assert info["engine"] == "fake"
    assert info["batching"]["max_batch_size"] == 16
    page = client.get("/")
    assert page.status_code == 200
    assert "Inference Service" in page.text


def test_benchmark_endpoint(client):
    assert client.get("/benchmark").status_code == 404
    client.app.state.settings.benchmark_file.write_text(json.dumps({"model_id": "m"}))
    assert client.get("/benchmark").json() == {"model_id": "m"}
