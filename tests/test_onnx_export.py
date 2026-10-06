"""End-to-end: build a tiny random model, export it, and serve it with ONNX Runtime."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.engines import OnnxEngine, PyTorchEngine  # noqa: E402
from app.main import create_app  # noqa: E402
from scripts.export_onnx import export  # noqa: E402
from scripts.make_test_model import build  # noqa: E402
from scripts.samples import SAMPLE_TEXTS  # noqa: E402


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    source = tmp_path_factory.mktemp("source")
    out = tmp_path_factory.mktemp("model")
    build("tiny", source)
    export(str(source), out, opset=17, max_len=64, quantize=True)
    return out


def test_export_writes_all_artifacts(model_dir):
    for name in (
        "model.onnx",
        "model.quant.onnx",
        "tokenizer.json",
        "config.json",
        "export_report.json",
    ):
        assert (model_dir / name).exists(), name


def test_fp32_onnx_matches_pytorch(model_dir):
    reference, _, _ = PyTorchEngine(model_dir, max_len=64).logits(SAMPLE_TEXTS)
    logits, _, _ = OnnxEngine(model_dir, "model.onnx", max_len=64).logits(SAMPLE_TEXTS)
    np.testing.assert_allclose(logits, reference, atol=1e-4)


def test_dynamic_axes_accept_any_batch_and_length(model_dir):
    engine = OnnxEngine(model_dir, "model.onnx", max_len=64)
    assert len(engine.predict(["short"]).predictions) == 1
    assert len(engine.predict(SAMPLE_TEXTS).predictions) == len(SAMPLE_TEXTS)
    assert len(engine.predict(["word " * 500]).predictions) == 1  # truncated to max_len


def test_batching_does_not_change_results(model_dir):
    engine = OnnxEngine(model_dir, "model.onnx", max_len=64)
    batched, _, _ = engine.logits(SAMPLE_TEXTS[:8])
    single = np.concatenate([engine.logits([t])[0] for t in SAMPLE_TEXTS[:8]])
    np.testing.assert_allclose(batched, single, atol=1e-4)


def test_quantized_engine_is_the_default_and_scores_sum_to_one(model_dir):
    engine = OnnxEngine(model_dir, max_len=64)
    assert engine.name == "onnx-int8"
    prediction = engine.predict(["Great film."]).predictions[0]
    assert prediction.label in engine.labels
    assert sum(prediction.scores.values()) == pytest.approx(1.0)


def test_service_runs_on_the_exported_model(model_dir, tmp_path):
    settings = Settings(
        model_dir=model_dir, max_seq_len=64, benchmark_file=tmp_path / "benchmark.json"
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json={"text": "Great film."})
        assert response.status_code == 200
        assert response.json()["engine"] == "onnx-int8"
        assert client.get("/info").json()["labels"] == ["NEGATIVE", "POSITIVE"]
