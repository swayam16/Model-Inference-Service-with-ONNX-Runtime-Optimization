"""Shared fixtures. Most tests use a fake engine so they need no model files."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engines import BatchOutput, Prediction
from app.main import create_app


class FakeEngine:
    """Deterministic stand-in for a model: POSITIVE if the text contains 'good'."""

    name = "fake"
    model_id = "fake-model"
    labels = ["NEGATIVE", "POSITIVE"]

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()

    def predict(self, texts: list[str]) -> BatchOutput:
        with self._lock:
            self.batch_sizes.append(len(texts))
        if any("boom" in t for t in texts):
            raise RuntimeError("model exploded")
        if self.delay_s:
            time.sleep(self.delay_s)
        predictions = []
        for text in texts:
            positive = 0.9 if "good" in text else 0.1
            label = "POSITIVE" if positive > 0.5 else "NEGATIVE"
            predictions.append(
                Prediction(
                    label=label,
                    score=max(positive, 1 - positive),
                    scores={"NEGATIVE": 1 - positive, "POSITIVE": positive},
                )
            )
        return BatchOutput(predictions, tokenize_ms=0.1, inference_ms=self.delay_s * 1e3)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine, tmp_path):
    settings = Settings(max_batch_wait_ms=1.0, benchmark_file=tmp_path / "benchmark.json")
    with TestClient(create_app(settings, engine=engine)) as test_client:
        yield test_client
