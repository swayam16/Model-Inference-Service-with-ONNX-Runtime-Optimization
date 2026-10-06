"""FastAPI application: prediction API, metrics, traces and the dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .batching import DynamicBatcher, QueueFullError
from .cache import TTLCache
from .config import Settings
from .engines import Engine, Prediction, build_engine
from .metrics import Metrics
from .tracing import Trace, TraceBuffer, new_request_id

STATIC_DIR = Path(__file__).parent / "static"
PREDICT_PATHS = {"/predict", "/predict/batch"}


class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify.", examples=["I loved this movie."])


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Texts to classify.")


class PredictionOut(BaseModel):
    label: str
    score: float
    scores: dict[str, float]
    cached: bool


class PredictResponse(PredictionOut):
    request_id: str
    model: str
    engine: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    request_id: str
    model: str
    engine: str
    latency_ms: float
    results: list[PredictionOut]


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    """Build the app. Pass ``engine`` to inject a model (used by the tests)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        state = app.state
        state.settings = settings
        state.metrics = Metrics(window_s=settings.metrics_window_s)
        state.traces = TraceBuffer(settings.trace_buffer_size)
        state.cache = TTLCache(settings.cache_max_items, settings.cache_ttl_s)
        state.engine = engine or await asyncio.to_thread(
            build_engine,
            settings.engine,
            settings.model_dir,
            settings.onnx_file,
            settings.max_seq_len,
            settings.intra_op_threads,
        )
        state.batcher = DynamicBatcher(
            state.engine,
            max_batch_size=settings.max_batch_size if settings.batching_enabled else 1,
            max_wait_ms=settings.max_batch_wait_ms if settings.batching_enabled else 0.0,
            max_queue_size=settings.max_queue_size,
            on_batch=state.metrics.observe_batch,
        )
        # Warm up so the first real request does not pay one-off initialisation costs.
        await asyncio.to_thread(state.engine.predict, ["warm up"])
        await state.batcher.start()
        state.ready = True
        yield
        state.ready = False
        await state.batcher.stop()

    app = FastAPI(
        title="ONNX Runtime Inference Service",
        version=__version__,
        description="Transformer text classification served with ONNX Runtime, "
        "dynamic batching, result caching and per-request tracing.",
        lifespan=lifespan,
    )
    app.state.ready = False

    # -- tracing + metrics middleware ---------------------------------------

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        trace = Trace(
            request_id=new_request_id(request.headers.get("x-request-id")),
            method=request.method,
            path=request.url.path,
        )
        request.state.trace = trace
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # unhandled error -> traced 500
            logging.getLogger("inference").exception("unhandled error")
            trace.set(error=type(exc).__name__)
            response = JSONResponse(
                {"detail": "internal server error", "request_id": trace.request_id},
                status_code=500,
            )
        trace.status = response.status_code
        trace.total_ms = (time.perf_counter() - started) * 1e3
        response.headers["X-Request-ID"] = trace.request_id
        response.headers["Server-Timing"] = trace.server_timing()

        if request.method == "POST" and request.url.path in PREDICT_PATHS:
            app.state.metrics.observe_request(
                trace.total_ms,
                trace.status,
                items=trace.attrs.get("items", 1),
                cache_hits=trace.attrs.get("cache_hits", 0),
            )
            app.state.traces.record(trace)
        return response

    # -- prediction ---------------------------------------------------------

    def _validate(text: str) -> str:
        text = text.strip()
        if not text:
            raise HTTPException(422, "text must not be empty")
        if len(text) > settings.max_text_chars:
            raise HTTPException(422, f"text exceeds {settings.max_text_chars} characters")
        return text

    async def _score(text: str, trace: Trace) -> tuple[Prediction, bool]:
        state = app.state
        key = TTLCache.make_key(f"{state.engine.model_id}/{state.engine.name}", text)
        if settings.cache_enabled:
            t0 = time.perf_counter()
            hit = state.cache.get(key)
            trace.add_span("cache", (time.perf_counter() - t0) * 1e3)
            if hit is not None:
                return hit, True
        try:
            result = await asyncio.wait_for(
                state.batcher.submit(text), timeout=settings.request_timeout_s
            )
        except QueueFullError:
            raise HTTPException(
                503, "server is overloaded, retry shortly", headers={"Retry-After": "1"}
            ) from None
        except asyncio.TimeoutError:
            raise HTTPException(504, "inference timed out") from None
        trace.add_span("queue", result.queue_ms)
        trace.add_span("tokenize", result.tokenize_ms)
        trace.add_span("inference", result.inference_ms)
        trace.set(batch_size=max(trace.attrs.get("batch_size", 0), result.batch_size))
        if settings.cache_enabled:
            state.cache.set(key, result.prediction)
        return result.prediction, False

    def _out(prediction: Prediction, cached: bool) -> PredictionOut:
        return PredictionOut(
            label=prediction.label, score=prediction.score, scores=prediction.scores, cached=cached
        )

    def _require_ready() -> None:
        if not app.state.ready:
            raise HTTPException(503, "model is still loading")

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(body: PredictRequest, request: Request):
        """Classify one text."""
        _require_ready()
        trace: Trace = request.state.trace
        started = time.perf_counter()
        trace.set(items=1, cache_hits=0)
        prediction, cached = await _score(_validate(body.text), trace)
        trace.set(cache_hits=int(cached), label=prediction.label)
        return PredictResponse(
            **_out(prediction, cached).model_dump(),
            request_id=trace.request_id,
            model=app.state.engine.model_id,
            engine=app.state.engine.name,
            latency_ms=(time.perf_counter() - started) * 1e3,
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
    async def predict_batch(body: BatchPredictRequest, request: Request):
        """Classify several texts in one call. They join the same dynamic batches."""
        _require_ready()
        trace: Trace = request.state.trace
        started = time.perf_counter()
        if len(body.texts) > settings.max_texts_per_request:
            raise HTTPException(422, f"at most {settings.max_texts_per_request} texts per request")
        texts = [_validate(t) for t in body.texts]
        trace.set(items=len(texts), cache_hits=0)
        scored = await asyncio.gather(*(_score(t, trace) for t in texts))
        trace.set(cache_hits=sum(1 for _, cached in scored if cached))
        return BatchPredictResponse(
            request_id=trace.request_id,
            model=app.state.engine.model_id,
            engine=app.state.engine.name,
            latency_ms=(time.perf_counter() - started) * 1e3,
            results=[_out(p, cached) for p, cached in scored],
        )

    # -- operations ---------------------------------------------------------

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz():
        """Readiness: the model is loaded and warmed up."""
        if not app.state.ready:
            raise HTTPException(503, "model is still loading")
        return {"status": "ready"}

    @app.get("/info", tags=["ops"])
    async def info():
        _require_ready()
        eng = app.state.engine
        return {
            "version": __version__,
            "model": eng.model_id,
            "engine": eng.name,
            "model_file": getattr(eng, "model_file", None),
            "labels": eng.labels,
            "max_seq_len": settings.max_seq_len,
            "batching": {
                "enabled": settings.batching_enabled,
                "max_batch_size": app.state.batcher.max_batch_size,
                "max_wait_ms": app.state.batcher.max_wait_s * 1e3,
                "max_queue_size": settings.max_queue_size,
            },
            "cache": {"enabled": settings.cache_enabled, **app.state.cache.stats()},
        }

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    async def prometheus_metrics():
        """Prometheus text exposition format."""
        _require_ready()
        return app.state.metrics.prometheus(
            {
                "inference_queue_depth": app.state.batcher.queue_depth,
                "inference_cache_entries": len(app.state.cache),
            }
        )

    @app.get("/metrics/summary", tags=["ops"])
    async def metrics_summary(
        window: int = Query(60, ge=1, le=3600, description="Headline window in seconds."),
        step: int = Query(5, ge=1, le=60, description="Time-series resolution in seconds."),
    ):
        """JSON metrics for the dashboard."""
        _require_ready()
        summary = app.state.metrics.summary(window_s=window, step_s=step)
        summary["queue_depth"] = app.state.batcher.queue_depth
        summary["cache"] = app.state.cache.stats()
        return summary

    @app.get("/traces", tags=["ops"])
    async def traces(limit: int = Query(50, ge=1, le=500)):
        """Most recent request traces, newest first."""
        return app.state.traces.recent(limit) if app.state.ready else []

    @app.get("/benchmark", tags=["ops"])
    async def benchmark():
        """Offline PyTorch-vs-ONNX benchmark recorded by scripts/benchmark.py."""
        path = settings.benchmark_file
        if not path.exists():
            raise HTTPException(404, "no benchmark recorded; run scripts/benchmark.py")
        return json.loads(path.read_text())

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    return app


app = create_app()
