"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model
    model_dir: Path = Path("artifacts/model")
    engine: str = "onnx"  # "onnx" or "pytorch" (pytorch needs the dev requirements)
    onnx_file: str = "model.quant.onnx"  # falls back to model.onnx if missing
    max_seq_len: int = 128
    intra_op_threads: int = 0  # 0 = let the runtime decide

    # Dynamic batching
    batching_enabled: bool = True
    max_batch_size: int = 16
    max_batch_wait_ms: float = 5.0
    max_queue_size: int = 512
    request_timeout_s: float = 10.0

    # Result cache
    cache_enabled: bool = True
    cache_max_items: int = 10_000
    cache_ttl_s: float = 600.0

    # Observability
    metrics_window_s: int = 300
    trace_buffer_size: int = 200
    benchmark_file: Path = Path("artifacts/benchmark.json")

    # API limits
    max_text_chars: int = 5_000
    max_texts_per_request: int = 64

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            model_dir=Path(os.environ.get("MODEL_DIR", d.model_dir)),
            engine=os.environ.get("ENGINE", d.engine).lower(),
            onnx_file=os.environ.get("ONNX_FILE", d.onnx_file),
            max_seq_len=_int("MAX_SEQ_LEN", d.max_seq_len),
            intra_op_threads=_int("INTRA_OP_THREADS", d.intra_op_threads),
            batching_enabled=_bool("BATCHING_ENABLED", d.batching_enabled),
            max_batch_size=_int("MAX_BATCH_SIZE", d.max_batch_size),
            max_batch_wait_ms=_float("MAX_BATCH_WAIT_MS", d.max_batch_wait_ms),
            max_queue_size=_int("MAX_QUEUE_SIZE", d.max_queue_size),
            request_timeout_s=_float("REQUEST_TIMEOUT_S", d.request_timeout_s),
            cache_enabled=_bool("CACHE_ENABLED", d.cache_enabled),
            cache_max_items=_int("CACHE_MAX_ITEMS", d.cache_max_items),
            cache_ttl_s=_float("CACHE_TTL_S", d.cache_ttl_s),
            metrics_window_s=_int("METRICS_WINDOW_S", d.metrics_window_s),
            trace_buffer_size=_int("TRACE_BUFFER_SIZE", d.trace_buffer_size),
            benchmark_file=Path(os.environ.get("BENCHMARK_FILE", d.benchmark_file)),
            max_text_chars=_int("MAX_TEXT_CHARS", d.max_text_chars),
            max_texts_per_request=_int("MAX_TEXTS_PER_REQUEST", d.max_texts_per_request),
        )
