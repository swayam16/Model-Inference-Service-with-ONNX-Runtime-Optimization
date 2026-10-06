"""Dynamic request batching.

Concurrent requests are collected into one model call. A batch is dispatched as
soon as it reaches ``max_batch_size`` or ``max_wait_ms`` after its first item
arrived, whichever comes first. While a batch is running, new requests keep
queueing, so under load batches fill up without any added waiting.

The model runs on a single worker thread: one batch at a time, with ONNX
Runtime's intra-op thread pool using the available cores for that batch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .engines import Engine, Prediction


class QueueFullError(Exception):
    """The request queue is at capacity; the caller should shed load (HTTP 503)."""


@dataclass(frozen=True)
class BatchResult:
    prediction: Prediction
    queue_ms: float
    tokenize_ms: float
    inference_ms: float
    batch_size: int


@dataclass
class _Pending:
    text: str
    future: asyncio.Future
    enqueued_at: float


BatchObserver = Callable[[int, float, float], None]  # (batch_size, inference_ms, queue_ms)


class DynamicBatcher:
    def __init__(
        self,
        engine: Engine,
        max_batch_size: int = 16,
        max_wait_ms: float = 5.0,
        max_queue_size: int = 512,
        on_batch: BatchObserver | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_s = max(max_wait_ms, 0.0) / 1e3
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue(maxsize=max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._worker: asyncio.Task | None = None
        self._on_batch = on_batch

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def submit(self, text: str) -> BatchResult:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Pending(text, future, time.perf_counter()))
        except asyncio.QueueFull:
            raise QueueFullError("inference queue is full") from None
        return await future

    async def _collect(self) -> list[_Pending]:
        batch = [await self._queue.get()]
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        # Drop requests whose caller already gave up (timeout / disconnect).
        return [item for item in batch if not item.future.done()]

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            if not batch:
                continue
            started = time.perf_counter()
            try:
                output = await loop.run_in_executor(
                    self._executor, self.engine.predict, [item.text for item in batch]
                )
            except asyncio.CancelledError:
                for item in batch:
                    if not item.future.done():
                        item.future.cancel()
                raise
            except Exception as exc:  # one bad batch must not kill the worker
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            size = len(batch)
            queue_waits = [(started - item.enqueued_at) * 1e3 for item in batch]
            for item, prediction, queue_ms in zip(batch, output.predictions, queue_waits):
                if not item.future.done():
                    item.future.set_result(
                        BatchResult(
                            prediction=prediction,
                            queue_ms=queue_ms,
                            tokenize_ms=output.tokenize_ms,
                            inference_ms=output.inference_ms,
                            batch_size=size,
                        )
                    )
            if self._on_batch is not None:
                self._on_batch(size, output.inference_ms, sum(queue_waits) / size)
