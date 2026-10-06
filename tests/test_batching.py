import asyncio

import pytest

from app.batching import DynamicBatcher, QueueFullError
from tests.conftest import FakeEngine


async def test_concurrent_requests_share_a_batch():
    engine = FakeEngine()
    batcher = DynamicBatcher(engine, max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(f"text {i}") for i in range(8)))
    finally:
        await batcher.stop()
    assert engine.batch_sizes == [8]
    assert all(r.batch_size == 8 for r in results)


async def test_batches_never_exceed_the_maximum():
    engine = FakeEngine(delay_s=0.01)
    batcher = DynamicBatcher(engine, max_batch_size=4, max_wait_ms=20)
    await batcher.start()
    try:
        await asyncio.gather(*(batcher.submit(f"text {i}") for i in range(10)))
    finally:
        await batcher.stop()
    assert sum(engine.batch_sizes) == 10
    assert max(engine.batch_sizes) <= 4


async def test_results_map_back_to_the_right_request():
    batcher = DynamicBatcher(FakeEngine(), max_batch_size=8, max_wait_ms=20)
    await batcher.start()
    try:
        texts = ["good one", "bad one", "really good", "awful"]
        results = await asyncio.gather(*(batcher.submit(t) for t in texts))
    finally:
        await batcher.stop()
    assert [r.prediction.label for r in results] == [
        "POSITIVE",
        "NEGATIVE",
        "POSITIVE",
        "NEGATIVE",
    ]


async def test_lone_request_is_dispatched_after_max_wait():
    engine = FakeEngine()
    batcher = DynamicBatcher(engine, max_batch_size=8, max_wait_ms=10)
    await batcher.start()
    try:
        result = await asyncio.wait_for(batcher.submit("only one"), timeout=2)
    finally:
        await batcher.stop()
    assert result.batch_size == 1
    assert result.queue_ms >= 9


async def test_full_queue_sheds_load():
    batcher = DynamicBatcher(FakeEngine(), max_queue_size=2)  # worker not started
    pending = [asyncio.ensure_future(batcher.submit(f"t{i}")) for i in range(2)]
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        await batcher.submit("one too many")
    await batcher.stop()
    results = await asyncio.gather(*pending, return_exceptions=True)
    assert all(isinstance(r, asyncio.CancelledError) for r in results)


async def test_engine_failure_reaches_callers_and_worker_survives():
    batcher = DynamicBatcher(FakeEngine(), max_batch_size=4, max_wait_ms=5)
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="model exploded"):
            await batcher.submit("boom")
        result = await batcher.submit("good again")
    finally:
        await batcher.stop()
    assert result.prediction.label == "POSITIVE"


async def test_batch_observer_is_called():
    seen = []
    batcher = DynamicBatcher(
        FakeEngine(),
        max_batch_size=4,
        max_wait_ms=20,
        on_batch=lambda size, inference_ms, queue_ms: seen.append(size),
    )
    await batcher.start()
    try:
        await asyncio.gather(*(batcher.submit(f"t{i}") for i in range(4)))
    finally:
        await batcher.stop()
    assert seen == [4]
