"""Closed-loop load generator for a running service.

    python -m scripts.load_test --url http://localhost:8000 --requests 2000 --concurrency 32

``--repeat-ratio`` controls how many requests reuse a text that was already sent
(these should be served from the cache); the rest are made unique so they reach
the model and exercise batching.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx
import numpy as np

from scripts.samples import SAMPLE_TEXTS


async def run(url: str, requests: int, concurrency: int, repeat_ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    cached = 0
    counter = iter(range(requests))

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal cached
        for i in counter:
            text = rng.choice(SAMPLE_TEXTS)
            if rng.random() >= repeat_ratio:
                text = f"{text} (request {i})"
            t0 = time.perf_counter()
            try:
                response = await client.post("/predict", json={"text": text})
                status = response.status_code
                if status == 200 and response.json().get("cached"):
                    cached += 1
            except httpx.HTTPError:
                status = 0
            latencies.append((time.perf_counter() - t0) * 1e3)
            statuses[status] = statuses.get(status, 0) + 1

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.perf_counter() - started

    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    ok = statuses.get(200, 0)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed, 2),
        "throughput_rps": round(requests / elapsed, 1),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "error_rate": round(1 - ok / requests, 4),
        "cache_hit_rate": round(cached / ok, 4) if ok else 0.0,
        "statuses": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--repeat-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(
        run(args.url, args.requests, args.concurrency, args.repeat_ratio, args.seed)
    )
    width = max(len(k) for k in result)
    for key, value in result.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
