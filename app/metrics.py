"""Service metrics: latency percentiles, throughput, error rate, cache and batching.

Observations are stored in one-second buckets over a rolling window, which
feeds the dashboard (``summary``). Lifetime counters and a latency histogram are
kept alongside for Prometheus scraping (``prometheus``).

All methods are called from the event loop thread, so no locking is needed.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Histogram bucket upper bounds, in seconds.
LATENCY_BUCKETS_S = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass
class _Bucket:
    ts: int
    requests: int = 0
    errors: int = 0  # 5xx
    rejected: int = 0  # 4xx
    cache_hits: int = 0
    items: int = 0
    batches: int = 0
    batch_items: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    arr = np.asarray(values)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(arr.mean())}


def _ratio(num: float, den: float) -> float | None:
    return (num / den) if den else None


class Metrics:
    def __init__(self, window_s: int = 300, clock: Callable[[], float] = time.time) -> None:
        self.window_s = window_s
        self._clock = clock
        self._started = clock()
        self._buckets: deque[_Bucket] = deque()

        # Lifetime totals (Prometheus counters).
        self.requests_total = 0
        self.errors_total = 0
        self.rejected_total = 0
        self.items_total = 0
        self.cache_hits_total = 0
        self.batches_total = 0
        self.batch_items_total = 0
        self.latency_sum_s = 0.0
        self._hist = [0] * (len(LATENCY_BUCKETS_S) + 1)

    # -- recording ---------------------------------------------------------

    def _bucket(self) -> _Bucket:
        now = int(self._clock())
        if not self._buckets or self._buckets[-1].ts != now:
            self._buckets.append(_Bucket(ts=now))
            cutoff = now - self.window_s
            while self._buckets and self._buckets[0].ts < cutoff:
                self._buckets.popleft()
        return self._buckets[-1]

    def observe_request(
        self, latency_ms: float, status: int, items: int = 1, cache_hits: int = 0
    ) -> None:
        b = self._bucket()
        b.requests += 1
        b.items += items
        b.cache_hits += cache_hits
        b.latencies_ms.append(latency_ms)
        self.requests_total += 1
        self.items_total += items
        self.cache_hits_total += cache_hits
        if status >= 500:
            b.errors += 1
            self.errors_total += 1
        elif status >= 400:
            b.rejected += 1
            self.rejected_total += 1

        seconds = latency_ms / 1e3
        self.latency_sum_s += seconds
        for i, bound in enumerate(LATENCY_BUCKETS_S):
            if seconds <= bound:
                self._hist[i] += 1
                break
        else:
            self._hist[-1] += 1

    def observe_batch(self, batch_size: int, inference_ms: float, queue_ms: float = 0.0) -> None:
        b = self._bucket()
        b.batches += 1
        b.batch_items += batch_size
        b.inference_ms.append(inference_ms)
        self.batches_total += 1
        self.batch_items_total += batch_size

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _aggregate(buckets: list[_Bucket], seconds: float) -> dict:
        requests = sum(b.requests for b in buckets)
        items = sum(b.items for b in buckets)
        latencies = [v for b in buckets for v in b.latencies_ms]
        inference = [v for b in buckets for v in b.inference_ms]
        return {
            "requests": requests,
            "throughput_rps": requests / seconds if seconds > 0 else 0.0,
            "latency_ms": _percentiles(latencies),
            "inference_ms": _percentiles(inference),
            "error_rate": _ratio(sum(b.errors for b in buckets), requests),
            "rejected_rate": _ratio(sum(b.rejected for b in buckets), requests),
            "cache_hit_rate": _ratio(sum(b.cache_hits for b in buckets), items),
            "avg_batch_size": _ratio(
                sum(b.batch_items for b in buckets), sum(b.batches for b in buckets)
            ),
        }

    def summary(self, window_s: int = 60, step_s: int = 5) -> dict:
        """Headline stats for the last ``window_s`` plus a time series over the full window."""
        now = int(self._clock())
        window_s = max(1, min(window_s, self.window_s))
        step_s = max(1, step_s)
        uptime = max(self._clock() - self._started, 1e-9)

        recent = [b for b in self._buckets if b.ts > now - window_s]
        headline = self._aggregate(recent, min(window_s, uptime))

        # Align steps to wall-clock multiples so points stay stable between polls.
        end = now - (now % step_s) + step_s
        start = end - (self.window_s // step_s) * step_s
        slots: dict[int, list[_Bucket]] = {}
        for b in self._buckets:
            if b.ts >= start:
                slots.setdefault((b.ts - start) // step_s, []).append(b)
        series = []
        for i in range((end - start) // step_s):
            agg = self._aggregate(slots.get(i, []), step_s)
            series.append(
                {
                    "t": start + i * step_s,
                    "rps": agg["throughput_rps"],
                    "p50": agg["latency_ms"]["p50"],
                    "p95": agg["latency_ms"]["p95"],
                    "error_rate": agg["error_rate"],
                    "cache_hit_rate": agg["cache_hit_rate"],
                    "avg_batch_size": agg["avg_batch_size"],
                }
            )

        return {
            "now": now,
            "uptime_s": uptime,
            "window_s": window_s,
            "step_s": step_s,
            **headline,
            "totals": {
                "requests": self.requests_total,
                "errors": self.errors_total,
                "rejected": self.rejected_total,
                "items": self.items_total,
                "cache_hits": self.cache_hits_total,
                "batches": self.batches_total,
            },
            "series": series,
        }

    def prometheus(self, extra_gauges: dict[str, float] | None = None) -> str:
        lines = [
            "# HELP inference_requests_total Prediction requests handled.",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {self.requests_total}",
            "# HELP inference_request_errors_total Requests that failed with a 5xx status.",
            "# TYPE inference_request_errors_total counter",
            f"inference_request_errors_total {self.errors_total}",
            "# HELP inference_request_rejected_total Requests rejected with a 4xx status.",
            "# TYPE inference_request_rejected_total counter",
            f"inference_request_rejected_total {self.rejected_total}",
            "# HELP inference_items_total Individual texts scored.",
            "# TYPE inference_items_total counter",
            f"inference_items_total {self.items_total}",
            "# HELP inference_cache_hits_total Texts answered from the result cache.",
            "# TYPE inference_cache_hits_total counter",
            f"inference_cache_hits_total {self.cache_hits_total}",
            "# HELP inference_batches_total Model batches executed.",
            "# TYPE inference_batches_total counter",
            f"inference_batches_total {self.batches_total}",
            "# HELP inference_batch_items_total Texts sent to the model across all batches.",
            "# TYPE inference_batch_items_total counter",
            f"inference_batch_items_total {self.batch_items_total}",
            "# HELP inference_request_duration_seconds End-to-end request latency.",
            "# TYPE inference_request_duration_seconds histogram",
        ]
        cumulative = 0
        for bound, count in zip(LATENCY_BUCKETS_S, self._hist):
            cumulative += count
            lines.append(f'inference_request_duration_seconds_bucket{{le="{bound}"}} {cumulative}')
        lines.append(
            f'inference_request_duration_seconds_bucket{{le="+Inf"}} {self.requests_total}'
        )
        lines.append(f"inference_request_duration_seconds_sum {self.latency_sum_s:.6f}")
        lines.append(f"inference_request_duration_seconds_count {self.requests_total}")
        for name, value in (extra_gauges or {}).items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"
