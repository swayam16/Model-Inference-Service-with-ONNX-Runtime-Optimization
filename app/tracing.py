"""Per-request tracing.

Every request gets a trace: a request id plus the time spent in each stage
(cache lookup, queue wait, tokenization, model inference). The trace is

* returned to the caller in the ``X-Request-ID`` and ``Server-Timing`` headers,
* written as one structured JSON log line,
* kept in a small ring buffer that the dashboard reads from ``/traces``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("inference.trace")


@dataclass
class Trace:
    request_id: str
    method: str
    path: str
    started_at: float = field(default_factory=time.time)
    spans: dict[str, float] = field(default_factory=dict)  # stage -> milliseconds
    attrs: dict[str, Any] = field(default_factory=dict)
    status: int = 0
    total_ms: float = 0.0

    def add_span(self, name: str, ms: float) -> None:
        """Record time for a stage. Parallel items in one request keep the slowest."""
        self.spans[name] = max(self.spans.get(name, 0.0), ms)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def server_timing(self) -> str:
        parts = [f"{name};dur={ms:.2f}" for name, ms in self.spans.items()]
        parts.append(f"total;dur={self.total_ms:.2f}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ts": self.started_at,
            "method": self.method,
            "path": self.path,
            "status": self.status,
            "total_ms": round(self.total_ms, 3),
            "spans_ms": {k: round(v, 3) for k, v in self.spans.items()},
            **self.attrs,
        }


def new_request_id(incoming: str | None = None) -> str:
    """Reuse a caller-supplied id (so traces join up across services) or mint one."""
    if incoming and len(incoming) <= 64 and incoming.replace("-", "").replace("_", "").isalnum():
        return incoming
    return uuid.uuid4().hex[:16]


class TraceBuffer:
    def __init__(self, size: int = 200) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=size)

    def record(self, trace: Trace) -> None:
        entry = trace.to_dict()
        self._items.append(entry)
        logger.info(json.dumps(entry, separators=(",", ":")))

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._items)[-max(limit, 0) :] if limit > 0 else []
        return items[::-1]
