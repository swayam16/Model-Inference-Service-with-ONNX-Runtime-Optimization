"""In-memory LRU result cache with per-entry TTL.

Inference is deterministic for a given (model, text), so repeated inputs can be
answered without touching the model. The cache is only ever accessed from the
event loop thread, so it needs no locking.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class TTLCache:
    def __init__(
        self,
        max_items: int = 10_000,
        ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def make_key(namespace: str, text: str) -> str:
        """Key on the model identity plus the exact input text."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (self._clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "max_items": self.max_items,
            "ttl_s": self.ttl_s,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": (self.hits / total) if total else 0.0,
        }
