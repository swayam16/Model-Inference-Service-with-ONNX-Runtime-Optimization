import pytest

from app.cache import TTLCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_miss_then_hit():
    cache = TTLCache()
    assert cache.get("a") is None
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hit_rate"] == 0.5


def test_entries_expire_after_ttl():
    clock = Clock()
    cache = TTLCache(ttl_s=10, clock=clock)
    cache.set("a", 1)
    clock.now = 9.9
    assert cache.get("a") == 1
    clock.now = 10.0
    assert cache.get("a") is None
    assert len(cache) == 0


def test_least_recently_used_is_evicted():
    cache = TTLCache(max_items=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # "a" is now the most recently used
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.stats()["evictions"] == 1


def test_key_depends_on_namespace_and_text():
    assert TTLCache.make_key("m1", "hello") == TTLCache.make_key("m1", "hello")
    assert TTLCache.make_key("m1", "hello") != TTLCache.make_key("m2", "hello")
    assert TTLCache.make_key("m1", "hello") != TTLCache.make_key("m1", "hello ")


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        TTLCache(max_items=0)
