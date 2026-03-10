import uuid
from pathlib import Path

from cache.cache_manager import CacheManager


def test_cache_manager_supports_persistent_cache():
    manager = CacheManager()
    db_path = Path("tests") / f"cache_test_{uuid.uuid4().hex}.db"
    manager.persistent = manager.persistent.__class__(str(db_path))

    @manager.cached(cache_type="persistent", ttl=60, key_prefix="unit")
    def _expensive(value: str):
        return {"value": value}

    first = _expensive("demo")
    second = _expensive("demo")
    stats = manager.get_stats()

    assert first == {"value": "demo"}
    assert second == {"value": "demo"}
    assert stats["persistent"]["type"] == "persistent"
    assert stats["persistent"]["size"] >= 1
    assert stats["persistent"]["hits"] >= 1
