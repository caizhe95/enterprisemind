import uuid
from pathlib import Path

from cache.cache_manager import CacheManager
from cache import langchain_cache as langchain_cache_module


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
    assert stats["persistent"]["size"] >= 1
    assert stats["persistent"]["hits"] >= 1


def test_initialize_llm_cache_uses_in_memory_backend(monkeypatch):
    captured = []

    monkeypatch.setattr(langchain_cache_module, "_CACHE_INITIALIZED", False)
    monkeypatch.setattr(langchain_cache_module, "_build_in_memory_cache", lambda: "memory-cache")
    monkeypatch.setattr(langchain_cache_module, "set_llm_cache", lambda cache: captured.append(cache))
    monkeypatch.setattr("config.config.LLM_CACHE_BACKEND", "memory")

    langchain_cache_module.initialize_llm_cache(force=True)

    assert captured == ["memory-cache"]
