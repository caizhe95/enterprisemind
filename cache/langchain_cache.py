"""LangChain LLM cache bootstrap helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.globals import set_llm_cache

from logger import logger

_CACHE_INITIALIZED = False


def _build_in_memory_cache() -> Any:
    from langchain_core.caches import InMemoryCache

    return InMemoryCache()


def _build_redis_cache(redis_url: str, ttl: int) -> Any:
    import redis
    from langchain_redis import RedisCache

    redis_client = redis.Redis.from_url(redis_url)
    return RedisCache(redis_client, ttl=ttl or None)


def initialize_llm_cache(force: bool = False) -> None:
    """Initialize LangChain's global LLM cache exactly once per process."""
    global _CACHE_INITIALIZED

    if _CACHE_INITIALIZED and not force:
        return

    from config import config

    backend = config.LLM_CACHE_BACKEND

    if backend in {"", "none", "off", "disabled"}:
        set_llm_cache(None)
        _CACHE_INITIALIZED = True
        logger.info("[LLMCache] disabled")
        return

    try:
        if backend in {"memory", "inmemory"}:
            set_llm_cache(_build_in_memory_cache())
        elif backend == "redis":
            set_llm_cache(
                _build_redis_cache(config.REDIS_URL, config.LLM_CACHE_TTL_SECONDS)
            )
        else:
            raise ValueError(f"unsupported LLM_CACHE_BACKEND: {backend}")

        _CACHE_INITIALIZED = True
        logger.info("[LLMCache] backend={} initialized", backend)
    except Exception as exc:
        if config.LLM_CACHE_FAIL_OPEN:
            set_llm_cache(None)
            _CACHE_INITIALIZED = True
            logger.warning(
                "[LLMCache] init failed, cache disabled. backend={} error={}",
                backend,
                str(exc),
            )
            return
        raise
