"""Layered cache manager with pluggable persistent backends."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol


@dataclass
class CacheStats:
    """Basic cache stats shared by all backends."""

    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Persistent cache entry metadata."""

    key: str
    value: Any
    created_at: float
    expire_at: Optional[float]
    access_count: int = 0


class CacheBackend(Protocol):
    stats: CacheStats

    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ...

    def clear(self):
        ...

    def get_stats(self) -> Dict[str, Any]:
        ...


class MemoryCache:
    """Simple in-process LRU cache."""

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []
        self.max_size = max_size
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            item = self._cache[key]
            if item["expire"] and time.time() > item["expire"]:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self.stats.misses += 1
                self.stats.size = len(self._cache)
                return None

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self.stats.hits += 1
            return item["value"]

        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        expire_time = time.time() + ttl if ttl else None

        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = {"value": value, "expire": expire_time}

        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        self.stats.size = len(self._cache)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "memory",
            "size": len(self._cache),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.2%}",
        }


class SQLitePersistentCache:
    """Persistent cache backed by SQLite."""

    def __init__(
        self, db_path: str = "./cache/persistent_cache.db", default_ttl: int = 3600
    ):
        self.backend_name = "sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache
                (
                    key          TEXT PRIMARY KEY,
                    value        TEXT NOT NULL,
                    created_at   REAL    DEFAULT (unixepoch()),
                    expire_at    REAL,
                    access_count INTEGER DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expire ON cache(expire_at)")
            conn.execute("DELETE FROM cache WHERE expire_at < unixepoch()")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT value, expire_at, access_count
                    FROM cache
                    WHERE key = ?
                      AND (expire_at IS NULL OR expire_at > unixepoch())
                    """,
                    (key,),
                )
                row = cursor.fetchone()

                if row:
                    value, _expire_at, count = row
                    conn.execute(
                        "UPDATE cache SET access_count = ? WHERE key = ?",
                        (count + 1, key),
                    )
                    conn.commit()
                    self.stats.hits += 1
                    return json.loads(value)

                self.stats.misses += 1
                return None
        except Exception as exc:
            print(f"[SQLitePersistentCache] 读取错误: {exc}")
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            expire_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            serialized = json.dumps(value, ensure_ascii=False, default=str)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, expire_at, access_count)
                    VALUES (?, ?, ?, 0)
                    """,
                    (key, serialized, expire_at),
                )
                conn.commit()
                self.stats.size = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[
                    0
                ]
        except Exception as exc:
            print(f"[SQLitePersistentCache] 写入错误: {exc}")

    def delete(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE expire_at < unixepoch()"
                ).fetchone()[0]
        except Exception:
            total = 0
            expired = 0

        return {
            "type": self.backend_name,
            "total_keys": total,
            "expired_keys": expired,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.2%}",
            "db_path": str(self.db_path),
        }

    def get_cache_info(self, key: str) -> Optional[CacheEntry]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT value, created_at, expire_at, access_count
                    FROM cache
                    WHERE key = ?
                    """,
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    return CacheEntry(
                        key=key,
                        value=json.loads(row[0]),
                        created_at=row[1],
                        expire_at=row[2],
                        access_count=row[3],
                    )
        except Exception as exc:
            print(f"[SQLitePersistentCache] 查询错误: {exc}")
        return None


class RedisPersistentCache:
    """Persistent cache backed by Redis."""

    def __init__(self, redis_url: str, default_ttl: int = 3600, prefix: str = "cache:"):
        import redis
        import redis.asyncio as redis_asyncio

        self.backend_name = "redis"
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.async_client = redis_asyncio.Redis.from_url(redis_url, decode_responses=True)
        self.stats = CacheStats()

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(self._key(key))
            if value is None:
                self.stats.misses += 1
                return None
            self.client.hincrby(self._key(f"meta:{key}"), "access_count", 1)
            self.stats.hits += 1
            return json.loads(value)
        except Exception as exc:
            print(f"[RedisPersistentCache] 读取错误: {exc}")
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            expire_seconds = ttl if ttl is not None else self.default_ttl
            payload = json.dumps(value, ensure_ascii=False, default=str)
            cache_key = self._key(key)
            meta_key = self._key(f"meta:{key}")
            pipeline = self.client.pipeline()
            pipeline.set(cache_key, payload, ex=expire_seconds or None)
            pipeline.hset(
                meta_key,
                mapping={
                    "created_at": time.time(),
                    "expire_at": time.time() + expire_seconds if expire_seconds else "",
                    "access_count": 0,
                },
            )
            if expire_seconds:
                pipeline.expire(meta_key, expire_seconds)
            pipeline.execute()
        except Exception as exc:
            print(f"[RedisPersistentCache] 写入错误: {exc}")

    def clear(self):
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        finally:
            self.stats = CacheStats()

    async def aget(self, key: str) -> Optional[Any]:
        try:
            value = await self.async_client.get(self._key(key))
            if value is None:
                self.stats.misses += 1
                return None
            await self.async_client.hincrby(self._key(f"meta:{key}"), "access_count", 1)
            self.stats.hits += 1
            return json.loads(value)
        except Exception as exc:
            print(f"[RedisPersistentCache] 异步读取错误: {exc}")
            self.stats.misses += 1
            return None

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            expire_seconds = ttl if ttl is not None else self.default_ttl
            payload = json.dumps(value, ensure_ascii=False, default=str)
            cache_key = self._key(key)
            meta_key = self._key(f"meta:{key}")
            pipeline = self.async_client.pipeline()
            await pipeline.set(cache_key, payload, ex=expire_seconds or None)
            await pipeline.hset(
                meta_key,
                mapping={
                    "created_at": time.time(),
                    "expire_at": time.time() + expire_seconds if expire_seconds else "",
                    "access_count": 0,
                },
            )
            if expire_seconds:
                await pipeline.expire(meta_key, expire_seconds)
            await pipeline.execute()
        except Exception as exc:
            print(f"[RedisPersistentCache] 异步写入错误: {exc}")

    async def aclear(self):
        try:
            keys = await self.async_client.keys(f"{self.prefix}*")
            if keys:
                await self.async_client.delete(*keys)
        finally:
            self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        try:
            total_keys = len(self.client.keys(f"{self.prefix}*"))
        except Exception:
            total_keys = 0
        return {
            "type": self.backend_name,
            "total_keys": total_keys,
            "expired_keys": 0,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.2%}",
            "redis_prefix": self.prefix,
        }


def _build_persistent_cache() -> CacheBackend:
    from config import config

    backend = getattr(config, "PERSISTENT_CACHE_BACKEND", "sqlite").strip().lower()
    default_ttl = getattr(config, "PERSISTENT_CACHE_TTL_SECONDS", 3600)

    if backend == "redis":
        return RedisPersistentCache(
            redis_url=config.REDIS_URL,
            default_ttl=default_ttl,
            prefix=getattr(config, "PERSISTENT_CACHE_REDIS_PREFIX", "persistent_cache:"),
        )
    return SQLitePersistentCache(
        db_path=getattr(config, "PERSISTENT_CACHE_DB_PATH", "./cache/persistent_cache.db"),
        default_ttl=default_ttl,
    )


class CacheManager:
    """Facade for memory cache + configurable persistent cache."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.memory = MemoryCache(max_size=1000)
        self.persistent = _build_persistent_cache()
        self._initialized = True
        print("[CacheManager] 缓存管理器初始化完成")

    def cached(self, cache_type: str = "memory", ttl: int = 3600, key_prefix: str = ""):
        """Decorator for memoizing function results."""

        def decorator(func: Callable):
            cache = self.memory if cache_type == "memory" else self.persistent

            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = (
                    f"{key_prefix}:{self._generate_key(func.__name__, *args, **kwargs)}"
                )

                result = cache.get(cache_key)
                if result is not None:
                    print(f"[Cache] 命中: {func.__name__}")
                    return result

                result = func(*args, **kwargs)
                if result and not str(result).startswith("错误"):
                    cache.set(cache_key, result, ttl=ttl)
                return result

            def clear_func():
                cache.clear()

            wrapper.cache_clear = clear_func
            return wrapper

        return decorator

    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        content = json.dumps(
            {"func": func_name, "args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(content.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        persistent_stats = self.persistent.get_stats()
        return {
            "memory": self.memory.get_stats(),
            "persistent": {
                "type": persistent_stats.get("type", "persistent"),
                "size": persistent_stats.get("total_keys", persistent_stats.get("size", 0)),
                "expired_keys": persistent_stats.get("expired_keys", 0),
                "hits": persistent_stats.get("hits", 0),
                "misses": persistent_stats.get("misses", 0),
                "hit_rate": persistent_stats.get("hit_rate", "0.00%"),
                "db_path": persistent_stats.get("db_path"),
                "redis_prefix": persistent_stats.get("redis_prefix"),
            },
        }

    def clear_all(self):
        self.memory.clear()
        self.persistent.clear()
        print("[CacheManager] 所有缓存已清空")


cache_manager = CacheManager()
