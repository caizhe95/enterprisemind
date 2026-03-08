"""分层缓存管理器 - 支持内存+持久化"""

import hashlib
import json
import time
from typing import Any, Optional, Dict, Callable
from functools import wraps
from pathlib import Path
import sqlite3
from dataclasses import dataclass


@dataclass
class CacheStats:
    """缓存统计"""

    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryCache:
    """LRU内存缓存"""

    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._access_order = []
        self.max_size = max_size
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            item = self._cache[key]
            if item["expire"] and time.time() > item["expire"]:
                del self._cache[key]
                self.stats.misses += 1
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

    def get_stats(self) -> Dict:
        return {
            "type": "memory",
            "size": len(self._cache),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.2%}",
        }


class PersistentCache:
    """SQLite持久化缓存"""

    def __init__(
        self, db_path: str = "./cache/persistent_cache.db", default_ttl: int = 3600
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._init_db()
        self.stats = CacheStats()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expire_at REAL,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expire ON cache(expire_at)")
            conn.execute("DELETE FROM cache WHERE expire_at < unixepoch()")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM cache WHERE key = ? AND (expire_at IS NULL OR expire_at > unixepoch())",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    self.stats.hits += 1
                    return json.loads(row[0])
                self.stats.misses += 1
                return None
        except Exception:
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            expire_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            serialized = json.dumps(value, ensure_ascii=False, default=str)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire_at) VALUES (?, ?, ?)",
                    (key, serialized, expire_at),
                )
                conn.commit()
                self.stats.size = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[
                    0
                ]
        except Exception:
            pass

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def get_stats(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        except Exception:
            count = 0

        return {
            "type": "persistent",
            "size": count,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.2%}",
        }


class CacheManager:
    """缓存管理器门面"""

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
        self.persistent = PersistentCache()
        self._initialized = True
        print("[CacheManager] 缓存管理器初始化完成")

    def cached(self, cache_type: str = "memory", ttl: int = 3600, key_prefix: str = ""):
        """装饰器：缓存函数结果"""

        def decorator(func: Callable):
            # 在这里获取 cache 实例，这样内部函数都能访问到
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

            # 现在 cache 变量在作用域内了
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

    def get_stats(self) -> Dict:
        return {
            "memory": self.memory.get_stats(),
            "persistent": self.persistent.get_stats(),
        }

    def clear_all(self):
        self.memory.clear()
        self.persistent.clear()
        print("[CacheManager] 所有缓存已清空")


# 全局实例
cache_manager = CacheManager()
