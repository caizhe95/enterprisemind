"""分层缓存管理器 - 支持内存+持久化。"""

import hashlib
import json
import time
from typing import Any, Optional, Dict, Callable
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
import sqlite3


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


@dataclass
class CacheEntry:
    """持久化缓存条目。"""

    key: str
    value: Any
    created_at: float
    expire_at: Optional[float]
    access_count: int = 0


class PersistentCache:
    """基于SQLite的持久化缓存。"""

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

    def _generate_key(self, *args, **kwargs) -> str:
        content = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()

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
        except Exception as e:
            print(f"[PersistentCache] 读取错误: {e}")
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
        except Exception as e:
            print(f"[PersistentCache] 写入错误: {e}")

    def delete(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        self.stats = CacheStats()

    def get_stats(self) -> Dict:
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
        except Exception as e:
            print(f"[PersistentCache] 查询错误: {e}")
        return None


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
        persistent_stats = self.persistent.get_stats()
        return {
            "memory": self.memory.get_stats(),
            "persistent": {
                "type": "persistent",
                "size": persistent_stats.get("total_keys", 0),
                "expired_keys": persistent_stats.get("expired_keys", 0),
                "hits": persistent_stats.get("hits", 0),
                "misses": persistent_stats.get("misses", 0),
                "hit_rate": persistent_stats.get("hit_rate", "0.00%"),
                "db_path": persistent_stats.get("db_path"),
            },
        }

    def clear_all(self):
        self.memory.clear()
        self.persistent.clear()
        print("[CacheManager] 所有缓存已清空")


# 全局实例
cache_manager = CacheManager()
