"""SQLite持久化缓存 - 用于SQL生成等长期缓存"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """缓存条目"""

    key: str
    value: Any
    created_at: float
    expire_at: Optional[float]
    access_count: int = 0


class PersistentCache:
    """基于SQLite的持久化缓存"""

    def __init__(
        self, db_path: str = "./cache/persistent_cache.db", default_ttl: int = 3600
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._init_db()

        # 统计
        self._hits = 0
        self._misses = 0

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS cache
                         (
                             key          TEXT PRIMARY KEY,
                             value        TEXT NOT NULL,
                             created_at   REAL    DEFAULT (unixepoch()),
                             expire_at    REAL,
                             access_count INTEGER DEFAULT 0
                         )
                         """)
            # 索引加速过期查询
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expire ON cache(expire_at)")
            # 清理过期数据
            conn.execute("DELETE FROM cache WHERE expire_at < unixepoch()")
            conn.commit()

    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        content = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 查询并更新访问计数
                cursor = conn.execute(
                    """SELECT value, expire_at, access_count
                       FROM cache
                       WHERE key = ?
                         AND (expire_at IS NULL OR expire_at > unixepoch())""",
                    (key,),
                )
                row = cursor.fetchone()

                if row:
                    value, expire_at, count = row
                    # 更新访问计数
                    conn.execute(
                        "UPDATE cache SET access_count = ? WHERE key = ?",
                        (count + 1, key),
                    )
                    conn.commit()

                    self._hits += 1
                    return json.loads(value)

                self._misses += 1
                return None

        except Exception as e:
            print(f"[PersistentCache] 读取错误: {e}")
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        try:
            expire_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            serialized = json.dumps(value, ensure_ascii=False, default=str)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO cache (key, value, expire_at, access_count)
                       VALUES (?, ?, ?, 0)""",
                    (key, serialized, expire_at),
                )
                conn.commit()
        except Exception as e:
            print(f"[PersistentCache] 写入错误: {e}")

    def delete(self, key: str):
        """删除缓存项"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self):
        """清空所有缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict:
        """获取缓存统计"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE expire_at < unixepoch()"
                ).fetchone()[0]
        except Exception:
            total = 0
            expired = 0

        total_ops = self._hits + self._misses
        hit_rate = self._hits / total_ops if total_ops > 0 else 0

        return {
            "total_keys": total,
            "expired_keys": expired,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}",
            "db_path": str(self.db_path),
        }

    def get_cache_info(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目详细信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, created_at, expire_at, access_count FROM cache WHERE key = ?",
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


# 单例实例（可选）
_default_cache = None


def get_persistent_cache(
    db_path: str = "./cache/persistent_cache.db",
) -> PersistentCache:
    """获取默认缓存实例"""
    global _default_cache
    if _default_cache is None:
        _default_cache = PersistentCache(db_path)
    return _default_cache
