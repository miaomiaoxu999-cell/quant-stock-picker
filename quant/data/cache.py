"""SQLite 本地缓存层"""

import sqlite3
import io
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class DataCache:
    """SQLite 缓存，避免重复请求 API"""

    _DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "cache.db"

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else self._DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expire_hours REAL NOT NULL
                )
            """)
            conn.commit()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存，过期返回 None"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data, created_at, expire_hours FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        data_json, created_at, expire_hours = row
        elapsed_hours = (time.time() - created_at) / 3600

        if elapsed_hours > expire_hours:
            logger.debug(f"Cache expired: {key}")
            self.delete(key)
            return None

        try:
            df = pd.read_json(io.StringIO(data_json), orient="records")
            logger.debug(f"Cache hit: {key} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache parse error for {key}: {e}")
            self.delete(key)
            return None

    def set(self, key: str, df: pd.DataFrame, expire_hours: float = 24.0):
        """写入缓存"""
        data_json = df.to_json(orient="records", date_format="iso", force_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, created_at, expire_hours) VALUES (?, ?, ?, ?)",
                (key, data_json, time.time(), expire_hours),
            )
            conn.commit()
        logger.debug(f"Cache set: {key} ({len(df)} rows, expire={expire_hours}h)")

    def delete(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear_expired(self):
        """清理所有过期缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache WHERE (? - created_at) / 3600.0 > expire_hours",
                (time.time(),),
            )
            conn.commit()
        logger.info("Cleared expired cache entries")

    def clear_pattern(self, pattern: str):
        """清除匹配 pattern 的缓存（SQL LIKE 语法，如 'pb_%'）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE key LIKE ?", (pattern,)
            )
            conn.commit()
        logger.info(f"Cleared {cursor.rowcount} cache entries matching '{pattern}'")

    def clear_all(self):
        """清空全部缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cleared all cache")
