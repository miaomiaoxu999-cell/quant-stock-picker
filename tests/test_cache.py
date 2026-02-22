"""单元测试：SQLite 缓存层"""

import os
import tempfile
import pandas as pd
import pytest

from quant.data.cache import DataCache


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    return DataCache(db_path)


def test_set_and_get(cache):
    df = pd.DataFrame({"code": ["000001", "000002"], "name": ["平安银行", "万科A"]})
    cache.set("test_key", df, expire_hours=1.0)
    result = cache.get("test_key")
    assert result is not None
    assert len(result) == 2
    assert list(result.columns) == ["code", "name"]


def test_get_nonexistent(cache):
    result = cache.get("nonexistent_key")
    assert result is None


def test_expiry(cache):
    df = pd.DataFrame({"a": [1]})
    cache.set("expire_test", df, expire_hours=0.0001)  # ~0.36 seconds
    import time
    time.sleep(0.5)
    result = cache.get("expire_test")
    assert result is None


def test_delete(cache):
    df = pd.DataFrame({"a": [1]})
    cache.set("del_key", df)
    cache.delete("del_key")
    assert cache.get("del_key") is None


def test_clear_all(cache):
    df = pd.DataFrame({"a": [1]})
    cache.set("k1", df)
    cache.set("k2", df)
    cache.clear_all()
    assert cache.get("k1") is None
    assert cache.get("k2") is None


def test_overwrite(cache):
    df1 = pd.DataFrame({"val": [1]})
    df2 = pd.DataFrame({"val": [2]})
    cache.set("overwrite", df1)
    cache.set("overwrite", df2)
    result = cache.get("overwrite")
    assert result is not None
    assert result.iloc[0]["val"] == 2
