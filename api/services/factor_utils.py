"""Factor parsing/validation utilities extracted from quant/dashboard/sector_factors.py.

These functions are pure logic with no Streamlit dependency,
so they can be safely imported by FastAPI routers.
"""

from __future__ import annotations

import json
import re


PRESET_SECTORS = [
    # 新能源
    "锂电池/锂盐", "光伏", "光伏玻璃", "风电", "储能",
    # 传统能源
    "煤炭", "石油开采", "炼化",
    # 金属
    "钢铁", "铜", "铝", "稀土",
    # 化工
    "基础化工", "磷化工", "氟化工", "钛白粉",
    # 建筑地产链
    "水泥", "浮法玻璃", "建筑工程", "建材", "房地产",
    # 农业
    "养殖(猪周期)", "白糖", "种植业",
    # 交运
    "干散货航运", "集装箱航运", "油运", "造纸",
    # 制造
    "工程机械", "重卡/商用车",
    # 科技
    "半导体", "面板/显示", "存储芯片", "消费电子",
    # 消费
    "白酒",
]


def extract_json_from_text(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON，支持 code fence / 裸 JSON / 正则"""
    # 1) ```json ... ``` 块
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) 裸 JSON 对象
    m = re.search(r"\{[\s\S]*\"factors\"[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_factors(data: dict) -> list[dict] | None:
    """校验因子数据，合法则返回 factors 列表"""
    if not isinstance(data, dict) or "factors" not in data:
        return None
    factors = data["factors"]
    if not isinstance(factors, list) or not (2 <= len(factors) <= 5):
        return None
    total = sum(f.get("weight", 0) for f in factors)
    if not (95 <= total <= 105):
        return None
    for f in factors:
        if not all(k in f for k in ("name", "weight", "description", "data_source")):
            return None
    return factors
