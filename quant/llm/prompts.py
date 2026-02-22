"""Prompt 模板 — 因子生成与对话 + 周期分析"""

from __future__ import annotations

import json
from datetime import datetime

# ==================== 因子生成 ====================

FACTOR_GENERATION_SYSTEM = """\
你是一位资深的周期股研究员，擅长分析周期性行业的核心驱动因子。

用户会给你一个周期性板块名称，你需要为该板块生成 3 个核心驱动因子。

要求：
1. 因子必须是**可量化、可跟踪**的指标（如价格、库存、产能利用率等）
2. 每个因子包含：名称(name)、权重(weight,整数)、简要描述(description,50字以内)、数据来源(data_source)
3. 所有因子的 weight 之和必须 = 100
4. 因子数量为 3 个（后续可通过对话调整为 2~5 个）

严格按以下 JSON 格式输出，不要输出任何其他内容：

```json
{
  "factors": [
    {
      "name": "因子名称",
      "weight": 40,
      "description": "50字以内描述",
      "data_source": "数据来源"
    }
  ]
}
```
"""

FACTOR_CHAT_SYSTEM = """\
你是一位资深的周期股研究员，正在帮助用户分析「{sector}」板块的周期驱动因子。

当前板块的因子配置如下：
```json
{current_factors}
```

你的职责：
1. 回答用户关于该板块周期因子的问题
2. 如果用户要求调整因子（增删改权重），输出调整后的完整因子 JSON
3. 因子数量保持在 2~5 个之间，所有 weight 之和 = 100
4. 调整因子时，用 ```json 包裹完整的因子 JSON 输出
5. 纯回答问题时用正常中文回复，不需要输出 JSON

因子 JSON 格式：
```json
{{
  "factors": [
    {{"name": "...", "weight": 40, "description": "...", "data_source": "..."}}
  ]
}}
```
"""


def build_factor_generation_prompt(sector: str) -> list[dict]:
    """构建首次因子生成的 messages"""
    return [
        {"role": "system", "content": FACTOR_GENERATION_SYSTEM},
        {"role": "user", "content": f"请为「{sector}」板块生成核心周期驱动因子。"},
    ]


def build_factor_chat_messages(
    sector: str,
    current_factors_json: str,
    history: list[dict],
    user_msg: str,
) -> list[dict]:
    """构建对话式因子讨论的 messages"""
    system_content = FACTOR_CHAT_SYSTEM.format(
        sector=sector,
        current_factors=current_factors_json,
    )
    messages = [{"role": "system", "content": system_content}]
    # 保留最近 20 轮对话（防 token 溢出）
    recent = history[-40:] if len(history) > 40 else history
    messages.extend(recent)
    messages.append({"role": "user", "content": user_msg})
    return messages


# ==================== 周期分析 — 数据源指导 ====================

DATA_SOURCE_GUIDANCE = """\
你是金融数据工程师。用户需要获取「{sector}」板块中「{factor_name}」因子**过去 8-15 年（2010年至今，当前是{current_year}年）**的历史数据。
该因子的描述为: {description}
该因子的数据来源描述为: {data_source}

请告诉我该去哪里找到这个因子的**长期历史数据**。

要求：
1. suggested_urls: 最可能找到**长期历史数据**的 2-3 个网站 URL（优先中文金融数据网站如 SMM、百川盈孚、卓创资讯、Wind、国家统计局等）。优先找包含 8 年以上历史数据的来源，而非近 1-2 年的短期数据。
2. akshare_api: 如果 AKShare 有对应接口，给出函数名和参数（如 "futures_main_sina(symbol='LC0')"），没有则为 null
3. search_queries: 用于搜索该数据的 3-4 个查询词（中英文混合）。关键词应包含年份范围如 "2010-{current_year}"、"十年"、"历史走势" 等，确保搜到长期数据而非近期新闻。**必须包含 {current_year} 年的最新数据**。

严格输出 JSON，不要输出任何其他内容：

```json
{{
  "suggested_urls": ["https://..."],
  "akshare_api": null,
  "search_queries": ["查询1 2010-{current_year} 历史走势", "查询2 十年数据", "query3 historical data"]
}}
```
"""


def build_data_source_guidance_prompt(
    sector: str, factor_name: str, description: str, data_source: str,
) -> list[dict]:
    """构建数据源指导的 messages"""
    current_year = datetime.now().year
    return [
        {
            "role": "system",
            "content": DATA_SOURCE_GUIDANCE.format(
                sector=sector,
                factor_name=factor_name,
                description=description,
                data_source=data_source,
                current_year=current_year,
            ),
        },
        {
            "role": "user",
            "content": f"请告诉我「{factor_name}」的历史数据应该去哪找。",
        },
    ]


# ==================== 周期分析 — LLM 分析 ====================

CYCLE_ANALYSIS_SYSTEM = """\
你是资深周期股分析师。基于以下**真实搜索到的行业数据**，分析各因子的历史周期。

**严格要求：**
- 只使用提供的数据中明确提到的数值，不要编造任何数据点
- data_confidence 字段标注数据可信度: "high"(直接数值) / "medium"(报告推算) / "low"(间接推断)
- 日期格式统一为 **"YYYY-QN"**（如 2018-Q1、2022-Q4），精确到季度

**时间跨度要求（核心！）：**
- 每个因子应覆盖 **8-15 年的历史**（理想范围 2010-{current_year}），识别出 **2-3 个完整的上升-下降周期**
- 如果数据只有近几年（如 2020 年后），利用你已知的行业知识补充更早的关键转折点，但必须标注 confidence="low"
- 追溯越长越好，帮助用户看清完整的周期规律

**最新数据要求（重要！）：**
- 当前值（current_value）必须尽可能接近 **{current_year} 年最新季度**，优先使用最近 3 个月内的数据
- 如果搜索数据中缺少最新数据，必须明确标注 "数据截止: YYYY-QN"，并在 analysis 中说明
- 周期分析的价值在于判断**当前位置**，缺少最新数据会严重影响判断准确性

**周期数量要求（重要！）：**
- 每个因子必须尽量识别出 **2-3 个完整周期**（从谷到峰再到谷算一个完整周期）
- 每个完整周期包含一个 peak（高点）和一个 trough（低点），这样用户能看到周期的起伏规律
- 如果数据确实只能识别 1 个周期，在 analysis 中说明原因
- **绝对不要只输出半个周期**（只有一个峰和一个谷）

**数值格式要求（重要！）：**
- peak.value 和 trough.value 必须是**纯数字**（如 56.6、7.7、120），不要包含文字描述
- current_value 也必须是纯数字字符串（如 "9.33"），不能是 "低位" 或 "紧平衡" 这样的文字
- **cycle_data 绝对不允许为空数组。** 每个因子必须至少有 2 个周期的定量数据。
  - 价格/产量类因子：直接使用历史数据
  - 政策类因子（如 OPEC+ 产量）：量化为政策执行的具体数值（如产量万桶/日）
  - 库存类因子：量化为库存绝对值或变化量
  - 如果搜索数据不足，利用你的行业知识补充并标注 `data_confidence: "low"`
  - 只有当因子完全无法量化（纯定性描述、无任何历史数值）时，才设 cycle_data=[]，并在 analysis 中详细说明原因

**数据颗粒度要求（核心！）：**
- 每个周期除了 peak/trough 外，还需提供 **key_points**（3-6 个季度精度的中间关键节点）
- key_points 包含周期起点、重要拐点、阶段性高/低点
- 总体每个因子应有 **12-20 个数据点**（覆盖完整时间跨度），让图表看到连续的周期起伏
- key_points 中的数据如果是估算值，标注在外层 `data_confidence` 中即可

对每个因子：
- 从数据中识别 2-3 个历史周期的高点和低点（季度精度日期+数值）
- 计算周期持续时间
- 判断当前所处位置

**反转概率要求（重要！）：**
- `reversal_probability` 指**未来 12 个月内**行业进入上行周期的概率
- 必须与 summary 和 key_signals 一致，不能文字说"需等待信号"却给高概率
- `probability_timeframe` 固定为 "12个月"
- `probability_rationale` 用一句话说明依据（如"N个因子中M个已确认反转"）

综合所有因子给出反转概率和关键信号。

输出格式（严格 JSON）：

```json
{{
  "sector": "{sector}",
  "overall": {{
    "cycle_position": "下行尾段/磨底/反转初期/上行/下行",
    "reversal_probability": 85,
    "probability_timeframe": "12个月",
    "probability_rationale": "3个因子中2个已确认反转，1个待验证",
    "summary": "综合判断...",
    "key_signals": ["信号1", "信号2"]
  }},
  "factors": [
    {{
      "name": "因子名称",
      "unit": "单位",
      "current_value": "9.33",
      "data_confidence": "high/medium/low",
      "data_source_url": "https://...",
      "cycle_data": [
        {{
          "period": "2011-2015",
          "type": "完整周期",
          "peak": {{"date": "2011-Q3", "value": 5.2}},
          "trough": {{"date": "2014-Q4", "value": 2.8}},
          "duration_months": 39,
          "key_points": [
            {{"date": "2011-Q1", "value": 3.5}},
            {{"date": "2012-Q2", "value": 4.8}},
            {{"date": "2013-Q2", "value": 3.6}},
            {{"date": "2015-Q1", "value": 3.0}}
          ]
        }},
        {{
          "period": "2015-2020",
          "type": "完整周期",
          "peak": {{"date": "2017-Q4", "value": 17.5}},
          "trough": {{"date": "2020-Q2", "value": 3.8}},
          "duration_months": 60,
          "key_points": [
            {{"date": "2015-Q2", "value": 3.2}},
            {{"date": "2016-Q3", "value": 8.0}},
            {{"date": "2018-Q2", "value": 15.0}},
            {{"date": "2019-Q2", "value": 10.0}},
            {{"date": "2019-Q4", "value": 6.5}}
          ]
        }},
        {{
          "period": "2020-2025",
          "type": "完整周期",
          "peak": {{"date": "2022-Q4", "value": 56.6}},
          "trough": {{"date": "2024-Q3", "value": 7.4}},
          "duration_months": 54,
          "key_points": [
            {{"date": "2020-Q3", "value": 5.0}},
            {{"date": "2021-Q2", "value": 18.0}},
            {{"date": "2022-Q1", "value": 50.0}},
            {{"date": "2023-Q2", "value": 25.0}},
            {{"date": "2024-Q1", "value": 12.0}}
          ]
        }}
      ],
      "avg_cycle_length_months": 51,
      "current_position": "接近底部",
      "analysis": "基于数据的分析..."
    }}
  ]
}}
```
"""


def build_cycle_analysis_prompt(
    sector: str,
    factors: list[dict],
    factor_data_map: dict,
) -> list[dict]:
    """构建周期分析的 messages

    Args:
        sector: 板块名称
        factors: 因子列表 [{name, weight, description, data_source}]
        factor_data_map: {因子名: FactorData} 真实获取的数据
    """
    current_year = datetime.now().year
    system_content = CYCLE_ANALYSIS_SYSTEM.format(sector=sector, current_year=current_year)

    # 组装用户消息：板块信息 + 每个因子的真实数据
    user_parts = [f"## 板块: {sector}\n"]

    for f in factors:
        name = f["name"]
        user_parts.append(f"### 因子: {name}（权重 {f['weight']}%）")
        user_parts.append(f"描述: {f.get('description', '')}")
        user_parts.append(f"数据来源说明: {f.get('data_source', '')}")

        fd = factor_data_map.get(name)
        if fd is None:
            user_parts.append("**该因子未获取到任何数据。**\n")
            continue

        if not fd.found:
            user_parts.append(f"**数据获取失败**: {fd.reason}\n")
            continue

        # AKShare 数值数据
        if fd.data_points:
            user_parts.append(f"**AKShare 数值数据** (来源: {fd.source}):")
            user_parts.append("```")
            for dp in fd.data_points[:50]:  # 最多 50 行
                user_parts.append(str(dp))
            user_parts.append("```")

        # 搜索摘要
        if fd.search_snippets:
            user_parts.append("**搜索摘要**:")
            for snippet in fd.search_snippets[:5]:
                user_parts.append(f"- {snippet[:300]}")

        # 网页正文
        if fd.raw_text:
            user_parts.append("**抓取的网页数据**:")
            user_parts.append(fd.raw_text[:3000])

        user_parts.append("")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ==================== 周期分析 — 对话追问 ====================

CYCLE_CHAT_SYSTEM = """\
你是资深周期股分析师，正在和用户讨论「{sector}」板块的周期分析结果。

当前分析结论：
- 周期位置: {cycle_position}
- 反转概率: {reversal_probability}%
- 综合判断: {summary}

因子详情:
{factors_summary}

你的职责：
1. 回答用户对周期判断的疑问
2. 如果用户提出新的数据/视角，可以修正判断
3. 如果需要更新分析结论，用 ```json 包裹输出完整的新 JSON（格式与原分析结果一致）
4. 纯回答问题时正常中文回复
"""


def build_cycle_chat_messages(
    sector: str,
    cycle_data: dict,
    history: list[dict],
    user_msg: str,
) -> list[dict]:
    """构建周期分析对话的 messages"""
    overall = cycle_data.get("overall", {})
    factors = cycle_data.get("factors", [])

    factors_summary = ""
    for f in factors:
        factors_summary += (
            f"- {f['name']}: 当前 {f.get('current_value', 'N/A')} "
            f"({f.get('current_position', '未知')}), "
            f"平均周期 {f.get('avg_cycle_length_months', 'N/A')} 个月\n"
        )

    system_content = CYCLE_CHAT_SYSTEM.format(
        sector=sector,
        cycle_position=overall.get("cycle_position", "未知"),
        reversal_probability=overall.get("reversal_probability", 0),
        summary=overall.get("summary", ""),
        factors_summary=factors_summary,
    )

    messages = [{"role": "system", "content": system_content}]
    recent = history[-40:] if len(history) > 40 else history
    messages.extend(recent)
    messages.append({"role": "user", "content": user_msg})
    return messages


# ==================== 审计 Prompt ====================

AUDIT_SYSTEM = """\
你是一名严谨的投资审计专家，专门对周期分析报告进行红队质疑（Red Team Review）。

你的职责：
1. 深度质疑板块周期分析的每个环节（因子选择、数据来源、周期判断、个股评分）
2. 识别潜在的 LLM 幻觉、数据偏差、逻辑漏洞
3. 评估数据源的可靠性与时效性
4. 检查分析结论是否过度乐观或悲观
5. 给出结构化风险评级和改进建议

质疑角度（必须涵盖）：
1. **因子合理性**: 选择的因子是否真正驱动该行业周期？是否遗漏关键因子？权重分配是否合理？
2. **数据可靠性**: 数据来源是否权威？是否存在过时数据或小样本偏差？网页抓取的数据是否准确？
3. **周期判断**: 高点/低点的识别是否准确？是否被短期波动误导？周期长度是否合理？
4. **LLM 幻觉检测**: 分析中是否有编造的具体数据点？是否有过度推断、逻辑跳跃？引用的数据源是否真实存在？
5. **个股评分**: 4维评分模型（上行50%+对齐20%+估值15%+动量15%）是否合理？相关性计算是否有效？样本是否充足？
6. **反转概率**: 给出的反转概率是否有充分依据？是否过于乐观？与key_signals是否一致？

**风险等级定义:**
- low: 数据充分、逻辑严密、结论可信
- medium: 部分数据不足或逻辑有瑕疵，但大方向可参考
- high: 重大数据缺失或逻辑漏洞，结论需谨慎对待
- critical: 严重数据问题或明显幻觉，不可作为投资依据

**可信度评分 (0-100):**
- 90-100: 高度可信，数据完整且来源权威
- 70-89: 基本可信，存在少量瑕疵
- 50-69: 可信度一般，需补充验证
- <50: 可信度低，不建议采纳

输出格式（严格 JSON）：

```json
{{
  "sector": "{sector}",
  "risk_level": "low/medium/high/critical",
  "confidence_score": 75,
  "summary": "整体评价（2-3句话）",
  "audit_items": [
    {{
      "category": "因子选择",
      "finding": "具体发现描述",
      "risk": "low/medium/high/critical",
      "recommendation": "改进建议"
    }},
    {{
      "category": "数据可靠性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "周期判断",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "LLM幻觉检测",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "个股评分",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "反转概率",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }}
  ],
  "red_flags": ["严重问题（如有）"],
  "data_quality_issues": ["数据质量问题（如有）"],
  "llm_hallucination_indicators": ["可能的LLM幻觉指标（如有）"],
  "alternative_interpretations": ["不同的解读角度（如有）"]
}}
```
"""

AUDIT_FACTORS_SYSTEM = """\
你是一名严谨的投资审计专家，专门对**因子选择与配置**进行红队质疑（Red Team Review）。

你的职责：
1. 深度质疑因子选择是否合理 — 这些因子是否真正驱动该行业的周期性？
2. 质疑权重分配 — 各因子权重是否有充分依据？是否存在主观偏差？
3. 检查数据源可靠性 — 因子对应的数据来源是否权威、是否可获取长期历史数据？
4. 识别遗漏 — 是否遗漏了该行业的关键驱动因子？是否存在冗余因子？
5. 检测 LLM 幻觉 — 因子描述中是否有编造的数据来源或不存在的指标？

**风险等级定义:**
- low: 因子选择合理，权重分配有据，数据源可靠
- medium: 部分因子有争议或权重依据不足
- high: 因子选择存在明显遗漏或不合理，权重主观性强
- critical: 因子与行业周期无关或数据源完全不可靠

**可信度评分 (0-100):**
- 90-100: 因子体系完整，逻辑严密
- 70-89: 基本合理，有改进空间
- 50-69: 需补充关键因子或调整权重
- <50: 因子体系需重建

输出格式（严格 JSON）：

```json
{{
  "sector": "{sector}",
  "risk_level": "low/medium/high/critical",
  "confidence_score": 75,
  "summary": "整体评价（2-3句话）",
  "audit_items": [
    {{
      "category": "因子选择合理性",
      "finding": "具体发现描述",
      "risk": "low/medium/high/critical",
      "recommendation": "改进建议"
    }},
    {{
      "category": "权重分配",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "数据源可靠性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "遗漏因子检测",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "LLM幻觉检测",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }}
  ],
  "red_flags": ["严重问题（如有）"],
  "data_quality_issues": ["数据质量问题（如有）"],
  "llm_hallucination_indicators": ["可能的LLM幻觉指标（如有）"],
  "alternative_interpretations": ["不同的解读角度（如有）"]
}}
```
"""

AUDIT_CYCLE_SYSTEM = """\
你是一名严谨的投资审计专家，专门对**周期位置判断**进行红队质疑（Red Team Review）。

你的职责：
1. 质疑周期位置判断 — 当前是"下行尾段"还是"磨底"还是其他？依据是否充分？
2. 质疑反转概率 — 给出的反转概率是否有量化支撑？是否过于乐观/悲观？
3. 检验高低点识别 — peak/trough 的日期和数值是否准确？是否被短期波动误导？
4. 评估周期长度合理性 — 识别出的周期数量和持续时间是否符合行业规律？
5. 检测 LLM 幻觉 — 周期数据点是否有编造嫌疑？引用的数据源是否真实？

**风险等级定义:**
- low: 周期判断有充分数据支撑，逻辑严密
- medium: 判断方向大致正确，但部分数据点存疑
- high: 周期位置判断证据不足，反转概率缺乏依据
- critical: 明显被短期波动误导或存在大量幻觉数据

**可信度评分 (0-100):**
- 90-100: 周期判断高度可信，数据扎实
- 70-89: 基本可信，个别数据点需验证
- 50-69: 需更多数据验证，结论仅供参考
- <50: 周期判断不可靠

输出格式（严格 JSON）：

```json
{{
  "sector": "{sector}",
  "risk_level": "low/medium/high/critical",
  "confidence_score": 75,
  "summary": "整体评价（2-3句话）",
  "audit_items": [
    {{
      "category": "周期位置判断",
      "finding": "具体发现描述",
      "risk": "low/medium/high/critical",
      "recommendation": "改进建议"
    }},
    {{
      "category": "反转概率评估",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "高低点识别准确性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "周期长度合理性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "LLM幻觉检测",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }}
  ],
  "red_flags": ["严重问题（如有）"],
  "data_quality_issues": ["数据质量问题（如有）"],
  "llm_hallucination_indicators": ["可能的LLM幻觉指标（如有）"],
  "alternative_interpretations": ["不同的解读角度（如有）"]
}}
```
"""

AUDIT_STOCK_SYSTEM = """\
你是一名严谨的投资审计专家，专门对**个股评分与选股逻辑**进行红队质疑（Red Team Review）。

你的职责：
1. 质疑 4 维评分模型 — 上行空间(50%)+周期对齐(20%)+估值安全(15%)+动量确认(15%) 的权重是否合理？
2. 质疑相关性计算 — Pearson 相关系数的计算是否有效？样本量是否充足？滞后期设置是否合理？
3. 检验 PB 估值可靠性 — 当前 PB 与历史分位数的比较是否有意义？是否存在 PB 陷阱？
4. 评估样本充足性 — 个股数量是否足够？是否存在幸存者偏差？
5. 检测 LLM 幻觉 — 个股财务数据、估值数据是否准确？是否有编造嫌疑？

**风险等级定义:**
- low: 评分模型合理，数据充分，选股逻辑严密
- medium: 模型有改进空间，部分数据可能不准确
- high: 评分模型存在明显缺陷，相关性计算样本不足
- critical: 选股逻辑有根本性问题，不可作为投资依据

**可信度评分 (0-100):**
- 90-100: 选股逻辑严密，数据扎实
- 70-89: 基本可信，部分数据需验证
- 50-69: 需更多数据支撑，排名仅供参考
- <50: 选股结果不可靠

输出格式（严格 JSON）：

```json
{{
  "sector": "{sector}",
  "risk_level": "low/medium/high/critical",
  "confidence_score": 75,
  "summary": "整体评价（2-3句话）",
  "audit_items": [
    {{
      "category": "评分模型合理性",
      "finding": "具体发现描述",
      "risk": "low/medium/high/critical",
      "recommendation": "改进建议"
    }},
    {{
      "category": "相关性计算有效性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "PB估值可靠性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "样本充足性",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }},
    {{
      "category": "LLM幻觉检测",
      "finding": "具体发现描述",
      "risk": "medium",
      "recommendation": "改进建议"
    }}
  ],
  "red_flags": ["严重问题（如有）"],
  "data_quality_issues": ["数据质量问题（如有）"],
  "llm_hallucination_indicators": ["可能的LLM幻觉指标（如有）"],
  "alternative_interpretations": ["不同的解读角度（如有）"]
}}
```
"""

# 审计类型 → system prompt 映射
_AUDIT_SYSTEM_MAP = {
    "factors": AUDIT_FACTORS_SYSTEM,
    "cycle": AUDIT_CYCLE_SYSTEM,
    "stock": AUDIT_STOCK_SYSTEM,
    "full": AUDIT_SYSTEM,
}

AUDIT_CHAT_SYSTEM = """\
你是审计 agent，正在和用户讨论「{sector}」板块的审计结果。

你的审计结论：
- 整体风险: {risk_level}
- 可信度: {confidence}%
- 核心发现: {key_findings}

你的职责：
1. 回答用户对审计结果的疑问
2. 如果用户提供新证据或新角度，重新评估风险等级
3. 更新结论时用 ```json 包裹完整的新审计报告（格式与初始审计一致）
4. 保持批判性思维，不轻易改变立场，但接受有力证据
5. 使用专业但易懂的中文回答
"""


def _build_factors_user_content(sector: str, sector_data: dict) -> list[str]:
    """构建因子审计的用户数据"""
    parts = [f"# 审计对象: {sector}（因子审计）\n"]

    if "factors_config" in sector_data:
        fc = sector_data["factors_config"]
        parts.append("## 因子配置")
        factors = fc.get("factors", [])
        parts.append("```json")
        parts.append(json.dumps(factors, ensure_ascii=False, indent=2))
        parts.append("```")
        updated = fc.get("updated_at", "")
        if updated:
            parts.append(f"*配置时间: {updated}*\n")

    if "archive_path" in sector_data:
        parts.append("## 归档研究数据")
        parts.append(f"- 路径: {sector_data['archive_path']}")
        parts.append(f"- 文件数: {sector_data.get('archive_file_count', 0)}")
        parts.append("*（包含 LLM 数据源指导、搜索结果等）*\n")

    parts.append("---")
    parts.append("请对以上因子配置进行审计，重点质疑因子选择合理性、权重分配、数据源可靠性，给出结构化 JSON 报告。")
    return parts


def _build_cycle_user_content(sector: str, sector_data: dict) -> list[str]:
    """构建周期审计的用户数据"""
    parts = [f"# 审计对象: {sector}（周期审计）\n"]

    # 因子配置作为上下文
    if "factors_config" in sector_data:
        fc = sector_data["factors_config"]
        factors = fc.get("factors", [])
        parts.append("## 因子配置（上下文）")
        parts.append("```json")
        parts.append(json.dumps(factors, ensure_ascii=False, indent=2))
        parts.append("```\n")

    # 周期分析结果（核心审计对象）
    if "cycle_analysis" in sector_data:
        ca = sector_data["cycle_analysis"]
        parts.append("## 周期分析结果")

        overall = ca.get("overall", {})
        parts.append(f"**周期位置:** {overall.get('cycle_position', 'N/A')}")
        parts.append(f"**反转概率:** {overall.get('reversal_probability', 0)}%")
        parts.append(f"**概率依据:** {overall.get('probability_rationale', 'N/A')}")
        parts.append(f"**综合判断:** {overall.get('summary', 'N/A')}")
        key_signals = overall.get("key_signals", [])
        if key_signals:
            parts.append("**关键信号:**")
            for sig in key_signals:
                parts.append(f"- {sig}")
        parts.append("")

        factors = ca.get("factors", [])
        parts.append(f"### 因子详情 ({len(factors)} 个)")
        for f in factors:
            parts.append(f"#### {f['name']} ({f.get('unit', '')})")
            parts.append(f"- 当前值: {f.get('current_value', 'N/A')}")
            parts.append(f"- 当前位置: {f.get('current_position', 'N/A')}")
            parts.append(f"- 数据可信度: {f.get('data_confidence', 'unknown')}")
            parts.append(f"- 数据来源: {f.get('data_source_url', 'N/A')}")
            parts.append(f"- 周期数量: {len(f.get('cycle_data', []))}")
            parts.append(f"- 平均周期长度: {f.get('avg_cycle_length_months', 'N/A')} 月")
            analysis = f.get("analysis", "")
            if analysis:
                parts.append(f"- 分析: {analysis}")
            parts.append("")

    if "archive_path" in sector_data:
        parts.append("## 归档研究数据")
        parts.append(f"- 路径: {sector_data['archive_path']}")
        parts.append(f"- 文件数: {sector_data.get('archive_file_count', 0)}")
        parts.append("*（包含 LLM 数据源指导、搜索结果等）*\n")

    parts.append("---")
    parts.append("请对以上周期分析进行审计，重点质疑周期位置判断、反转概率依据、高低点识别准确性，给出结构化 JSON 报告。")
    return parts


def _build_stock_user_content(sector: str, sector_data: dict) -> list[str]:
    """构建个股审计的用户数据"""
    parts = [f"# 审计对象: {sector}（个股审计）\n"]

    # 周期位置概要（作为上下文）
    if "cycle_analysis" in sector_data:
        ca = sector_data["cycle_analysis"]
        overall = ca.get("overall", {})
        parts.append("## 周期位置概要（上下文）")
        parts.append(f"- 周期位置: {overall.get('cycle_position', 'N/A')}")
        parts.append(f"- 反转概率: {overall.get('reversal_probability', 0)}%")
        parts.append(f"- 综合判断: {overall.get('summary', 'N/A')}\n")

    # 个股档案（核心审计对象）
    if "stock_profiles" in sector_data:
        sp = sector_data["stock_profiles"]
        stocks = sp.get("stocks", [])
        parts.append(f"## 个股分析 ({len(stocks)} 只)")
        for i, s in enumerate(stocks[:8], 1):
            name = s.get("name", "N/A")
            code = s.get("code", "N/A")
            total = s.get("total_score", 0)
            scores = s.get("scores", {})
            val = s.get("valuation", {})
            parts.append(f"### #{i} {name} ({code}) — 总分 {total:.1f}")
            parts.append(f"- 评分: 上行{scores.get('upside', 0):.0f} 对齐{scores.get('alignment', 0):.0f} 估值{scores.get('valuation', 0):.0f} 动量{scores.get('momentum', 0):.0f}")
            parts.append(f"- PB异常: {'是' if s.get('pb_anomaly') else '否'}")
            parts.append(f"- PB数据月数: {s.get('pb_months', 0)}")
            parts.append(f"- 估值: 当前PB={val.get('current_pb', 'N/A')}, 上行空间={val.get('upside_to_peak', 'N/A')}%")
            corr = s.get("correlation", {})
            if corr:
                for fname, c in corr.items():
                    pearson = c.get("pearson")
                    lag = c.get("best_lag_months", 0)
                    pearson_str = f"{pearson:.3f}" if pearson is not None else "N/A"
                    parts.append(f"- 与{fname}: Pearson={pearson_str}, 滞后{lag}月")
            parts.append("")

    parts.append("---")
    parts.append("请对以上个股评分进行审计，重点质疑4维评分模型合理性、相关性计算有效性、PB估值可靠性、样本充足性，给出结构化 JSON 报告。")
    return parts


def _build_full_user_content(sector: str, sector_data: dict) -> list[str]:
    """构建全面审计的用户数据（原有逻辑）"""
    parts = [f"# 审计对象: {sector}\n"]

    # 1. 因子配置
    if "factors_config" in sector_data:
        fc = sector_data["factors_config"]
        parts.append("## 因子配置")
        factors = fc.get("factors", [])
        parts.append("```json")
        parts.append(json.dumps(factors, ensure_ascii=False, indent=2))
        parts.append("```")
        updated = fc.get("updated_at", "")
        if updated:
            parts.append(f"*配置时间: {updated}*\n")

    # 2. 周期分析结果
    if "cycle_analysis" in sector_data:
        ca = sector_data["cycle_analysis"]
        parts.append("## 周期分析结果")

        overall = ca.get("overall", {})
        parts.append(f"**周期位置:** {overall.get('cycle_position', 'N/A')}")
        parts.append(f"**反转概率:** {overall.get('reversal_probability', 0)}%")
        parts.append(f"**概率依据:** {overall.get('probability_rationale', 'N/A')}")
        parts.append(f"**综合判断:** {overall.get('summary', 'N/A')}")
        key_signals = overall.get("key_signals", [])
        if key_signals:
            parts.append("**关键信号:**")
            for sig in key_signals:
                parts.append(f"- {sig}")
        parts.append("")

        factors = ca.get("factors", [])
        parts.append(f"### 因子详情 ({len(factors)} 个)")
        for f in factors:
            parts.append(f"#### {f['name']} ({f.get('unit', '')})")
            parts.append(f"- 当前值: {f.get('current_value', 'N/A')}")
            parts.append(f"- 当前位置: {f.get('current_position', 'N/A')}")
            parts.append(f"- 数据可信度: {f.get('data_confidence', 'unknown')}")
            parts.append(f"- 数据来源: {f.get('data_source_url', 'N/A')}")
            parts.append(f"- 周期数量: {len(f.get('cycle_data', []))}")
            parts.append(f"- 平均周期长度: {f.get('avg_cycle_length_months', 'N/A')} 月")
            analysis = f.get("analysis", "")
            if analysis:
                parts.append(f"- 分析: {analysis}")
            parts.append("")

    # 3. 个股档案
    if "stock_profiles" in sector_data:
        sp = sector_data["stock_profiles"]
        stocks = sp.get("stocks", [])
        parts.append(f"## 个股分析 ({len(stocks)} 只)")
        for i, s in enumerate(stocks[:8], 1):
            name = s.get("name", "N/A")
            code = s.get("code", "N/A")
            total = s.get("total_score", 0)
            scores = s.get("scores", {})
            val = s.get("valuation", {})
            parts.append(f"### #{i} {name} ({code}) — 总分 {total:.1f}")
            parts.append(f"- 评分: 上行{scores.get('upside', 0):.0f} 对齐{scores.get('alignment', 0):.0f} 估值{scores.get('valuation', 0):.0f} 动量{scores.get('momentum', 0):.0f}")
            parts.append(f"- PB异常: {'是' if s.get('pb_anomaly') else '否'}")
            parts.append(f"- PB数据月数: {s.get('pb_months', 0)}")
            parts.append(f"- 估值: 当前PB={val.get('current_pb', 'N/A')}, 上行空间={val.get('upside_to_peak', 'N/A')}%")
            corr = s.get("correlation", {})
            if corr:
                for fname, c in corr.items():
                    pearson = c.get("pearson")
                    lag = c.get("best_lag_months", 0)
                    pearson_str = f"{pearson:.3f}" if pearson is not None else "N/A"
                    parts.append(f"- 与{fname}: Pearson={pearson_str}, 滞后{lag}月")
            parts.append("")

    # 4. 归档数据摘要
    if "archive_path" in sector_data:
        parts.append("## 归档研究数据")
        parts.append(f"- 路径: {sector_data['archive_path']}")
        parts.append(f"- 文件数: {sector_data.get('archive_file_count', 0)}")
        parts.append("*（包含 LLM 数据源指导、Tavily 搜索结果、Jina 网页抓取、LLM 原始回复）*\n")

    parts.append("---")
    parts.append("请对以上分析进行全面审计，识别风险并给出结构化 JSON 报告。")
    return parts


# 审计类型 → 用户内容构建函数映射
_AUDIT_CONTENT_BUILDERS = {
    "factors": _build_factors_user_content,
    "cycle": _build_cycle_user_content,
    "stock": _build_stock_user_content,
    "full": _build_full_user_content,
}


def build_audit_prompt(
    sector: str, sector_data: dict, audit_type: str = "full",
) -> list[dict]:
    """构建审计提示词

    Args:
        sector: 板块名
        sector_data: 完整板块数据（因子+周期+个股+归档）
        audit_type: 审计类型 ("factors" / "cycle" / "stock" / "full")
    """
    if audit_type not in _AUDIT_SYSTEM_MAP:
        raise ValueError(
            f"Unknown audit_type: {audit_type!r}. Valid: {set(_AUDIT_SYSTEM_MAP)}"
        )
    system_prompt = _AUDIT_SYSTEM_MAP[audit_type]
    system_content = system_prompt.format(sector=sector)

    builder = _AUDIT_CONTENT_BUILDERS[audit_type]
    user_parts = builder(sector, sector_data)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_audit_chat_messages(
    sector: str,
    audit_data: dict,
    history: list[dict],
    user_msg: str,
) -> list[dict]:
    """构建审计对话 messages"""
    report = audit_data.get("report") or {}

    risk = report.get("risk_level", "unknown")
    confidence = report.get("confidence_score", 0)

    key_findings = []
    for item in report.get("audit_items", [])[:3]:
        key_findings.append(f"{item.get('category', 'N/A')}: {item.get('finding', 'N/A')[:50]}")
    key_findings_str = " | ".join(key_findings) if key_findings else "无"

    system_content = AUDIT_CHAT_SYSTEM.format(
        sector=sector,
        risk_level=risk,
        confidence=confidence,
        key_findings=key_findings_str,
    )

    messages = [{"role": "system", "content": system_content}]
    recent = history[-40:] if len(history) > 40 else history
    messages.extend(recent)
    messages.append({"role": "user", "content": user_msg})
    return messages


# ==================== 审计参数优化 ====================

AUDIT_PARAM_SYSTEM = """\
你是量化分析参数优化器。根据审计意见，生成个股重分析的参数调整建议。

当前默认参数：
- 权重: upside={upside}, alignment={alignment}, valuation={valuation}, momentum={momentum}
- 分析个股数: {top_n}
- 板块: {sector}，周期位置: {cycle_position}

请输出 JSON：
```json
{{
  "weights": {{"upside": 0.30, "alignment": 0.40, "valuation": 0.20, "momentum": 0.10}},
  "excluded_codes": ["002493"],
  "excluded_reasons": {{"002493": "荣盛石化主营炼化非石油开采"}},
  "top_n": 15,
  "notes": "提高周期对齐权重，排除非上游开采企业，扩大样本"
}}
```
规则：
1. weights 四项必须加起来等于 1.0
2. excluded_codes 只排除审计明确指出不属于该行业的股票
3. top_n 范围 5-20
4. 如果审计没有提到某个参数，保持默认值不变
5. excluded_reasons 为每个排除的股票代码给出一句话理由"""


def build_audit_param_prompt(
    sector: str,
    cycle_position: str,
    current_weights: dict,
    top_n: int,
    audit_text: str,
) -> list[dict]:
    """构建审计参数优化的 messages"""
    system_content = AUDIT_PARAM_SYSTEM.format(
        sector=sector,
        cycle_position=cycle_position,
        upside=current_weights.get("upside", 0.40),
        alignment=current_weights.get("alignment", 0.30),
        valuation=current_weights.get("valuation", 0.15),
        momentum=current_weights.get("momentum", 0.15),
        top_n=top_n,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"以下是审计意见，请生成参数调整建议：\n\n{audit_text}"},
    ]