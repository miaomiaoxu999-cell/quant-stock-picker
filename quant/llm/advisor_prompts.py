"""投资顾问 — LLM Prompt 模板与 message builder"""

from __future__ import annotations

import json

# ==================== Prompt A: AI 诊断 ====================

ADVISOR_DIAGNOSIS_SYSTEM = """\
你是一位资深的 A 股投资顾问，擅长周期股分析。用户会提供：
- 总投资资金
- 看好的板块和个股列表（含估值/周期数据）
- 现有持仓

你的职责：
1. 逐一评价用户看好的每只股票（结合 PB 估值、周期位置、行业前景）
2. 指出不合理的选择（如高估、周期顶部、行业风险），给出替代建议
3. 补充推荐用户未选但值得关注的股票（从已分析股票中）
4. 对每只股票标注：推荐级别(strong_buy/buy/hold/reduce/sell)、核心理由、主要风险

输出要求：
- 先输出详细的文字分析（中文），对每只股票逐一点评
- 最后输出一个 JSON 块，格式如下（用 ```json 包裹）：

```json
{
  "recommended_stocks": [
    {
      "code": "002466",
      "name": "天齐锂业",
      "rating": "strong_buy",
      "reason": "一句话核心理由",
      "risk": "一句话主要风险"
    }
  ]
}
```
"""


def build_diagnosis_messages(
    total_capital: float,
    bullish_sectors: list[str],
    favored_stocks: list[dict],
    portfolio: dict,
    stock_data: list[dict],
    cycle_data: dict | None = None,
    audit_data: dict | None = None,
) -> list[dict]:
    """构建 AI 诊断的 messages

    Args:
        total_capital: 总资金
        bullish_sectors: 看好板块列表
        favored_stocks: 用户选中的股票 [{code, name, pb, price, ...}]
        portfolio: 现有持仓 {code: {avg_cost, shares, weight, ...}}
        stock_data: 所有可选股票数据
        cycle_data: 周期分析数据（可选）
        audit_data: 审计结果（可选）
    """
    parts = [f"## 投资者信息\n- 总资金: {total_capital:,.0f} 元"]
    if bullish_sectors:
        parts.append(f"- 看好板块: {', '.join(bullish_sectors)}")

    parts.append("\n## 用户看好的股票")
    for s in favored_stocks:
        pb_str = f"PB={s.get('pb', 'N/A')}" if s.get('pb') else "PB=N/A"
        price_str = f"价格={s.get('price', 'N/A')}" if s.get('price') else ""
        status = s.get('valuation_status', '')
        cycle = s.get('cycle_position', '')
        score = s.get('total_score', '')
        parts.append(
            f"- {s.get('name', '')}({s.get('code', '')}) | {price_str} | {pb_str} | "
            f"估值={status} | 周期={cycle} | 总分={score}"
        )

    if portfolio:
        parts.append("\n## 现有持仓")
        for code, h in portfolio.items():
            parts.append(
                f"- {code}: 均价={h.get('avg_cost', 0):.2f}, "
                f"数量={h.get('shares', 0)}股, "
                f"仓位={h.get('weight', 0):.1%}"
            )

    if cycle_data:
        parts.append("\n## 周期分析摘要")
        for sector, data in cycle_data.items():
            overall = data.get("overall", {})
            parts.append(
                f"- {sector}: {overall.get('cycle_position', 'N/A')}, "
                f"反转概率 {overall.get('reversal_probability', 'N/A')}%"
            )

    parts.append("\n## 所有可选股票")
    for s in stock_data:
        pb_str = f"PB={s.get('pb', 'N/A')}"
        parts.append(
            f"- {s.get('name', '')}({s.get('code', '')}) | {pb_str} | "
            f"估值={s.get('valuation_status', 'N/A')} | "
            f"周期={s.get('cycle_position', 'N/A')} | "
            f"总分={s.get('total_score', 'N/A')}"
        )

    return [
        {"role": "system", "content": ADVISOR_DIAGNOSIS_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ==================== Prompt B: 仓位配置 ====================

ADVISOR_ALLOCATION_SYSTEM = """\
你是一位资深的 A 股仓位管理专家。根据用户确认的股票清单和资金情况，给出仓位配置建议。

输入信息：
- 总投资资金
- 确认的股票清单（含估值/周期数据）
- 现有持仓详情

你的职责：
1. 基于分散化原则，为每只股票建议合理的仓位比例和金额
2. 对已有持仓的股票，明确说明操作：加仓/减仓/持有/卖出，以及具体数量
3. 建议保留适当现金比例（不设硬性限制，根据市场情况灵活建议）
4. 解释每只股票配置比例的逻辑

输出要求：
- 先输出详细的文字分析（中文），解释配置逻辑
- 最后输出一个 JSON 块（用 ```json 包裹）：

```json
{
  "allocations": [
    {
      "code": "002466",
      "name": "天齐锂业",
      "ratio": 0.25,
      "amount": 125000,
      "action": "buy",
      "shares": 2400,
      "price_range": "50-53",
      "reason": "配置理由"
    }
  ],
  "cash_reserve": {
    "ratio": 0.10,
    "amount": 50000,
    "reason": "保留理由"
  }
}
```

action 取值: buy(新买入) / add(加仓) / hold(持有) / reduce(减仓) / sell(卖出)
shares 为建议操作的股数（买入/加仓为正数，减仓为负数）
price_range 为建议的买入/操作价格区间
"""


def build_allocation_messages(
    total_capital: float,
    confirmed_stocks: list[dict],
    portfolio: dict,
    history: list[dict] | None = None,
) -> list[dict]:
    """构建仓位配置的 messages"""
    parts = [f"## 投资配置请求\n- 总资金: {total_capital:,.0f} 元"]

    parts.append("\n## 确认的股票清单")
    for s in confirmed_stocks:
        pb_str = f"PB={s.get('pb', 'N/A')}"
        price_str = f"价格={s.get('price', 'N/A')}"
        parts.append(
            f"- {s.get('name', '')}({s.get('code', '')}) | {price_str} | {pb_str} | "
            f"估值={s.get('valuation_status', 'N/A')} | "
            f"周期={s.get('cycle_position', 'N/A')}"
        )

    if portfolio:
        parts.append("\n## 现有持仓")
        for code, h in portfolio.items():
            parts.append(
                f"- {code}: 均价={h.get('avg_cost', 0):.2f}, "
                f"数量={h.get('shares', 0)}股, "
                f"仓位={h.get('weight', 0):.1%}"
            )

    msgs = [
        {"role": "system", "content": ADVISOR_ALLOCATION_SYSTEM},
    ]
    if history:
        msgs.extend(history[-20:])
    msgs.append({"role": "user", "content": "\n".join(parts)})
    return msgs


# ==================== Prompt C: 收益预测 ====================

ADVISOR_PREDICTION_SYSTEM = """\
你是一位量化分析师，擅长投资收益预测。根据用户的最终配置，预测不同时间和情景下的收益。

预测维度：
- 3 个时间段: 3个月、6个月、12个月
- 3 种情景: 乐观、基准、悲观

费用扣除：
- 印花税: 0.05%（仅卖出时）
- 佣金: 万2.5（买入卖出双向）
- 过户费: 0.001%（双向）

额外考虑：
- 股息分红（基于个股历史分红率估算）
- 各情景下的假设条件需明确说明

输出要求：
- 先输出详细的文字分析（中文），说明各情景假设和逻辑
- 最后输出一个 JSON 块（用 ```json 包裹）：

```json
{
  "predictions": {
    "3m": {
      "optimistic": {"return_rate": 0.15, "total_value": 575000, "dividend": 0, "cost": 250},
      "baseline": {"return_rate": 0.05, "total_value": 525000, "dividend": 0, "cost": 250},
      "pessimistic": {"return_rate": -0.08, "total_value": 460000, "dividend": 0, "cost": 250}
    },
    "6m": { ... },
    "12m": { ... }
  },
  "assumptions": {
    "optimistic": "乐观情景假设说明",
    "baseline": "基准情景假设说明",
    "pessimistic": "悲观情景假设说明"
  }
}
```
"""


def build_prediction_messages(
    total_capital: float,
    allocation: list[dict],
    cycle_data: dict | None = None,
) -> list[dict]:
    """构建收益预测的 messages"""
    parts = [f"## 投资收益预测请求\n- 总投入资金: {total_capital:,.0f} 元"]

    parts.append("\n## 最终配置")
    for a in allocation:
        parts.append(
            f"- {a.get('name', '')}({a.get('code', '')}) | "
            f"比例={a.get('ratio', 0):.0%} | 金额={a.get('amount', 0):,.0f} | "
            f"操作={a.get('action', '')} | "
            f"价格区间={a.get('price_range', 'N/A')}"
        )

    if cycle_data:
        parts.append("\n## 周期分析参考")
        for sector, data in cycle_data.items():
            overall = data.get("overall", {})
            parts.append(
                f"- {sector}: {overall.get('cycle_position', 'N/A')}, "
                f"反转概率 {overall.get('reversal_probability', 'N/A')}%"
            )

    return [
        {"role": "system", "content": ADVISOR_PREDICTION_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ==================== Prompt D: 风险监控计划 ====================

ADVISOR_RISK_PLAN_SYSTEM = """\
你是一位风险管理专家，为投资组合制定系统性风险监控计划。

基于用户的投资配置，输出三层风险监控：

1. **个股风险**: 每只股票 2-3 个核心风险
2. **板块风险**: 涉及板块的系统性风险
3. **宏观风险**: 影响整个组合的宏观因素

每个风险项包含：
- signal: 风险信号描述
- frequency: 监控频率（日/周/月）
- threshold: 触发阈值
- action: 触发后的应对动作
- source: 数据来源/查看渠道

输出要求：
- 先输出详细的文字分析（中文）
- 最后输出一个 JSON 块（用 ```json 包裹）：

```json
{
  "stock_risks": [
    {
      "code": "002466",
      "name": "天齐锂业",
      "risks": [
        {
          "signal": "碳酸锂价格跌破成本线",
          "frequency": "周",
          "threshold": "价格<7万元/吨",
          "action": "减仓50%",
          "source": "SMM/百川盈孚"
        }
      ]
    }
  ],
  "sector_risks": [
    {
      "sector": "锂电池/锂盐",
      "signal": "新能源车销量增速低于预期",
      "frequency": "月",
      "threshold": "同比增速<10%",
      "action": "降低板块整体仓位",
      "source": "中汽协月度数据"
    }
  ],
  "macro_risks": [
    {
      "signal": "美联储加息超预期",
      "frequency": "月",
      "threshold": "加息>50bp",
      "action": "增加现金比例至30%",
      "source": "美联储议息会议纪要"
    }
  ]
}
```
"""


def build_risk_plan_messages(
    allocation: list[dict],
    cycle_data: dict | None = None,
) -> list[dict]:
    """构建风险监控计划的 messages"""
    parts = ["## 风险监控计划请求\n"]

    parts.append("## 投资配置")
    for a in allocation:
        parts.append(
            f"- {a.get('name', '')}({a.get('code', '')}) | "
            f"比例={a.get('ratio', 0):.0%} | 金额={a.get('amount', 0):,.0f} | "
            f"操作={a.get('action', '')}"
        )

    if cycle_data:
        parts.append("\n## 周期分析参考")
        for sector, data in cycle_data.items():
            overall = data.get("overall", {})
            parts.append(
                f"- {sector}: {overall.get('cycle_position', 'N/A')}, "
                f"关键信号: {', '.join(overall.get('key_signals', [])[:3])}"
            )

    return [
        {"role": "system", "content": ADVISOR_RISK_PLAN_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]
