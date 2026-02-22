"""周期底部龙头策略 - 常量定义"""

# 行业映射（3行业）
INDUSTRY_MAP = {
    "lithium": {
        "name": "锂盐（碳酸锂）",
        "commodity": "碳酸锂",
        "commodity_key": "lithium_carbonate",
    },
    "phosphorus": {
        "name": "磷化工",
        "commodity": "磷铵",
        "commodity_key": "phosphate",
    },
    "basic_chem": {
        "name": "基础化工",
        "commodity": "MDI",
        "commodity_key": "mdi",
    },
}

# 行业中文名映射
IND_CN = {
    "lithium": "锂盐",
    "phosphorus": "磷化工",
    "basic_chem": "基础化工",
}

# 财务指标列名映射（AKShare 返回的中文列名 -> 标准英文名）
FINANCIAL_COLUMNS = {
    "净资产收益率(%)": "roe",
    "营业总收入同比增长(%)": "revenue_growth",
    "销售毛利率(%)": "gross_margin",
    "每股经营现金流量(元)": "cashflow_per_share",
    "每股收益(元)": "eps",
    "净利润(元)": "net_profit",
    "每股净资产(元)": "bvps",
}

# 个股详细信息（龙头定性分析，用于个股档案页面）
STOCK_PROFILES = {
    "002466": {
        "name": "天齐锂业",
        "industry": "lithium",
        "core_advantage": "全球锂资源龙头，控股澳洲格林布什矿（全球最大硬岩锂矿），参股SQM",
        "cost_advantage": "锂精矿自给率高，成本优势显著",
        "risks": ["高负债率（收购SQM股权）", "碳酸锂价格周期波动大", "海外资产政策风险"],
    },
    "002460": {
        "name": "赣锋锂业",
        "industry": "lithium",
        "core_advantage": "锂产业链一体化龙头，涵盖上游锂矿-碳酸锂/氢氧化锂-锂电池回收",
        "cost_advantage": "多元化锂资源布局（锂辉石+盐湖+锂云母），供应稳定",
        "risks": ["锂价下行周期利润承压", "海外矿产开发不确定性", "技术路线变化风险"],
    },
    "000792": {
        "name": "盐湖股份",
        "industry": "lithium",
        "core_advantage": "国内盐湖提锂龙头，察尔汗盐湖资源禀赋优异",
        "cost_advantage": "盐湖提锂成本远低于矿石提锂，成本优势显著",
        "risks": ["提锂技术突破进度不确定", "钾肥业务受农业周期影响", "环保政策趋严"],
    },
    "600096": {
        "name": "云天化",
        "industry": "phosphorus",
        "core_advantage": "磷化工行业龙头，磷矿资源储量大，磷铵产能全国前列",
        "cost_advantage": "自有磷矿，矿化一体化运营，成本控制能力强",
        "risks": ["磷铵价格受农业周期影响", "环保投入持续增加", "新能源材料转型尚在早期"],
    },
    "600309": {
        "name": "万华化学",
        "industry": "basic_chem",
        "core_advantage": "全球MDI龙头，技术壁垒极高（全球仅6家企业掌握MDI技术）",
        "cost_advantage": "规模效应+技术领先，MDI单位成本全球最低",
        "risks": ["MDI价格周期波动", "石化业务受油价影响", "海外产能扩张风险"],
    },
    "600426": {
        "name": "华鲁恒升",
        "industry": "basic_chem",
        "core_advantage": "煤化工龙头，一头多线柔性联产，产品结构灵活",
        "cost_advantage": "煤气化技术领先，单位能耗低，成本优势明显",
        "risks": ["煤炭价格波动影响成本", "化工品价格周期波动", "环保限产政策"],
    },
}
