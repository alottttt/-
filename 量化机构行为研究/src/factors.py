# -*- coding: utf-8 -*-
"""
factors.py — 因子层：四族因子，输出 factor_values.parquet（长表：日期/因子名/值）。

设计（2026-11-22 与用户定稿）：
  - 一个名单 SPECS = 每个因子的 {名称, 族, 公式参数, Excel 对账位置}；
  - 新增因子只需往 SPECS 加一行，不动 compute_all()；
  - 所有因子值存一张长表，新增因子=追加行。

四族（已逐因子与 Excel 粘贴值对账验证）：
  1. 买债强度   : 某机构某券种(可选期限桶)净买入，5d/20d 滚动和 或 直接提取
  2. 分券种     : (证券+基金)净买入 - (大行+保险)净买入，11 个日度券种/期限口径
  3. 机构分歧度 : (买入-卖出)/(买入+卖出)，剔除低覆盖并与识别族去重后保存 24 个，2023 起
  4. 买入加速度 : 大行买短卖长 = 净买入(短端) - 净买入(长端)，仅 2 个（其余列 Excel 为空，暂不实现）

用法：python factors.py
"""
import sys
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.utils import column_index_from_string, get_column_letter

# ---------- 路径 ----------
PROJECT = Path(__file__).resolve().parents[1]   # 量化机构行为研究/
DATA = PROJECT / "data"
DETAIL = DATA / "daily_detail.parquet"
OUT = DATA / "factor_values.parquet"

# ---------- 口径映射（与 data.py 产出、Excel 因子口径一致） ----------
BONDS = {
    "同业存单": ["同业存单"],
    "地方政府债": ["地方政府债"],
    "国债": ["国债-新债", "国债-老债"],
    "政金债": ["政策性金融债-新债", "政策性金融债-老债"],
    "中票": ["中期票据"],
    "利率债合计": ["国债-新债", "国债-老债", "政策性金融债-新债", "政策性金融债-老债", "地方政府债"],
    "信用债合计": ["中期票据", "企业债", "短期/超短期融资券", "资产支持证券"],
}
TENOR = {
    "≦1Y": ["≦1Y"], "1-3Y": ["1-3Y"], "3-5Y": ["3-5Y"], "5-7Y": ["5-7Y"], "7-10Y": ["7-10Y"],
    "10年以上": ["10-15Y", "15-20Y", "20-30Y", ">30Y"],
    "20-30Y": ["20-30Y"],
    "<3": ["≦1Y", "1-3Y"], "≤3Y": ["≦1Y", "1-3Y"],
    "10+": ["10-15Y", "15-20Y", "20-30Y", ">30Y"],
}
INST = {
    "基金": "基金公司及产品", "保险": "保险公司", "大行": "大型银行",
    "中小行": "中小型银行", "证券": "证券公司", "理财子": "理财子公司及理财类产品",
    "券商": "证券公司",
}
# 11 个券种/期限口径（分券种、分歧度共用）
TYPES = [
    ("1年及以下国债", "国债", "≦1Y"), ("1-3年国债", "国债", "1-3Y"),
    ("3-5年国债", "国债", "3-5Y"), ("5-7年国债", "国债", "5-7Y"),
    ("7-10年国债", "国债", "7-10Y"), ("10年以上国债", "国债", "10年以上"),
    ("政金债合计", "政金债", None), ("地方政府债", "地方政府债", None),
    ("同业存单", "同业存单", None), ("中票", "中票", None),
    ("利率债合计", "利率债合计", None),
]

# 识别信号族：37 个机构/期限组合 × 8 形态，剔除低覆盖和高零值因子后为 269 个。
RECOG_BONDS = ["国债-新债", "国债-老债"]
RECOG_BUCKETS = ["≦1Y", "1-3Y", "3-5Y", "5-7Y", "7-10Y", "20-30Y"]
RECOG_FORMS = ["净买入", "净买入占比", "成交量", "净买入zscore252", "MA10偏离", "买入放量", "卖出放量", "VR放量"]
# 2026-11-24：用户指定删除“其他”机构，货币市场基金只保留一年及以下。
# 这是因子候选范围；源明细继续保留完整机构与期限数据。
RECOG_INST_BUCKETS = {
    inst: tuple(RECOG_BUCKETS) for inst in (
        "大型银行", "中小型银行", "证券公司", "保险公司",
        "基金公司及产品", "理财子公司及理财类产品",
    )
}
RECOG_INST_BUCKETS["货币市场基金"] = ("≦1Y",)
# 2026-11-24 固定撤出名单：统计区间为 2023-01-03 至 2026-11-11。
# 先撤出覆盖率不足 80% 的占比，再按用户要求撤出有限观测中零值率 >30% 的因子。
# 不按运行时的全样本统计自动选因子；来源明细与保留形态的计算不受影响。
EXCLUDED_FACTORS = frozenset({
    # 低覆盖：六个识别占比 + 四个同口径基础分歧度。
    "识别_保险公司_≦1Y_净买入占比",
    "识别_保险公司_1-3Y_净买入占比",
    "识别_理财子公司及理财类产品_≦1Y_净买入占比",
    "识别_理财子公司及理财类产品_1-3Y_净买入占比",
    "识别_理财子公司及理财类产品_3-5Y_净买入占比",
    "识别_理财子公司及理财类产品_5-7Y_净买入占比",
    "分歧度_理财子_1年及以下国债",
    "分歧度_理财子_1-3年国债",
    "分歧度_理财子_3-5年国债",
    "分歧度_理财子_5-7年国债",
    # 高零值：二十一个识别因子 + 一个基础净买入。
    "保险_国债≦1Y",
    "识别_保险公司_≦1Y_净买入",
    "识别_保险公司_≦1Y_成交量",
    "识别_保险公司_≦1Y_买入放量",
    "识别_保险公司_≦1Y_卖出放量",
    "识别_保险公司_≦1Y_VR放量",
    "识别_保险公司_1-3Y_买入放量",
    "识别_保险公司_1-3Y_卖出放量",
    "识别_理财子公司及理财类产品_≦1Y_净买入",
    "识别_理财子公司及理财类产品_≦1Y_成交量",
    "识别_理财子公司及理财类产品_≦1Y_买入放量",
    "识别_理财子公司及理财类产品_≦1Y_卖出放量",
    "识别_理财子公司及理财类产品_≦1Y_VR放量",
    "识别_理财子公司及理财类产品_1-3Y_净买入",
    "识别_理财子公司及理财类产品_1-3Y_成交量",
    "识别_理财子公司及理财类产品_1-3Y_买入放量",
    "识别_理财子公司及理财类产品_1-3Y_卖出放量",
    "识别_理财子公司及理财类产品_1-3Y_VR放量",
    "识别_理财子公司及理财类产品_3-5Y_买入放量",
    "识别_理财子公司及理财类产品_3-5Y_卖出放量",
    "识别_理财子公司及理财类产品_5-7Y_买入放量",
    "识别_理财子公司及理财类产品_5-7Y_卖出放量",
})
# 2026-11-24：18 对全历史数值及缺失位置相同的序列只保存识别族名称。
# 别名用于追溯原基础名称，不再重复注册或写出；不把同源但不同公式的形态合并。
FACTOR_ALIASES = {
    "中小行_国债≦1Y": "识别_中小型银行_≦1Y_净买入",
    "证券_国债7-10Y": "识别_证券公司_7-10Y_净买入",
    "分歧度_理财子_7-10年国债": "识别_理财子公司及理财类产品_7-10Y_净买入占比",
}
for _short, _institution in [("中小行", INST["中小行"]), ("券商", INST["券商"]), ("基金", INST["基金"])]:
    for _label, _bond, _bucket in TYPES[:5]:
        FACTOR_ALIASES[f"分歧度_{_short}_{_label}"] = f"识别_{_institution}_{_bucket}_净买入占比"
# 含当日的尾随窗口，沿用已实现的预热门槛；不把本次修复变成信号定义变更。
ZSCORE_WINDOW, ZSCORE_MIN = 252, 60
DEVIATION_WINDOW, DEVIATION_MIN = 10, 5
VOLUME_WINDOW, VOLUME_MIN = 20, 10

_detail = None


def _load():
    global _detail
    if _detail is None:
        _detail = pd.read_parquet(DETAIL)
    return _detail


def _filter(insts, bond, tenor=None):
    if isinstance(insts, str):
        insts = [insts]
    d = _load()
    m = d[(d["机构"].isin(insts)) & (d["券种"].isin(BONDS[bond]))]
    if tenor is None:
        m = m[m["期限桶标签"] == "合计"]      # 券种总额取「合计」桶，不逐桶相加
    else:
        m = m[m["期限桶标签"].isin(TENOR[tenor])]
    return m


def net(insts, bond, tenor=None):
    """净买入（亿元）按日求和，返回以日期为索引的 Series。"""
    return _sum_complete(_filter(insts, bond, tenor), ["日期"],
                         ["净买入（亿元）"])["净买入（亿元）"]


def buy_sell(inst, bond, tenor=None):
    """返回 (买入, 卖出) 两个按日 Series（仅 2023 起有值）。"""
    g = _sum_complete(_filter(inst, bond, tenor), ["日期"],
                      ["买入（亿元）", "卖出（亿元）"])
    return g["买入（亿元）"], g["卖出（亿元）"]


def _sum_complete(frame, keys, columns):
    """保留真实零；源行全缺失或部分缺失时，不把不完整金额汇总为完整总额。"""
    data = frame[keys + columns].copy()
    data[columns] = data[columns].where(np.isfinite(data[columns]))
    grouped = data.groupby(keys, observed=True)[columns]
    total = grouped.sum(min_count=1)
    complete = grouped.count().eq(grouped.size(), axis=0)
    return total.where(complete)


def _roll(s, w):
    return s.rolling(w).sum() if w else s


def _col(base, offset):
    """Excel 列字母 + 偏移量（正确处理 AA 之后的跨列）。"""
    return get_column_letter(column_index_from_string(base) + offset)


# ---------- 因子注册表 SPECS ----------
# kind:
#   "net"             : 净买入（可选 rolling 窗口）
#   "trade_alloc"     : (证券+基金)净买入 - (大行+保险)净买入（日度）
#   "divergence"      : (买入-卖出)/(买入+卖出)
#   "short_minus_long": 净买入(短端) - 净买入(长端)
SPECS = []


def _add(name, family, kind, sheet, col, **params):
    if name in EXCLUDED_FACTORS or name in FACTOR_ALIASES:
        return
    SPECS.append({"name": name, "family": family, "kind": kind,
                  "sheet": sheet, "col": col, "params": params})


# --- 族1 买债强度（机构 × 券种/期限 × 窗口）---
_BQ = "买债强度因子"
for name, col, inst, bond, tenor, w in [
    ("基金_同业存单5d", "B", "基金", "同业存单", None, 5),
    ("基金_同业存单20d", "C", "基金", "同业存单", None, 20),
    ("基金_地方政府债5d", "D", "基金", "地方政府债", None, 5),
    ("基金_地方政府债20d", "E", "基金", "地方政府债", None, 20),
    ("基金_国债5d", "F", "基金", "国债", None, 5),
    ("基金_国债20d", "G", "基金", "国债", None, 20),
    ("基金_政金债1-3Y5d", "H", "基金", "政金债", "1-3Y", 5),
    ("基金_政金债3-5Y20d", "I", "基金", "政金债", "3-5Y", 20),
    ("基金_中票", "J", "基金", "中票", None, 0),
    ("基金_信用债20d", "K", "基金", "信用债合计", None, 20),
    ("保险_国债10年以上", "M", "保险", "国债", "10年以上", 0),
    ("保险_国债≦1Y", "N", "保险", "国债", "≦1Y", 0),
    ("保险_地方政府债20-30Y", "O", "保险", "地方政府债", "20-30Y", 0),
    ("保险_国债10年以上20d", "P", "保险", "国债", "10年以上", 20),
    ("大行_国债7-10Y20d", "R", "大行", "国债", "7-10Y", 20),
    ("中小行_国债5-7Y5d", "U", "中小行", "国债", "5-7Y", 5),
    ("中小行_国债5-7Y20d", "V", "中小行", "国债", "5-7Y", 20),
    ("中小行_国债7-10Y5d", "W", "中小行", "国债", "7-10Y", 5),
    ("中小行_国债7-10Y20d", "X", "中小行", "国债", "7-10Y", 20),
    ("中小行_政金债5-7Y", "Y", "中小行", "政金债", "5-7Y", 0),
    ("中小行_国债≦1Y", "Z", "中小行", "国债", "≦1Y", 0),
    ("证券_国债7-10Y", "AB", "证券", "国债", "7-10Y", 0),
    ("证券_政金债7-10Y", "AC", "证券", "政金债", "7-10Y", 0),
    ("证券_地方政府债", "AD", "证券", "地方政府债", None, 0),
    ("证券_同业存单", "AE", "证券", "同业存单", None, 0),
    ("证券_中票", "AF", "证券", "中票", None, 0),
    ("证券_利率债合计", "AG", "证券", "利率债合计", None, 0),
    ("理财子_同业存单", "AI", "理财子", "同业存单", None, 0),
    ("理财子_中票≤3Y", "AJ", "理财子", "中票", "≤3Y", 0),
]:
    _add(name, "买债强度", "net", _BQ, col, inst=INST[inst], bond=bond, tenor=tenor, window=w)

# --- 族2 分券种（交易盘-配置盘，11 个日度口径；周度已移除）---
_FQ = "分券种因子"
for i, (label, bond, tenor) in enumerate(TYPES):
    _add(f"交易盘-配置盘_{label}_日度", "分券种", "trade_alloc", _FQ,
         _col("V", i), bond=bond, tenor=tenor)

# --- 族3 机构分歧度（4 机构 × 11 口径，撤出 4 个低覆盖组合，2023 起）---
_FD = "机构分歧度行为"
for i, (label, bond, tenor) in enumerate(TYPES):
    _add(f"分歧度_基金_{label}", "机构分歧度", "divergence", _FD,
         _col("B", i), inst=INST["基金"], bond=bond, tenor=tenor)
for i, (label, bond, tenor) in enumerate(TYPES):
    _add(f"分歧度_中小行_{label}", "机构分歧度", "divergence", _FD,
         _col("N", i), inst=INST["中小行"], bond=bond, tenor=tenor)
for i, (label, bond, tenor) in enumerate(TYPES):
    _add(f"分歧度_理财子_{label}", "机构分歧度", "divergence", _FD,
         _col("Z", i), inst=INST["理财子"], bond=bond, tenor=tenor)
for i, (label, bond, tenor) in enumerate(TYPES):
    _add(f"分歧度_券商_{label}", "机构分歧度", "divergence", _FD,
         _col("AL", i), inst=INST["券商"], bond=bond, tenor=tenor)

# --- 族4 买入加速度（大行买短卖长，仅 Excel 有值的 2 个）---
_JS = "机构买入加速度"
_add("大行买短卖长_利率债_买<3卖7-10", "买入加速度", "short_minus_long", _JS, "B",
     inst=INST["大行"], bond="利率债合计", short="<3", long="7-10Y")
_add("大行买短卖长_国债_买<3卖10+", "买入加速度", "short_minus_long", _JS, "C",
     inst=INST["大行"], bond="国债", short="<3", long="10+")


def compute(spec):
    kind, p = spec["kind"], spec["params"]
    if kind == "net":
        s = net(p["inst"], p["bond"], p.get("tenor"))
        return _roll(s, p.get("window", 0))
    if kind == "trade_alloc":
        t = net([INST["证券"], INST["基金"]], p["bond"], p.get("tenor"))
        a = net([INST["大行"], INST["保险"]], p["bond"], p.get("tenor"))
        return t - a
    if kind == "divergence":
        b, s = buy_sell(p["inst"], p["bond"], p.get("tenor"))
        return _divide_positive(b - s, b + s)
    if kind == "short_minus_long":
        return net(p["inst"], p["bond"], p["short"]) - net(p["inst"], p["bond"], p["long"])
    raise ValueError(f"未知 kind: {kind}")


def compute_all() -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        s = compute(spec)
        s = s.dropna()
        rows.append(pd.DataFrame({"日期": s.index, "因子名": spec["name"], "值": s.values}))
    long = pd.concat(rows, ignore_index=True)
    long["日期"] = pd.to_datetime(long["日期"])
    return long


def _safe(s):
    """把 inf/-inf 替换为 NaN（除零保护）。"""
    return s.replace([np.inf, -np.inf], np.nan)


def _divide_positive(numerator, denominator):
    """分母必须为有限正数，零成交/缺失分母不产生虚假的比例。"""
    valid = np.isfinite(denominator) & denominator.gt(0)
    return _safe(numerator / denominator.where(valid))


def _rolling_standardize(s, window, minimum):
    """逐个原始窗口重算，避免递推方差把常数窗口变成微小正方差。

    按局部离差缩放后计算样本标准差，不设统一绝对阈值，
    从而保留真实的小幅变化。当前值缺失、预热不足或常数窗口返回 NaN。
    """
    s = _safe(s.astype(float))

    def last_zscore(values):
        if not np.isfinite(values[-1]):
            return np.nan
        valid = values[np.isfinite(values)]
        if len(valid) < minimum or valid.min() == valid.max():
            return np.nan
        centered = valid - valid[0]
        centered -= centered.mean()
        scale = np.max(np.abs(centered))
        if not np.isfinite(scale) or scale == 0:
            return np.nan
        scaled = centered / scale
        std = np.sqrt(np.dot(scaled, scaled) / (len(valid) - 1))
        return scaled[-1] / std if std > 0 else np.nan

    return _safe(s.rolling(window, min_periods=minimum).apply(last_zscore, raw=True))


def _volume_ratio(s):
    """MA20 至少 10 个真实观测；真实零参与均值，全零窗口不定义放量。"""
    s = _safe(s.astype(float))
    roll = s.rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN)
    # max==0 明确识别全零，避免均值运算的微小残留被当作有效分母。
    mean = roll.mean().where(roll.max().gt(0))
    return _divide_positive(s, mean)


def _recognition_forms(net, buy, sell):
    gross = buy + sell
    return {
        "净买入": net,
        "净买入占比": _divide_positive(buy - sell, gross),
        "成交量": gross,
        "净买入zscore252": _rolling_standardize(net, ZSCORE_WINDOW, ZSCORE_MIN),
        "MA10偏离": _rolling_standardize(net, DEVIATION_WINDOW, DEVIATION_MIN),
        "买入放量": _volume_ratio(buy),
        "卖出放量": _volume_ratio(sell),
        "VR放量": _volume_ratio(gross),
    }


def recognition_signals() -> pd.DataFrame:
    """按 RECOG_INST_BUCKETS 的机构/期限范围生成八种识别形态。"""
    d = _load()
    cal = pd.DatetimeIndex(sorted(d["日期"].unique()))
    g = d[d["券种"].isin(RECOG_BONDS) & d["期限桶标签"].isin(RECOG_BUCKETS)]
    g = _sum_complete(g, ["机构", "期限桶标签", "日期"],
                      ["净买入（亿元）", "买入（亿元）", "卖出（亿元）"]).reset_index()

    rows = []
    for (inst, bucket), grp in g.groupby(["机构", "期限桶标签"], observed=True):
        if bucket not in RECOG_INST_BUCKETS.get(inst, ()):
            continue
        grp = grp.set_index("日期").reindex(cal)
        net = grp["净买入（亿元）"].astype(float)
        buy = grp["买入（亿元）"].astype(float)
        sell = grp["卖出（亿元）"].astype(float)
        forms = _recognition_forms(net, buy, sell)
        for form in RECOG_FORMS:
            name = f"识别_{inst}_{bucket}_{form}"
            if name in EXCLUDED_FACTORS:
                continue
            s = forms[form].dropna()
            if len(s) == 0:
                continue
            rows.append(pd.DataFrame({"日期": s.index, "因子名": name, "值": s.values}))
    if not rows:
        return pd.DataFrame(columns=["日期", "因子名", "值"])
    return pd.concat(rows, ignore_index=True)


def main():
    long = compute_all()           # 现有四族
    rec = recognition_signals()    # 识别信号族（阶段A）
    long = pd.concat([long, rec], ignore_index=True)
    long["日期"] = pd.to_datetime(long["日期"])
    if long[["日期", "因子名"]].isna().any().any() or long.duplicated(["日期", "因子名"]).any():
        raise ValueError("因子主键缺失或重复，未写出结果")
    if not np.isfinite(long["值"]).all():
        raise ValueError("因子含非有限值，未写出结果")
    DATA.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标在同一目录；全部写完再替换，失败不破坏现有数据。
    with tempfile.NamedTemporaryFile(dir=DATA, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        long.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"写出 {OUT}：{len(long):,} 行，{long['因子名'].nunique()} 个因子")
    print(f"  四族 {len(SPECS)} 个 + 识别信号族 {rec['因子名'].nunique()} 个")
    print("DONE")
    return OUT


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
