# -*- coding: utf-8 -*-
"""
data.py — 解析《机构行为研究 轻量公式版本.xlsx》，生成统一 8 家机构口径的日度长表 parquet。

【净买入口径（2026-11-21 定）】
  2023-01-03 起 : 净买入 = 买入 - 卖出（日度买入/日度卖出表；机构标签经交叉验证正确）
  2021-01-04 起 : 日度详细 新口径面板（行1255起）的净买入，表头错位修正后使用

【日度详细新口径面板表头错位修正】（2026-11-21 经泰康源文件 panel0/panel1 交叉验证确认）
  该面板的数据实际是 8 家新口径机构块，但表头贴的是 11 家旧口径模板，从第 2 块起整体错位。
  表头 -> 真实机构（依据：panel1 vs panel0 同文件 corr 1.0000/0.9989；与买入/卖出表交叉验证）：
    大型商业银行/政策性银行 -> 大型银行        (= 旧口径 大行+股份制, corr 1.0000)
    股份制商业银行          -> 中小型银行      (= 旧口径 城商+农商+外资, 经2023源数据验证 corr 1.0000)
    城市商业银行            -> 证券公司
    农村金融机构            -> 保险公司
    外资银行                -> 基金公司及产品  （仅指表头块；旧口径真实"外资银行"已并入新口径中小型银行）
    证券公司                -> 货币市场基金
    保险公司                -> 理财子公司及理财类产品
    基金公司及产品          -> 其他
    货币市场基金/理财子公司及理财类产品/其他产品类（表头） -> 整列为空，丢弃
  => 修正后 2021-01-04 起 8 家机构全部有数据，无缺失。

【已知瑕疵】
  净买入(日度详细) 与 买卖差(买入-卖出) 两个口径本身有差异（corr 0.71~0.99，因机构而异），
  2023-01-03 接缝处可能存在小幅口径断点；接缝前后各 20 个交易日的滚动窗口因子需谨慎。

宽表结构说明：
  日度详细: 行1机构/行2券种/行3久期权重(=期限桶中值,不输出)/行4期限桶, 行5起数据, A列日期(2021-2023为文本)
  日度买入/卖出: 行2机构/行3券种/行4平均久期/行5期限桶, 行6起数据, A列日期(datetime)

输出（唯一文件）：量化机构行为研究/data/daily_detail.parquet
  日期 / 机构 / 券种 / 期限桶标签 / 净买入（亿元） / 买入（亿元） / 卖出（亿元）
  买入/卖出仅 2023 年起有值；2023 年前 净买入 来自日度详细（表头已修正）。

用法：python data.py [可选: Excel路径]
"""
import sys
import io
import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- 可改参数 ----------
PROJECT = Path(__file__).resolve().parents[1]          # 量化机构行为研究/
ROOT = PROJECT.parent                                   # 研报清单目录
SRC_DEFAULT = ROOT / "机构行为研究 轻量公式版本.xlsx"
OUT_FILE = PROJECT / "data" / "daily_detail.parquet"
SHEET_NET = "日度详细"
SHEET_BUY = "日度买入"
SHEET_SELL = "日度卖出"
# -----------------------------

TENOR_ORDER = ["≦1Y", "1-3Y", "3-5Y", "5-7Y", "7-10Y", "10-15Y", "15-20Y", "20-30Y", ">30Y", "合计"]

# 日度详细新口径面板: 表头(旧11家模板) -> 真实机构(新8家数据块)；None = 空块丢弃
# 依据: 泰康源文件 panel1 vs panel0 交叉验证 + 与买入/卖出表交叉验证 (2026-11-21)
TRUE_INST_MAP = {
    "大型商业银行/政策性银行": "大型银行",
    "股份制商业银行": "中小型银行",
    "城市商业银行": "证券公司",
    "农村金融机构": "保险公司",
    "外资银行": "基金公司及产品",
    "证券公司": "货币市场基金",
    "保险公司": "理财子公司及理财类产品",
    "基金公司及产品": "其他",
    "货币市场基金": None,                   # 空块
    "理财子公司及理财类产品": None,          # 空块
    "其他产品类": None,                     # 空块
}

KEY = ["日期", "机构", "券种", "期限桶标签"]


def parse_date(v):
    """兼容 datetime 与 'YYYY/MM/DD' 等文本日期；失败返回 None。"""
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    return None


def ffill(seq):
    out, last = [], None
    for v in seq:
        if v is not None:
            last = v
        out.append(last)
    return out


def wide_to_long(all_rows, hdr_rows, data_start, value_name):
    """把 (机构, 券种, 期限桶) 三级表头的宽表转成长表。
    hdr_rows: (机构行号, 券种行号, 期限桶行号) 0-based；data_start: 首个数据行 0-based。
    只保留日期单调递增的最后一个面板（新口径）。
    """
    h_inst = ffill(all_rows[hdr_rows[0]])
    h_bond = ffill(all_rows[hdr_rows[1]])
    h_tenor = list(all_rows[hdr_rows[2]])
    col_meta = []
    for ci in range(1, len(h_tenor)):
        if h_tenor[ci] is None:
            continue
        col_meta.append({"ci": ci, "机构": h_inst[ci], "券种": h_bond[ci], "期限桶标签": str(h_tenor[ci])})
    meta = pd.DataFrame(col_meta)

    wide = pd.DataFrame(all_rows[data_start:])
    dates = wide[0].map(parse_date)
    wide = wide.loc[dates.notna()].reset_index(drop=True)
    dts = pd.Series(pd.to_datetime(dates[dates.notna()].values))
    panel = (dts.diff().dt.days < 0).cumsum()
    keep = panel == panel.max()                        # 新口径 = 最后一个面板
    wide, dts = wide.loc[keep].reset_index(drop=True), dts.loc[keep].reset_index(drop=True)

    vals = wide[meta["ci"].tolist()].apply(pd.to_numeric, errors="coerce")
    vals.columns = range(len(meta))
    vals.insert(0, "日期", dts)
    long = vals.melt(id_vars="日期", var_name="col", value_name=value_name)
    long = long.merge(meta, left_on="col", right_index=True, how="left", sort=False)
    return long.drop(columns="col")


def main(src_path: Path = SRC_DEFAULT) -> Path:
    print(f"[1/5] 读取工作簿三个 sheet ...（约 2-3 分钟）")
    wb = load_workbook(src_path, read_only=True, data_only=True)
    sheets = {}
    for name in (SHEET_NET, SHEET_BUY, SHEET_SELL):
        sheets[name] = list(wb[name].iter_rows(values_only=True))
        print(f"      {name}: {len(sheets[name])} 行 x {len(sheets[name][0])} 列")
    wb.close()

    print("[2/5] 日度买入/卖出 -> 长表 ...")
    buy = wide_to_long(sheets[SHEET_BUY], hdr_rows=(1, 2, 4), data_start=5, value_name="买入（亿元）")
    sell = wide_to_long(sheets[SHEET_SELL], hdr_rows=(1, 2, 4), data_start=5, value_name="卖出（亿元）")
    gross = buy.merge(sell, on=KEY, how="outer")
    cutoff = gross["日期"].min()
    print(f"      买入/卖出覆盖 {cutoff.date()} 起，{len(gross):,} 行")

    print(f"[3/5] 日度详细 -> 长表，修正错位表头，仅保留 {cutoff.date()} 之前的净买入 ...")
    net = wide_to_long(sheets[SHEET_NET], hdr_rows=(0, 1, 3), data_start=4, value_name="净买入（亿元）")
    net["机构真实"] = net["机构"].map(TRUE_INST_MAP)
    dropped = net[net["机构真实"].isna()].groupby("机构")["净买入（亿元）"].apply(lambda x: x.abs().sum())
    print(f"      空块丢弃核验（|值|总和应≈0）: {{ {', '.join(f'{k}: {v:.4f}' for k, v in dropped.items())} }}")
    net = net.dropna(subset=["机构真实"]).drop(columns="机构").rename(columns={"机构真实": "机构"})
    net = net[net["日期"] < cutoff]
    net = net.groupby(KEY, observed=True)["净买入（亿元）"].sum().reset_index()
    print(f"      2023年前净买入长表 {len(net):,} 行，机构: {sorted(net['机构'].unique())}")

    print("[4/5] 2023 起净买入 = 买入-卖出，拼接全历史 ...")
    post = gross.copy()
    post["净买入（亿元）"] = post["买入（亿元）"] - post["卖出（亿元）"]
    df = pd.concat([net, post[KEY + ["净买入（亿元）", "买入（亿元）", "卖出（亿元）"]]],
                   ignore_index=True)
    df["期限桶标签"] = pd.Categorical(df["期限桶标签"], categories=TENOR_ORDER, ordered=True)
    for c in ("机构", "券种"):
        df[c] = df[c].astype("category")
    df = df.sort_values(KEY, kind="stable").reset_index(drop=True)
    df = df[KEY + ["净买入（亿元）", "买入（亿元）", "卖出（亿元）"]]

    print("[5/5] 自检并写出 parquet ...")
    dup = df.duplicated(subset=KEY).sum()
    assert dup == 0, f"存在 {dup} 行重复键，需检查！"
    avail = df.groupby("机构", observed=True).agg(
        首日=("日期", "min"), 末日=("日期", "max"),
        净买入非空天数=("净买入（亿元）", lambda x: x.notna().sum()),
        买入非空天数=("买入（亿元）", lambda x: x.notna().sum()))
    print(avail.to_string())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    print(f"\n      {len(df):,} 行 -> {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
    print("DONE")
    return OUT_FILE


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_DEFAULT
    main(src)
