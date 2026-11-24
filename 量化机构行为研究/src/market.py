# -*- coding: utf-8 -*-
"""
market.py — 生成因子评估用的行情基准表 market.parquet（只存水平值，变化率用时再算）。

数据来源（同一目录两个工作簿）：
  轻量版 Sheet1        : 1/3/5/7/10/30Y国债收益率 + 信用债总指数（历史最长，~2018 起）
  历史版 预测值2       : 10Y国债/30Y国债/5Y二级资本债(AAA-)/30Y地方债/7Y中票(AAA)/1Y国债（2024 起，iFinD 导出）

重叠列（1Y/10Y/30Y国债）做交叉验证，差异超阈值会提示。

输出（唯一文件）：量化机构行为研究/data/market.parquet
  日期 / 1Y国债(%) / 3Y国债(%) / 5Y国债(%) / 7Y国债(%) / 10Y国债(%) / 30Y国债(%) /
        信用债总指数 / 5Y二级资本债AAA-(%) / 30Y地方债(%) / 7Y中票AAA(%)

用法：python market.py
"""
import sys
import io
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- 可改参数 ----------
PROJECT = Path(__file__).resolve().parents[1]          # 量化机构行为研究/
ROOT = PROJECT.parent
SRC_LIGHT = ROOT / "机构行为研究 轻量公式版本.xlsx"      # Sheet1 利率曲线
SRC_HIST = ROOT / "历史版本" / "机构行为研究.xlsx"        # 预测值2 多资产
OUT_FILE = PROJECT / "data" / "market.parquet"
# -----------------------------

# Sheet1 列序（A=日期, B..G=收益率, H=信用债指数）
SHEET1_COLS = {1: "1Y国债(%)", 2: "3Y国债(%)", 3: "5Y国债(%)", 4: "7Y国债(%)",
               5: "10Y国债(%)", 6: "30Y国债(%)", 7: "信用债总指数"}
# 预测值2 列序（A=日期; 水平值在 B/D/F/H/J/L，变化率在 C/E/G/I/K/M 不取）
PRED2_COLS = {1: "10Y国债(%)", 3: "30Y国债(%)", 5: "5Y二级资本债AAA-(%)",
              7: "30Y地方债(%)", 9: "7Y中票AAA(%)", 11: "1Y国债(%)"}


def read_sheet1(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    dates, recs = [], {v: [] for v in SHEET1_COLS.values()}
    for row in ws.iter_rows(min_row=3, values_only=True):   # 行1公式/行2指标名，数据行3起
        if not isinstance(row[0], (pd.Timestamp,)) and row[0] is None:
            continue
        import datetime as _dt
        d = row[0] if isinstance(row[0], _dt.datetime) else None
        if d is None:
            continue
        dates.append(d)
        for ci, name in SHEET1_COLS.items():
            recs[name].append(row[ci] if ci < len(row) else None)
    wb.close()
    df = pd.DataFrame(recs)
    df.insert(0, "日期", pd.to_datetime(dates))
    return df.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "日期" else c)


def read_pred2(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["预测值2"]
    dates, recs = [], {v: [] for v in PRED2_COLS.values()}
    for row in ws.iter_rows(min_row=6, values_only=True):   # 行1-5为 iFinD 元数据
        import datetime as _dt
        d = row[0] if isinstance(row[0], _dt.datetime) else None
        if d is None:
            continue
        dates.append(d)
        for ci, name in PRED2_COLS.items():
            recs[name].append(row[ci] if ci < len(row) else None)
    wb.close()
    df = pd.DataFrame(recs)
    df.insert(0, "日期", pd.to_datetime(dates))
    return df.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name != "日期" else c)


def main():
    print("[1/3] 读取 轻量版 Sheet1（利率曲线+信用债指数）...")
    s1 = read_sheet1(SRC_LIGHT)
    print(f"      {s1['日期'].min().date()} .. {s1['日期'].max().date()}, {len(s1)} 行")

    print("[2/3] 读取 历史版 预测值2（多资产）...")
    p2 = read_pred2(SRC_HIST)
    print(f"      {p2['日期'].min().date()} .. {p2['日期'].max().date()}, {len(p2)} 行")

    print("[3/3] 合并 + 重叠列交叉验证 ...")
    overlap_cols = ["1Y国债(%)", "10Y国债(%)", "30Y国债(%)"]
    chk = s1.merge(p2, on="日期", how="inner", suffixes=("_A", "_B"))
    for c in overlap_cols:
        d = (chk[f"{c}_A"] - chk[f"{c}_B"]).abs().dropna()
        print(f"      重叠列 {c}: n={len(d)}, 最大|差|={d.max():.4f}"
              f"{'  ✓' if d.max() < 0.011 else '  ⚠ 超阈值，请检查来源'}")
    # 合并：Sheet1 为长历史主表，预测值2 补充 3 个独有资产；重叠列以 Sheet1 为准
    df = s1.merge(p2[["日期", "5Y二级资本债AAA-(%)", "30Y地方债(%)", "7Y中票AAA(%)"]],
                  on="日期", how="outer")
    df = df.sort_values("日期").drop_duplicates(subset=["日期"]).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    print(f"      {len(df):,} 行 x {len(df.columns)} 列 -> {OUT_FILE} ({OUT_FILE.stat().st_size / 1e3:.0f} KB)")
    print(f"      覆盖 {df['日期'].min().date()} .. {df['日期'].max().date()}")
    print(df.tail(3).to_string(index=False))
    print("DONE")
    return OUT_FILE


if __name__ == "__main__":
    main()
