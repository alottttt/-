# -*- coding: utf-8 -*-
"""逐因子查看质量指标、分布统计和保存值实例（只读）。

默认展示全部已登记/已保存因子，包括全程不可定义、未写入长表的组合。
--match 关键词：按名称筛选；--rows N：首尾各 N 行，另附极值行；
--summary-only：只打印逐因子汇总；--strict：数据/数值异常时退出码为 1。
独立参考来自 test_factor_quality.py，不调用生产计算、不使用旧 IC 排名。
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_factor_quality import ATOL, RTOL, build_reference

DATA = Path(__file__).resolve().parents[1] / "data"
COMMON_START = pd.Timestamp("2023-01-03")
QUALITY_COLUMNS = [
    ["全期覆盖率%", "2023起覆盖率%", "有效首尾覆盖率%", "未保存日期数", "全期最长缺测", "有效首尾最长缺测"],
    ["保存NaN数", "+inf数", "-inf数", "非数值数", "无效日期数", "重复日期数", "重复涉及行数", "日历外行数"],
    ["不同取值数", "零值占比%", "2023起零值率%", "正值占比%", "负值占比%", "比例越界数", "负成交或放量数", "60日可检查窗口", "60日常数窗口"],
]
DISTRIBUTION_COLUMNS = [
    ["均值", "标准差(ddof=1)", "最小值", "P01", "P05", "P25", "中位数"],
    ["P75", "P95", "P99", "最大值", "偏度", "超额峰度"],
]
COMPARISON_COLUMNS = ["参考有效日数", "应空却有值", "应有却缺失", "有限值不一致", "最大有限差"]


def show(frame):
    print(frame.to_string(index=False, na_rep="—", max_colwidth=None,
                          float_format=lambda x: f"{x:.8g}"))


def date_text(value):
    return "—" if pd.isna(value) else value.strftime("%Y-%m-%d")


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else np.nan


def longest_gap(series):
    longest = current = 0
    for missing in series.isna():
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return longest


def inspect_factor(name, block, expected, category, calendar):
    """重复日期不择一保留；分布/窗口统计仅使用日历内、键唯一的有限值。"""
    block = block.sort_values("日期", kind="stable", na_position="last").reset_index(drop=True)
    values = block["值"]
    duplicate = block["日期"].notna() & block.duplicated("日期", keep=False)
    actual = block.loc[block["日期"].notna() & ~duplicate].set_index("日期")["值"]
    x = actual.reindex(calendar).where(lambda s: np.isfinite(s))
    finite = x.dropna()
    common = x.loc[x.index >= COMMON_START]
    active = x.loc[finite.index.min():finite.index.max()] if len(finite) else x.iloc[:0]
    registered = expected is not None
    reference = expected if registered else pd.Series(np.nan, index=calendar)
    aligned = pd.concat([actual.rename("保存"), reference.rename("参考")], axis=1, sort=True)
    a, b = aligned["保存"], aligned["参考"]
    both = np.isfinite(a) & np.isfinite(b)
    extra = a.notna() & ~np.isfinite(b)
    lost = np.isfinite(b) & ~np.isfinite(a)
    different = both & ~np.isclose(a, b, atol=ATOL, rtol=RTOL)
    maximum_error = float((a[both] - b[both]).abs().max()) if both.any() else np.nan
    roll = x.rolling(60, min_periods=30)
    count = x.rolling(60, min_periods=1).count()
    eligible = count.ge(30) & (np.arange(len(x)) >= 59)
    constant = eligible & roll.min().eq(roll.max())
    is_ratio = name.startswith("分歧度_") or name.endswith("_净买入占比")
    is_volume = name.startswith("识别_") and name.rsplit("_", 1)[-1] in {
        "成交量", "买入放量", "卖出放量", "VR放量"}
    finite_rows = values[np.isfinite(values)]
    row = {
        "因子": name, "类别": category, "保存行数": len(block),
        "有限值行数": int(np.isfinite(values).sum()), "有效日数": len(finite),
        "首个有效日": finite.index.min(), "末个有效日": finite.index.max(),
        "全期覆盖率%": percent(len(finite), len(calendar)),
        "2023起覆盖率%": percent(common.count(), len(common)),
        "有效首尾覆盖率%": percent(len(finite), len(active)),
        "未保存日期数": len(calendar.difference(block["日期"].dropna())),
        "全期最长缺测": longest_gap(x), "有效首尾最长缺测": longest_gap(active),
        "保存NaN数": int(values.isna().sum()),
        "+inf数": int(np.isposinf(values).sum()), "-inf数": int(np.isneginf(values).sum()),
        "非数值数": int(block["非数值"].sum()), "无效日期数": int(block["日期"].isna().sum()),
        "重复日期数": int(block.loc[duplicate, "日期"].nunique()),
        "重复涉及行数": int(duplicate.sum()),
        "日历外行数": int((block["日期"].notna() & ~block["日期"].isin(calendar)).sum()),
        "比例越界数": int(finite_rows.abs().gt(1 + ATOL).sum()) if is_ratio else 0,
        "负成交或放量数": int(finite_rows.lt(0).sum()) if is_volume else 0,
        "不同取值数": finite.nunique(), "零值占比%": percent(finite.eq(0).sum(), len(finite)),
        "2023起零值率%": percent(common.eq(0).sum(), common.count()),
        "正值占比%": percent(finite.gt(0).sum(), len(finite)),
        "负值占比%": percent(finite.lt(0).sum(), len(finite)),
        "60日可检查窗口": int(eligible.sum()), "60日常数窗口": int(constant.sum()),
        "参考有效日数": int(np.isfinite(reference).sum()) if registered else np.nan,
        "应空却有值": int(extra.sum()) if registered else np.nan,
        "应有却缺失": int(lost.sum()) if registered else np.nan,
        "有限值不一致": int(different.sum()) if registered else np.nan,
        "最大有限差": maximum_error if registered else np.nan,
        "均值": finite.mean(), "标准差(ddof=1)": finite.std(ddof=1),
        "最小值": finite.min(), "P01": finite.quantile(.01), "P05": finite.quantile(.05),
        "P25": finite.quantile(.25), "中位数": finite.median(),
        "P75": finite.quantile(.75), "P95": finite.quantile(.95),
        "P99": finite.quantile(.99), "最大值": finite.max(),
        "偏度": finite.skew() if len(finite) >= 3 and finite.nunique() > 1 else np.nan,
        "超额峰度": finite.kurt() if len(finite) >= 4 and finite.nunique() > 1 else np.nan,
    }
    bad_fields = QUALITY_COLUMNS[1] + ["比例越界数", "负成交或放量数"]
    issue = any(row[key] > 0 for key in bad_fields)
    issue = issue or not registered or bool(extra.any() or lost.any() or different.any())
    row["数值状态"] = ("未登记，待核验" if not registered else "存在异常" if issue
                       else "当前无可计算值" if not np.isfinite(reference).any() else "一致")
    row["数据或数值异常"] = issue
    return row, block, reference


def sample_rows(block, reference, rows):
    """首尾、极值样本去重；同一行兼具多个角色时保留全部标签。"""
    roles = {}

    def add(index, label):
        roles.setdefault(index, []).append(label)

    for index in block.head(rows).index:
        add(index, "首部")
    for index in block.tail(rows).index:
        add(index, "尾部")
    finite = block.loc[np.isfinite(block["值"]), "值"]
    if len(finite):
        add(finite.idxmin(), "最小值")
        add(finite.idxmax(), "最大值")
    bad = (~np.isfinite(block["值"]) | block["日期"].isna()
           | block.duplicated("日期", keep=False))
    if bad.any():
        add(block.index[bad][0], "异常样例")
    sample = block.loc[sorted(roles), ["日期", "值"]].copy()
    sample.insert(0, "样例类型", ["/".join(roles[index]) for index in sample.index])
    sample["独立参考"] = sample["日期"].map(reference)
    sample["保存减参考"] = sample["值"] - sample["独立参考"]
    return sample


def print_factor(number, total, row, block, reference, rows):
    print(f"\n[{number}/{total}] {row['因子']}  |  {row['类别']}  |  {row['数值状态']}")
    print("-" * 90)
    print(f"有效日期：{date_text(row['首个有效日'])} ~ {date_text(row['末个有效日'])}；"
          f"保存 {row['保存行数']} 行，有限 {row['有限值行数']} 行，有效唯一日期 {row['有效日数']} 天。")
    for columns in QUALITY_COLUMNS:
        show(pd.DataFrame([{key: row[key] for key in columns}]))
    print("分布统计（有限、日历内且日期不重复的保存值）：")
    for columns in DISTRIBUTION_COLUMNS:
        show(pd.DataFrame([{key: row[key] for key in columns}]))
    print("独立公式对照：")
    show(pd.DataFrame([{key: row[key] for key in COMPARISON_COLUMNS}]))
    if row["不同取值数"] == 1:
        print("说明：全期常数序列，相关系数不可定义；数值一致不等于有预测信息。")
    elif row["60日常数窗口"]:
        print("说明：存在局部常数窗口，相关评估必须单独过滤这些窗口。")
    if block.empty:
        print("实例行：无保存记录。参考全空时可能是来源缺失、预热不足或零分母/零方差。")
    else:
        print(f"实例行（首尾各 {rows} 行 + 极值，重叠行合并；— 表示缺失）：")
        show(sample_rows(block, reference, rows))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", default="", help="因子名包含的关键词，也可输入完整名称")
    parser.add_argument("--rows", type=positive_int, default=3, help="首尾各展示几行，默认 3")
    parser.add_argument("--summary-only", action="store_true", help="只打印逐因子质量汇总")
    parser.add_argument("--strict", action="store_true", help="所选因子有数据/数值异常时返回 1")
    args = parser.parse_args(argv)
    raw = pd.read_parquet(DATA / "daily_detail.parquet")
    factors = pd.read_parquet(DATA / "factor_values.parquet")
    factors["非数值"] = factors["值"].notna() & pd.to_numeric(factors["值"], errors="coerce").isna()
    factors["值"] = pd.to_numeric(factors["值"], errors="coerce")
    factors["日期"] = pd.to_datetime(factors["日期"], errors="coerce")
    invalid_names = factors["因子名"].isna() | factors["因子名"].astype(str).str.strip().eq("")
    print("因子逐项报告：读取保存表，并从 daily_detail 独立重算参考……", flush=True)
    refs, families, calendar = build_reference(raw)
    blocks = {name: block for name, block in factors.loc[~invalid_names].groupby("因子名", sort=False)}
    names = sorted(set(refs) | set(blocks))
    selected = [name for name in names if args.match in name]
    if not selected:
        parser.error(f"没有名称包含 {args.match!r} 的因子")
    common = calendar[calendar >= COMMON_START]
    print(f"数据：{DATA}\n保存表 {len(factors):,} 行 / {len(blocks)} 个名称；"
          f"登记 {len(refs)} 个规格；本次展示 {len(selected)} 个。")
    print(f"全期覆盖率分母：{len(calendar)} 个明细日期，{date_text(calendar.min())} ~ {date_text(calendar.max())}；"
          f"2023 起分母：{len(common)} 个明细日期。")
    print("有效首尾覆盖率仅统计首末有效日之间；最长缺测按明细日期计数。未保存日期与保存 NaN/inf 分开统计。")
    print("预热、来源缺失、零分母/零方差都可能使覆盖率降低；用独立对照判断是否缺了应有值。")
    print("60 日窗口保留缺失位置，完整 60 个明细日期中至少 30 个有限值才检查常数；这是可估计性诊断。")
    print(f"独立数值容差：atol={ATOL:g}、rtol={RTOL:g}；— 表示缺失或不可计算；零值比例分母为有限有效日。")
    print("零值占比为全期；2023起零值率仅统计共同区间，与本轮 >30% 撤出名单的统计口径一致。")
    print("不读取旧评估表：本报告描述数据与构造质量，不将 IC/RankIC 或回测收益作为本轮合格结论。")
    if invalid_names.any():
        print(f"发现 {int(invalid_names.sum())} 行缺失/空白因子名，无法归属；全局数据检查失败。")
    reports, details = [], []
    empty = factors.iloc[:0]
    for name in selected:
        row, block, reference = inspect_factor(name, blocks.get(name, empty), refs.get(name),
                                               families.get(name, "未登记"), calendar)
        reports.append(row)
        if not args.summary_only:
            details.append((row, block, reference))
    report = pd.DataFrame(reports)
    print("\n各类别概览：")
    show(report.groupby("类别", sort=False).agg(
        规格数=("因子", "size"), 保存有值数=("有效日数", lambda s: int(s.gt(0).sum())),
        异常因子数=("数据或数值异常", "sum"), 全期常数数=("不同取值数", lambda s: int(s.eq(1).sum())),
        平均全期覆盖率=("全期覆盖率%", "mean"),
    ).reset_index())
    if args.summary_only:
        print("\n逐因子质量汇总：")
        show(report[["因子", "数值状态", "有效日数", "全期覆盖率%", "2023起覆盖率%",
                     "2023起零值率%", "标准差(ddof=1)", "60日常数窗口", "应空却有值", "应有却缺失", "有限值不一致"]])
    else:
        for number, (row, block, reference) in enumerate(details, 1):
            print_factor(number, len(details), row, block, reference, args.rows)
    bad = int(report["数据或数值异常"].sum())
    print(f"\n本次展示 {len(report)} 个规格；数据/数值异常 {bad} 个；"
          f"当前无可计算值 {int(report['数值状态'].eq('当前无可计算值').sum())} 个。")
    print("报告只读：未写入 Parquet，未修改因子公式。")
    return int(args.strict and (bad > 0 or invalid_names.any()))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
