# -*- coding: utf-8 -*-
"""
从 daily_detail 独立重算全部基础因子与识别信号，检查值、缺失、覆盖和放量/MA10 构造。

运行：python -X utf8 -B "量化机构行为研究/scripts/test_factor_quality.py"
加 --focus 只看放量/MA10 的构造与错误来源；加 --details 打印每个因子的结果；
加 --strict 在发现数值问题时返回 1。
只读，不覆盖任何 Parquet，也不运行 factors.compute/main。
复核边界是统一明细表到因子表；不代表已经逐笔验证原始 Excel 或预测有效性。
"""
import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))
# 只读取已声明的机构/券种/期限映射和基础因子规格，不调用生产计算函数。
from factors import BONDS, TENOR, SPECS, RECOG_BUCKETS, RECOG_FORMS, RECOG_INST_BUCKETS, EXCLUDED_FACTORS


NET, BUY, SELL = "净买入（亿元）", "买入（亿元）", "卖出（亿元）"
ATOL, RTOL = 1e-8, 1e-9


def show(df):
    print(df.to_string(index=False, na_rep="NaN",
                       float_format=lambda v: f"{v:.8g}"))


def section(title):
    print("\n" + "=" * 86 + "\n" + title + "\n" + "=" * 86)


def direct_zscore(x, window, minimum):
    """每个原始窗口独立计算均值/样本标准差，常数窗口不定义 zscore。"""
    def one(a):
        if not np.isfinite(a[-1]):
            return np.nan
        valid = a[np.isfinite(a)]
        if len(valid) < minimum or np.ptp(valid) == 0:
            return np.nan
        std = np.std(valid, ddof=1)
        return (a[-1] - np.mean(valid)) / std if std > 0 else np.nan
    return x.rolling(window, min_periods=minimum).apply(one, raw=True)


def build_reference(raw):
    """重用声明的规格，独立实现公式；全缺失聚合保留 NaN。"""
    cal = pd.DatetimeIndex(sorted(raw["日期"].unique()))

    @lru_cache(maxsize=None)
    def extract(institutions, bond, tenor, value):
        buckets = ["合计"] if tenor is None else TENOR[tenor]
        subset = raw[
            raw["机构"].isin(institutions) & raw["券种"].isin(BONDS[bond])
            & raw["期限桶标签"].isin(buckets)
        ]
        return subset.groupby("日期")[value].sum(min_count=1).reindex(cal)

    refs, families = {}, {}
    for spec in SPECS:
        p, kind, name = spec["params"], spec["kind"], spec["name"]
        bond, tenor = p["bond"], p.get("tenor")
        if kind == "net":
            value = extract((p["inst"],), bond, tenor, NET)
            if p.get("window"):
                value = value.rolling(p["window"], min_periods=p["window"]).sum()
        elif kind == "trade_alloc":
            value = (extract(("证券公司", "基金公司及产品"), bond, tenor, NET)
                     - extract(("大型银行", "保险公司"), bond, tenor, NET))
        elif kind == "divergence":
            b = extract((p["inst"],), bond, tenor, BUY)
            s = extract((p["inst"],), bond, tenor, SELL)
            value = (b - s) / (b + s).where((b + s) != 0)
        elif kind == "short_minus_long":
            value = (extract((p["inst"],), bond, p["short"], NET)
                     - extract((p["inst"],), bond, p["long"], NET))
        else:
            raise ValueError(f"尚未覆盖的公式类型：{kind}")
        refs[name] = value
        families[name] = spec["family"]

    recognized = raw[
        raw["券种"].isin(["国债-新债", "国债-老债"])
        & raw["期限桶标签"].isin(RECOG_BUCKETS)
    ].groupby(["机构", "期限桶标签", "日期"], observed=True)[[NET, BUY, SELL]].sum(min_count=1)
    for (inst, bucket), block in recognized.groupby(level=[0, 1], observed=True):
        if bucket not in RECOG_INST_BUCKETS.get(inst, ()):
            continue
        block = block.droplevel([0, 1]).reindex(cal)
        n, b, s = block[NET], block[BUY], block[SELL]
        gross = b + s
        # 先沿用现有 60/5/10 预热门槛，只隔离缺失和计算差异；不擅自改变设计。
        computed = {
            "净买入": n,
            "净买入占比": (b - s) / gross.where(gross != 0),
            "成交量": gross,
            "净买入zscore252": direct_zscore(n, 252, 60),
            "MA10偏离": direct_zscore(n, 10, 5),
            "买入放量": b / b.rolling(20, min_periods=10).mean().replace(0, np.nan),
            "卖出放量": s / s.rolling(20, min_periods=10).mean().replace(0, np.nan),
            "VR放量": gross / gross.rolling(20, min_periods=10).mean().replace(0, np.nan),
        }
        for form in RECOG_FORMS:
            name = f"识别_{inst}_{bucket}_{form}"
            if name in EXCLUDED_FACTORS:
                continue
            refs[name] = computed[form].replace([np.inf, -np.inf], np.nan)
            families[name] = f"识别/{form}"
    return refs, families, cal


def compare_all(saved, refs, families):
    reports, examples = [], []
    for name, expected in refs.items():
        actual = saved.get(name, pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        aligned = pd.concat([actual.rename("保存值"), expected.rename("独立参考")],
                            axis=1, sort=True).rename_axis("日期")
        a, b = aligned["保存值"], aligned["独立参考"]
        both = np.isfinite(a) & np.isfinite(b)
        numerical = both & ~np.isclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
        extra = a.notna() & b.isna()
        lost = a.isna() & b.notna()
        nonfinite = a.notna() & ~np.isfinite(a)
        wrong = numerical | extra | lost | nonfinite
        finite = a[np.isfinite(a)]
        reports.append({
            "因子": name, "类别": families[name],
            "保存行数": len(actual), "参考有效数": int(b.notna().sum()),
            "应空却有值": int(extra.sum()), "应有却缺失": int(lost.sum()),
            "有限值不一致": int(numerical.sum()), "非有限值": int(nonfinite.sum()),
            "不同取值数": finite.nunique(),
            "异常行数": int(wrong.sum()),
            "异常首日": aligned.index[wrong].min() if wrong.any() else pd.NaT,
            "异常末日": aligned.index[wrong].max() if wrong.any() else pd.NaT,
            "最大有限差": float((a[both] - b[both]).abs().max()) if both.any() else np.nan,
            "数值通过": not wrong.any(),
        })
        if wrong.any():
            ex = aligned.loc[wrong].head(2).reset_index()
            ex.insert(0, "因子", name)
            examples.append(ex)
    return pd.DataFrame(reports), pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


def coverage(saved, refs, cal):
    section("4. 覆盖率、常数序列和滚动窗口（覆盖低不自动等于数据错误）")
    common = cal[cal >= pd.Timestamp("2023-01-03")]
    rows = []
    for name, expected in refs.items():
        if not name.startswith("识别_"):
            continue
        x = expected.reindex(common)
        finite = x.dropna()
        if finite.empty:
            continue
        # 每个窗口单独计算 min/max，不能只看全历史 nunique。
        existing = saved.get(name, pd.Series(dtype=float, index=pd.DatetimeIndex([]))).reindex(cal)
        low = existing.rolling(60, min_periods=30).min().reindex(common)
        high = existing.rolling(60, min_periods=30).max().reindex(common)
        count = existing.rolling(60, min_periods=1).count().reindex(common)
        rows.append({
            "因子": name, "共同区间有效日": len(finite), "覆盖率%": 100 * len(finite) / len(common),
            "零值占有效值%": 100 * finite.eq(0).mean(),
            "60日常数窗口数": int((low.eq(high) & count.ge(30)).sum()),
            "全历史不同取值数": existing.nunique(),
        })
    frame = pd.DataFrame(rows)
    print(f"统一比较区间：{common.min().date()} ~ {common.max().date()}，{len(common)} 个明细交易日。")
    print("覆盖率使用保留缺失后的参考；常数窗口检查当前保存值。低成交可使比例信号合理地不可定义。")
    show(frame.sort_values("覆盖率%").head(10))
    print("\n全历史保存值只有一个不同取值的序列：")
    constants = [{"因子": n, "有效行数": len(s), "固定值": s.iloc[0]}
                 for n, s in saved.items() if len(s) and s.nunique() == 1]
    show(pd.DataFrame(constants)) if constants else print("无。")
    print("\n全历史并非常数，但局部 60 日窗口为常数的高频案例：")
    show(frame[(frame["60日常数窗口数"] > 0) & (frame["全历史不同取值数"] > 1)]
         .sort_values("60日常数窗口数", ascending=False).head(8))
    print("这些窗口的相关系数应为空，不能因浮点残差产生 inf 后进入排名。")
    return frame


def construction_details(raw, saved, cal, market):
    section("5. 周度移除检查 + 放量/MA10 构造与错误来源")
    weekly = [name for name in saved if name.endswith("_周度")]
    registered = [spec["name"] for spec in SPECS if spec["name"].endswith("_周度")]
    print(f"周度因子：注册表 {len(registered)} 个，保存数据 {len(weekly)} 个（均应为 0）。")
    print("买入放量 = 买入 / MA20(买入)；卖出放量 = 卖出 / MA20(卖出)。")
    print("VR放量 = (买入+卖出) / MA20(买入+卖出)，表示参与度，不含净买卖方向。")
    print("三个均值窗口都含当日，目前至少 10 个有效观测即开始计算，不是必须满 20 日。")
    print("MA10偏离 = (净买入 - MA10(净买入)) / STD10(净买入)，ddof=1，至少 5 个观测。")
    print("它是净买入的标准化偏离，不是 MA10/MA20 均线交叉；正值表示高于近期均值。")
    relevant = raw[raw["券种"].isin(["国债-新债", "国债-老债"])
                   & raw["期限桶标签"].isin(RECOG_BUCKETS)]
    grouped = relevant.groupby(["机构", "期限桶标签", "日期"], observed=True)[[NET, BUY, SELL]]
    kept, legacy = grouped.sum(min_count=1), grouped.sum()
    volume_rows, ma_rows, samples = [], [], []
    for (inst, bucket), block in kept.groupby(level=[0, 1], observed=True):
        if bucket not in RECOG_INST_BUCKETS.get(inst, ()):
            continue
        block = block.droplevel([0, 1]).reindex(cal)
        old = legacy.xs((inst, bucket), level=[0, 1]).reindex(cal)
        n = block[NET]
        for form, cols in [("买入放量", [BUY]), ("卖出放量", [SELL]), ("VR放量", [BUY, SELL])]:
            name = f"识别_{inst}_{bucket}_{form}"
            if name in EXCLUDED_FACTORS:
                continue
            actual = saved.get(name, pd.Series(dtype=float, index=pd.DatetimeIndex([]))).reindex(cal)
            x = block[cols].sum(axis=1, min_count=len(cols))
            old_x = old[cols].sum(axis=1, min_count=len(cols))
            mean = x.rolling(20, min_periods=10).mean()
            old_mean = old_x.rolling(20, min_periods=10).mean()
            reference = (x / mean).replace([np.inf, -np.inf], np.nan)
            reproduction = (old_x / old_mean).replace([np.inf, -np.inf], np.nan)
            extra = actual.notna() & reference.isna()
            both = actual.notna() & reference.notna()
            bias = both & ~np.isclose(actual, reference, atol=ATOL, rtol=RTOL, equal_nan=True)
            wrong = extra | bias
            if wrong.any():
                matches = np.isclose(actual[wrong], reproduction[wrong], atol=ATOL,
                                     rtol=RTOL, equal_nan=True)
                volume_rows.append({"形态": form, "因子": name,
                                    "预热不足却有值": int(extra.sum()), "有限值偏差": int(bias.sum()),
                                    "旧算法解释数": int(matches.sum()),
                                    "首日": cal[wrong].min(), "末日": cal[wrong].max()})
            if inst == "基金公司及产品" and bucket == "7-10Y":
                valid_dates = x.index[x.notna()]
                for pos in [0, 8, 9, 18, 19]:
                    if pos >= len(valid_dates):
                        continue
                    date = valid_dates[pos]
                    ratio = actual.loc[date] / reference.loc[date] if (
                        pd.notna(reference.loc[date]) and reference.loc[date] != 0) else np.nan
                    samples.append({"形态": form, "日期": date,
                                    "有效日数": int(x.loc[:date].tail(20).count()),
                                    "填零MA20": old_mean.loc[date], "保留缺失MA20": mean.loc[date],
                                    "保存放量": actual.loc[date], "参考放量": reference.loc[date],
                                    "放大倍数": ratio})
        name = f"识别_{inst}_{bucket}_MA10偏离"
        if name in EXCLUDED_FACTORS:
            continue
        actual = saved.get(name, pd.Series(dtype=float, index=pd.DatetimeIndex([]))).reindex(cal)
        mean, std = n.rolling(10, min_periods=5).mean(), n.rolling(10, min_periods=5).std(ddof=1)
        span = n.rolling(10, min_periods=5).max() - n.rolling(10, min_periods=5).min()
        bad = actual.notna() & span.eq(0)
        for date in cal[bad]:
            window = n.loc[:date].tail(10).dropna()
            ma_rows.append({"因子": name, "日期": date, "分子": n.loc[date] - mean.loc[date],
                            "历史滚动STD": std.loc[date], "独立窗口STD": window.std(ddof=1),
                            "窗口极差": window.max() - window.min(), "保存值": actual.loc[date]})
    print("\n放量错误的全量归因（仅改变缺失聚合，其余窗口/门槛保持相同）：")
    if volume_rows:
        v = pd.DataFrame(volume_rows)
        show(v.groupby("形态", sort=False).agg(
            受影响序列=("因子", "size"), 预热不足却有值=("预热不足却有值", "sum"),
            有限值偏差=("有限值偏差", "sum"), 旧算法解释数=("旧算法解释数", "sum"),
            首日=("首日", "min"), 末日=("末日", "max")).reset_index())
    else:
        print("未发现保存值与保留缺失参考之间的放量差异。")
    print("\n基金 7-10Y：第 1/9/10/19/20 个有效观测的过渡过程：")
    show(pd.DataFrame(samples))
    print("当窗口里只有 n 个真实观测，其余被填零时，错误均值=真实量之和/20；")
    print("保留缺失后的均值=真实量之和/n。n<10 应为空；10≤n<20 时，错误放量被放大 20/n 倍。")
    print("第 20 个有效观测后伪零完全退出窗口，差异消失。真实 20 日窗口内偶发放量=20 并非一概错误。")
    print("\nMA10 的零波动窗口定位：")
    if ma_rows:
        m = pd.DataFrame(ma_rows)
        show(m.groupby("因子", sort=False).agg(
            异常数=("日期", "size"), 首日=("日期", "min"), 末日=("日期", "max"),
            最大残留STD=("历史滚动STD", "max"), 最大真实STD=("独立窗口STD", "max")).reset_index())
        first = m.iloc[0]
        _, inst, bucket, _ = first["因子"].split("_")
        net = kept.xs((inst, bucket), level=[0, 1])[NET].reindex(cal)
        window = net.loc[:first["日期"]].tail(10)
        print(f"第一例：{first['因子']} / {first['日期'].date()}，原始十日窗口：")
        show(window.rename("净买入").rename_axis("日期").reset_index())
        print(f"分子={first['分子']:.12g}；全历史 rolling.std={first['历史滚动STD']:.12g}；")
        print(f"同一十日切片重新 rolling.std={window.rolling(10, min_periods=5).std().iloc[-1]:.12g}；")
        print(f"独立窗口 std={first['独立窗口STD']:.12g}；保存偏离={first['保存值']:.12g}。")
        print("证据指向全历史滚动方差计算的浮点残留：0/微小正数变成有限的 0，_safe 仅过滤 inf，拦不住它。")
    else:
        print("未发现零波动窗口被保存为有限 MA10 偏离。")
    common_market = market.index[(market.index >= cal.min()) & (market.index <= cal.max())]
    absent = common_market.difference(cal)
    print(f"\n窗口日历补充：明细日期 {len(cal)}，同期行情日期 {len(common_market)}；明细缺 {len(absent)} 个行情日期。")
    if len(absent):
        print("缺失日期：" + ", ".join(str(d.date()) for d in absent))
    print("现有 MA10/MA20 按明细行数滚动；两套日期需核实有效交易日口径，不能直接填零补齐。")
    issue = bool(weekly or registered or volume_rows or ma_rows)
    if issue:
        print("仍需处理：以上缺失/预热/零方差差异或残留周度因子。")
    else:
        print("构造检查通过：放量与保留缺失参考一致，零波动 MA10 不再保存为有限值，周度已移除。")
    print("含当日的信号只能在当日数据发布后使用。本检查只读，不自动修复生产数据。")
    return issue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", action="store_true", help="只打印周度移除和放量/MA10错误来源")
    parser.add_argument("--details", action="store_true", help="打印逐因子复核表")
    parser.add_argument("--strict", action="store_true", help="数值检查失败时退出码为 1")
    args = parser.parse_args()
    raw = pd.read_parquet(DATA / "daily_detail.parquet")
    fv = pd.read_parquet(DATA / "factor_values.parquet")
    market = pd.read_parquet(DATA / "market.parquet").set_index("日期").sort_index()
    section("1. 输入表和保存因子的基础质量")
    duplicates = int(fv.duplicated(["日期", "因子名"]).sum())
    raw_dup = int(raw.duplicated(["日期", "机构", "券种", "期限桶标签"]).sum())
    print(f"保存因子：{fv['因子名'].nunique()} 个 / {len(fv):,} 行；因子重复键 {duplicates}；源表重复键 {raw_dup}。")
    invalid = int((~np.isfinite(fv["值"])).sum())
    negative = int(raw[[BUY, SELL]].lt(0).to_numpy().sum())
    identity = (raw[NET] - raw[BUY] + raw[SELL]).abs().max()
    print(f"保存值 NaN/inf：{invalid}；负买入/卖出量：{negative}；净买入恒等式最大误差：{identity:.3g}。")
    if duplicates or raw_dup:
        print("重复键会使按日期对照含义不唯一，先处理后再继续。")
        return 1
    saved = {name: g.set_index("日期")["值"].sort_index()
             for name, g in fv.groupby("因子名", sort=False)}
    if args.focus:
        cal = pd.DatetimeIndex(sorted(raw["日期"].unique()))
        issue = construction_details(raw, saved, cal, market)
        return int(args.strict and issue)
    print("开始从统一明细独立重算；比例保留缺失，标准化逐原始窗口计算……", flush=True)
    refs, families, cal = build_reference(raw)
    report, examples = compare_all(saved, refs, families)
    unknown = sorted(set(saved) - set(refs))
    print(f"规格覆盖 {len(refs)} 个组合；保存表未被复核规格覆盖的名称 {len(unknown)} 个。")
    if unknown:
        print(unknown)
    section("2. 各因子族独立重算结果")
    summary = report.groupby("类别", sort=False).agg(
        规格数=("因子", "size"), 保存有值因子数=("保存行数", lambda s: int(s.gt(0).sum())),
        数值未通过因子数=("数值通过", lambda s: int((~s).sum())),
        应空却有值=("应空却有值", "sum"), 应有却缺失=("应有却缺失", "sum"),
        有限值不一致=("有限值不一致", "sum"), 最大有限差=("最大有限差", "max"),
    ).reset_index()
    show(summary)
    print("一致性容差：绝对 1e-8、相对 1e-9。参考沿用现有 60/5/10 预热门槛。")
    print("“数值未通过因子”指至少一条记录不一致，不表示该因子的全部日期都错误。")
    print("数值通过只说明符合已声明公式和缺失规则，不表示具有样本外预测能力。")
    if args.details:
        show(report)
    section("3. 不一致的定位样本")
    if len(examples):
        ex = examples.merge(report[["因子", "类别"]], on="因子")
        # 每个类别展示前四条，避免同类问题刷屏；--details 可查全部因子名。
        show(ex.groupby("类别", sort=False).head(4))
        print("\n按类别汇总异常发生区间（不代表区间内每天均异常）：")
        affected = report[~report["数值通过"]].groupby("类别", sort=False).agg(
            因子数=("因子", "size"), 异常记录数=("异常行数", "sum"),
            最早异常=("异常首日", "min"), 最晚异常=("异常末日", "max"),
        ).reset_index()
        show(affected)
    else:
        print("独立重算未发现不一致。")
    ratios = fv[fv["因子名"].str.startswith("分歧度_")
                | fv["因子名"].str.endswith("_净买入占比")]
    out_of_range = int(ratios["值"].abs().gt(1 + ATOL).sum())
    print(f"买卖失衡比超出 [-1,1] 的值：{out_of_range}。")
    coverage(saved, refs, cal)
    construction_issue = construction_details(raw, saved, cal, market)
    section("6. 结论与检查边界")
    base = report[~report["因子"].str.startswith("识别_")]
    daily_base = base[~base["因子"].str.endswith("_周度")]
    print(f"原始四族日度因子：{int(daily_base['数值通过'].sum())}/{len(daily_base)} 个通过数值复核。")
    print(f"保存表中残留周度因子：{sum(n.endswith('_周度') for n in saved)} 个（应为 0）。")
    present = report[report["保存行数"] > 0]
    print(f"保存的 {len(present)} 个因子中，至少一条记录数值未通过：{int((~present['数值通过']).sum())} 个；"
          f"结构性全程无有效参考：{int(report['参考有效数'].eq(0).sum())} 个。")
    print("两条名为“买入加速度”的基础因子，按既定公式实际是短端净买入减长端净买入，"
          "属于期限偏好，不是时间差分加速度。")
    print("本检查未重新认证原始 Excel 来源，也未给因子做样本外收益回测；"
          "评估引擎未修复前，不用其 IC 榜单判断因子有效性。")
    print("本脚本只读，未重写任何数据或生产公式。")
    issue = invalid or negative or identity > ATOL or unknown or out_of_range or construction_issue or (~report["数值通过"]).any()
    return int(args.strict and bool(issue))


if __name__ == "__main__":
    raise SystemExit(main())
