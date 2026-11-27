# -*- coding: utf-8 -*-
"""只读检查主流程：入选 IC/投票、策略汇总、缺失买卖量与放量。
eval_rolling 已退出正式产出；不要求也不读取旧评估表。
--strict：发现问题或未能完成检查时返回非零；不修改任何数据。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
BUCKETS = {"1Y": "≦1Y", "3Y": "1-3Y", "5Y": "3-5Y",
           "7Y": "5-7Y", "10Y": "7-10Y", "30Y": "20-30Y"}
TOL = 1e-8

def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def show(frame):
    print(frame.to_string(index=False, na_rep="NaN",
                          float_format=lambda v: f"{v:.6f}"))


def series(factors, institution, bucket, form):
    name = f"识别_{institution}_{bucket}_{form}"
    if name not in factors:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return factors[name].dropna().sort_index()



def check_strategy(fv, market, positions, config):
    section("1. 策略入选 IC：有限性、同窗口独立重算、选因子与投票")
    new = positions.loc[positions["实验"].isin(["修正基准", "扩展因子池"])]
    if new.empty:
        print("[无法判断] 缺少两组新实验。")
        return None
    wide = fv.pivot(index="日期", columns="因子名", values="值").reindex(market.index)
    target = 100 * (market["10Y国债(%)"].shift(-1) - market["10Y国债(%)"])
    issues, total = 0, 0
    for row in new.to_dict("records"):
        details = row["入选明细"]
        issues += int(len(details) != row["实际入选数"])
        issues += int(bool(row["信号有效"]) != pd.notna(row["仓位"]))
        if len(details):
            issues += int(row["票数合计"] != sum(d["投票"] for d in details))
            issues += int(row["仓位"] != np.sign(row["票数合计"]))
        for d in details:
            total += 1
            issues += int(not np.isfinite(d["IC"]) or abs(d["IC"]) > 1
                          or d["有效样本数"] < config["min_pairs"]
                          or pd.Timestamp(d["最新标签成熟日期"]) > row["日期"])
    samples, max_error = 0, 0.0
    for (experiment, strategy), group in new.groupby(["实验", "策略"]):
        valid_rows = group.loc[group["信号有效"]].sort_values("日期")
        if valid_rows.empty:
            issues += 1
            continue
        for row in valid_rows.iloc[sorted({0, len(valid_rows)//2, len(valid_rows)-1})].to_dict("records"):
            t = market.index.get_loc(row["日期"])
            start = max(0, t - int(row["lookback"]))
            history = wide.iloc[start:t]
            yt = target.iloc[start:t]
            candidates = []
            for name in sorted(config["pools"][experiment]):
                x = history[name]
                pair = pd.DataFrame({"x": x, "y": yt}).replace([np.inf, -np.inf], np.nan).dropna()
                current, mean = wide.iloc[t][name], x.replace([np.inf, -np.inf], np.nan).mean()
                if len(pair) < config["min_pairs"] or not np.isfinite(current) or not np.isfinite(mean):
                    continue
                varying = all(np.ptp(pair[c]) > config["near_constant_rtol"] * np.abs(pair[c]).max()
                              for c in ["x", "y"])
                if not varying:
                    continue
                ic = float(np.corrcoef(pair.x, pair.y)[0, 1])
                if not np.isfinite(ic) or abs(ic) > 1 + 1e-12:
                    continue
                candidates.append({"因子名": name, "IC": ic, "有效样本数": len(pair),
                                   "当日值": current, "历史均值": mean,
                                   "投票": int(-np.sign(ic * (current - mean)))})
            candidates.sort(key=lambda d: (-round(abs(d["IC"]), 12), d["因子名"]))
            expected, used_groups = [], set()
            for candidate in candidates:
                group = tuple(config.get("migration_groups", {}).get(candidate["因子名"], []))
                if group and group in used_groups:
                    continue
                expected.append(candidate)
                if group:
                    used_groups.add(group)
                if len(expected) == int(row["topK"]):
                    break
            saved = row["入选明细"]
            issues += int(len(candidates) != row["合格因子数"])
            if [d["因子名"] for d in expected] != [d["因子名"] for d in saved]:
                issues += 1
                print("入选不一致：", experiment, strategy, row["日期"])
                continue
            for a, b in zip(expected, saved):
                error = abs(a["IC"] - b["IC"])
                max_error = max(max_error, error)
                issues += int(error > TOL or a["有效样本数"] != b["有效样本数"]
                              or a["投票"] != b["投票"]
                              or not np.isclose(a["历史均值"], b["历史均值"], rtol=1e-10, atol=1e-10))
            samples += 1
    print(f"全表入选明细 {total:,} 条；独立重算 {samples} 个策略日，最大 IC 差 {max_error:.3g}，问题数 {issues}。")
    return bool(issues) if samples else None


def check_metrics(positions, metrics):
    section("2. 汇总：用每日结果独立核验两组共同样本")
    names = ["修正基准", "扩展因子池"]
    new = positions.loc[positions["实验"].isin(names)]
    issues, checked = 0, 0
    for strategy in sorted(new["策略"].unique()):
        groups = {name: new.loc[(new["实验"] == name) & (new["策略"] == strategy)].set_index("日期")
                  for name in names}
        dates = groups[names[0]].index.intersection(groups[names[1]].index)
        good = pd.Series(True, index=dates)
        for g in groups.values():
            good &= g.reindex(dates)["信号有效"] & g.reindex(dates)["目标有效"]
        for name, g in groups.items():
            h = g.loc[dates[good]]
            right = h["仓位"] * h["目标变动bp"] < 0
            wrong = h["仓位"] * h["目标变动bp"] > 0
            denominator = int(right.sum() + wrong.sum())
            bp_right = h.loc[right, "目标变动bp"].abs().sum()
            bp_wrong = h.loc[wrong, "目标变动bp"].abs().sum()
            win = right.sum() / denominator if denominator else np.nan
            saved = metrics.loc[(metrics["实验"] == name) & (metrics["策略"] == strategy)
                                & (metrics["期间"] == "全期") & (metrics["样本口径"] == "共同有效")]
            if len(saved) != 1:
                issues += 1
                continue
            r = saved.iloc[0]
            issues += int(r["参评日数"] != len(h) or r["有方向样本数"] != denominator
                          or not np.allclose([r["对bp"], r["错bp"], r["胜率"]],
                                             [bp_right, bp_wrong, win], atol=1e-9, rtol=0, equal_nan=True))
            checked += 1
    print(f"独立核验 {checked} 行全期共同样本指标，问题数 {issues}。")
    return bool(issues) if checked else None

def check_missing_volume(raw, factors):
    section("3. 缺失买卖量：原始空值 -> 默认求和变零 -> 首日放量伪峰")
    print("复核历史问题：src/factors.py 的默认 groupby.sum() 曾将缺失转零；以下检查当前保存值。")
    print("最小例子：sum([NaN, NaN]) =", pd.Series([np.nan, np.nan]).sum(),
          "；sum(min_count=1) =", pd.Series([np.nan, np.nan]).sum(min_count=1))
    values = ["买入（亿元）", "卖出（亿元）"]
    selected = raw[
        raw["券种"].isin(["国债-新债", "国债-老债"])
        & raw["期限桶标签"].isin(BUCKETS.values())
    ]
    grouped = selected.groupby(["机构", "期限桶标签", "日期"], observed=True)[values]
    old, preserved = grouped.sum(), grouped.sum(min_count=1)
    # 按原始组逐行对齐保存的成交量，确保真实零与缺失转零不混为一谈。
    compare = preserved.reset_index()
    compare["因子名"] = (
        "识别_" + compare["机构"].astype(str) + "_"
        + compare["期限桶标签"].astype(str) + "_成交量")
    saved = factors.loc[:, factors.columns.str.endswith("_成交量")].stack().rename("保存成交量")
    saved.index.names = ["日期", "因子名"]
    compare = compare.merge(saved.reset_index(), on=["日期", "因子名"], how="left")
    missing = compare[values].isna().all(axis=1)
    fake_zero = missing & compare["保存成交量"].eq(0)
    before = compare["日期"] < pd.Timestamp("2023-01-03")
    print(f"源买卖量均缺失、保存成交量却为零的组数：{int(fake_zero.sum()):,}")
    print(f"其中 2023-01-03 之前：{int((fake_zero & before).sum()):,}")
    actual_zero = compare[values].notna().all(axis=1) & compare[values].eq(0).all(axis=1)
    print(f"源表买卖量确有数值且均为零的组数：{int(actual_zero.sum()):,}（含当前未选入因子的机构/期限组合）")
    if fake_zero.any():
        sample = compare.loc[fake_zero].iloc[0]
        source_rows = selected[
            selected["机构"].eq(sample["机构"])
            & selected["期限桶标签"].eq(sample["期限桶标签"])
            & selected["日期"].eq(sample["日期"])
        ]
        print("\n一组伪零对应的原始新债/老债记录：")
        show(source_rows[["日期", "机构", "券种", "期限桶标签"] + values])
        key = (sample["机构"], sample["期限桶标签"], sample["日期"])
        show(pd.DataFrame({
            "算法": ["默认 sum", "保留缺失 sum(min_count=1)"],
            "买入": [old.loc[key, values[0]], preserved.loc[key, values[0]]],
            "卖出": [old.loc[key, values[1]], preserved.loc[key, values[1]]],
        }))
        print(f"这一组保存的成交量 = {sample['保存成交量']}")

    # 使用同样的 MA20/min_periods=10，仅改变缺失处理，隔离放量伪峰原因。
    inst, bucket = "基金公司及产品", "7-10Y"
    legacy = old.xs((inst, bucket), level=["机构", "期限桶标签"]).sort_index()
    correct = preserved.xs((inst, bucket), level=["机构", "期限桶标签"]).sort_index()
    days = correct.index[correct[values].notna().all(axis=1)]
    ratio_issue = False
    if len(days):
        date = days[0]
        rows = []
        for form, cols in [("买入放量", [values[0]]), ("卖出放量", [values[1]]),
                           ("VR放量", values)]:
            x_old = legacy[cols].sum(axis=1, min_count=len(cols))
            x_valid = correct[cols].sum(axis=1, min_count=len(cols))
            mean_old = x_old.rolling(20, min_periods=10).mean().loc[date]
            mean_valid = x_valid.rolling(20, min_periods=10).mean().loc[date]
            persisted = series(factors, inst, bucket, form).get(date, np.nan)
            ratio = x_valid.loc[date] / mean_valid if pd.notna(mean_valid) and mean_valid != 0 else np.nan
            ratio_issue |= bool(np.isfinite(persisted) and not np.isfinite(ratio))
            rows.append({
                "信号": form, "当日量": x_valid.loc[date], "填零后MA20": mean_old,
                "填零后放量": x_old.loc[date] / mean_old if mean_old else np.nan,
                "实际有效日数": int(x_valid.loc[:date].tail(20).count()),
                "保留缺失MA20": mean_valid, "保留缺失放量": ratio, "保存放量": persisted,
            })
        print(f"\n{inst} / {bucket}，首个买卖量有效日 {date.date()}：")
        show(pd.DataFrame(rows))
        print("若此前 19 日被填零，则 MA20=当日量/20，放量机械地等于 20。")
        print("保留缺失时只有 1 个有效日，未达到 10 日门槛，放量应为 NaN。")
    issue = bool(fake_zero.any() or ratio_issue)
    print("[发现问题] 原始缺失量被当作零或导致过早放量。" if issue
          else "[未发现问题] 本次原始缺失量对照与首日放量检查通过。")
    return issue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    names = ["daily_detail", "factor_values", "market", "strategy_positions", "strategy_metrics"]
    absent = [name for name in names if not (DATA / f"{name}.parquet").is_file()]
    if absent:
        print("缺少主流程输入：", absent)
        return 2
    loaded = {name: pd.read_parquet(DATA / f"{name}.parquet") for name in names}
    try:
        configs = [json.loads(pq.read_schema(DATA / f"{name}.parquet").metadata[b"strategy_config"])
                   for name in ["strategy_positions", "strategy_metrics"]]
        if configs[0]["run_id"] != configs[1]["run_id"]:
            raise ValueError("两张策略表批次不同")
        config = configs[0]
        for name, key in [("factor_values", "factor_sha256"), ("market", "market_sha256")]:
            if hashlib.sha256((DATA / f"{name}.parquet").read_bytes()).hexdigest() != config[key]:
                raise ValueError(f"{name} 已变化，策略结果需要重建")
    except (KeyError, ValueError, TypeError) as error:
        print("策略元信息检查失败：", error)
        return 2
    fv = loaded["factor_values"]
    factors = fv[fv["因子名"].str.startswith("识别_")].pivot(
        index="日期", columns="因子名", values="值").sort_index()
    market = loaded["market"].set_index("日期").sort_index()
    outcomes = [
        ("入选 IC 与投票", check_strategy(fv, market, loaded["strategy_positions"], config)),
        ("策略汇总", check_metrics(loaded["strategy_positions"], loaded["strategy_metrics"])),
        ("原始缺失买卖量与放量", check_missing_volume(loaded["daily_detail"], factors)),
    ]
    section("检查汇总")
    for label, issue in outcomes:
        print(f"[{'发现问题' if issue is True else '通过' if issue is False else '未完成'}] {label}")
    return int(args.strict and any(issue is not False for _, issue in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
