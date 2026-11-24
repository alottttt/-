# -*- coding: utf-8 -*-
"""原四策略：修正基准（原有因子）与扩展池，同一口径预测下一行情观测日 10Y 变化。

机构因子在样本日收盘后可得；只用此前已成熟的历史标签训练。
仓位是方向模型输出，不表示按样本日收盘价成交。主流程只生成两张策略表。
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from factors import SPECS, FACTOR_ALIASES

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FACTORS = DATA / "factor_values.parquet"
MARKET = DATA / "market.parquet"
OUT_POS = DATA / "strategy_positions.parquet"
OUT_MET = DATA / "strategy_metrics.parquet"
STRATEGIES = [("策略1", 120, 5), ("策略2", 180, 5),
              ("策略3", 120, 7), ("策略4", 60, 7)]
BASE, EXPANDED, LEGACY = "修正基准", "扩展因子池", "旧版Excel参照"
EXPERIMENTS = (BASE, EXPANDED)
MIN_PAIRS = 30
NEAR_CONSTANT_RTOL = 64 * np.finfo(float).eps
RANK_DECIMALS = 12
VERSION = "strategy_pool_comparison_v1"
KEYS = ["实验", "策略", "日期"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_mapping():
    """旧策略只读取分券种 V..AA 六列，并非现有全部基础因子。"""
    mapping = []
    for spec in SPECS:
        keep = (spec["kind"] in {"net", "short_minus_long"}
                or (spec["kind"] == "trade_alloc"
                    and spec["col"] in {"V", "W", "X", "Y", "Z", "AA"}))
        if keep:
            mapping.append({"原名": spec["name"], "现名": spec["name"],
                            "工作表": spec["sheet"], "列": spec["col"]})
    for old, col in [("中小行_国债≦1Y", "Z"), ("证券_国债7-10Y", "AB")]:
        mapping.append({"原名": old, "现名": FACTOR_ALIASES[old],
                        "工作表": "买债强度因子", "列": col})
    return sorted(mapping, key=lambda d: (d["工作表"], d["列"]))


def load_inputs(as_of=None):
    fv = pd.read_parquet(FACTORS)
    fv["日期"] = pd.to_datetime(fv["日期"])
    market = pd.read_parquet(MARKET)
    market["日期"] = pd.to_datetime(market["日期"])
    if fv[["日期", "因子名", "值"]].isna().any().any() or fv.duplicated(["日期", "因子名"]).any():
        raise ValueError("因子表缺失或主键重复")
    if not np.isfinite(fv["值"]).all():
        raise ValueError("保存的因子值必须有限")
    if market["日期"].isna().any() or market["日期"].duplicated().any():
        raise ValueError("行情日期缺失或重复")
    market = market.set_index("日期").sort_index()
    if as_of is not None:
        market = market.loc[:pd.Timestamp(as_of)]
    if market.empty or market.index.max() < fv["日期"].min():
        raise ValueError("截止日期前没有因子/行情重叠")
    wide = fv.pivot(index="日期", columns="因子名", values="值").sort_index(axis=1)
    absent = wide.index.difference(market.index)
    if as_of is not None:
        absent = absent[absent <= pd.Timestamp(as_of)]
    if len(absent):
        raise ValueError(f"因子日期没有对应行情：{list(absent[:5])}")
    market = market.loc[market.index >= wide.index.min()]
    wide = wide.reindex(market.index)
    names = [d["现名"] for d in baseline_mapping()]
    if len(set(names)) != len(names) or not set(names).issubset(wide.columns):
        raise ValueError("基准因子映射重复或缺少当前值序列")
    return wide, market, sorted(names)


def targets(market):
    y = market["10Y国债(%)"].astype(float)
    y = y.where(np.isfinite(y))
    return 100 * (y.shift(-1) - y)


def _center_columns(values, valid):
    """窗口内独立缩放、中心化，不依赖递推方差，也不使用绝对小数阈值。"""
    n = valid.sum(axis=0)
    scale = np.max(np.where(valid, np.abs(values), 0), axis=0)
    unit = np.divide(np.where(valid, values, 0), scale,
                     out=np.zeros_like(values, dtype=float), where=scale > 0)
    first = unit[valid.argmax(axis=0), np.arange(unit.shape[1])]
    local = np.where(valid, unit - first, 0)
    avg = np.divide(local.sum(axis=0), n, out=np.zeros(len(n)), where=n > 0)
    centered = np.where(valid, local - avg, 0)
    norm = np.sqrt((centered * centered).sum(axis=0))
    span = (np.max(np.where(valid, unit, -np.inf), axis=0)
            - np.min(np.where(valid, unit, np.inf), axis=0))
    varying = (n >= 2) & (span > NEAR_CONSTANT_RTOL) & (norm > 0)
    mean = np.where(n > 0, (first + avg) * scale, np.nan)
    return centered, norm, varying, mean


def rolling_statistics(wide, target, window):
    """在 t 日收盘后计算：窗口 [t-W,t-1]；h=1 的最后标签在 t 已成熟。"""
    # 固定内存布局，避免 DataFrame 复制/赋值导致求和次序变化和末位数值漂移。
    values = np.ascontiguousarray(wide.to_numpy(dtype=float))
    target = target.reindex(wide.index).to_numpy(dtype=float)
    n_days, n_factors = values.shape
    ic = np.full((n_days, n_factors), np.nan)
    means = np.full_like(ic, np.nan)
    counts = np.zeros((n_days, n_factors), dtype=np.int16)
    last_pairs = np.full((n_days, n_factors), -1, dtype=np.int32)
    for t in range(1, n_days):
        start = max(0, t - window)
        x = values[start:t]
        y = np.broadcast_to(target[start:t, None], x.shape)
        valid = np.isfinite(x) & np.isfinite(y)
        counts[t] = valid.sum(axis=0)
        xc, xn, xv, _ = _center_columns(x, valid)
        yc, yn, yv, _ = _center_columns(y, valid)
        corr = np.divide((xc * yc).sum(axis=0), xn * yn,
                         out=np.full(n_factors, np.nan), where=(xn * yn) > 0)
        ok = ((counts[t] >= MIN_PAIRS) & xv & yv & np.isfinite(corr)
              & (np.abs(corr) <= 1 + 1e-12))
        ic[t] = np.where(ok, np.clip(corr, -1, 1), np.nan)
        # 沿用原投票均值定义：历史窗口内所有有限因子值，不按目标是否缺失再筛均值。
        means[t] = _center_columns(x, np.isfinite(x))[3]
        last_pairs[t] = np.max(np.where(valid, np.arange(start, t)[:, None], -1), axis=0)
    return ic, means, counts, last_pairs


def build_positions(wide, market, baseline, progress=False):
    if not wide.index.equals(market.index) or wide.index.has_duplicates or not wide.index.is_monotonic_increasing:
        raise ValueError("因子、行情必须按共同有序日期一一对齐")
    if not baseline or not set(baseline).issubset(wide.columns):
        raise ValueError("基准池为空或不是扩展池的子集")
    # 固定名称顺序，只决定完全并列时的最后取舍，不随输入列顺序改变。
    wide = wide.sort_index(axis=1)
    names = wide.columns.to_numpy()
    x = np.ascontiguousarray(wide.to_numpy(dtype=float))
    y = targets(market)
    dates = wide.index
    next_dates = pd.Series(dates, index=dates).shift(-1)
    pools = {BASE: np.flatnonzero(wide.columns.isin(baseline)),
             EXPANDED: np.arange(len(names))}
    rows = []
    for window in sorted({w for _, w, _ in STRATEGIES}):
        ic, means, counts, last = rolling_statistics(wide, y, window)
        for strategy, w, k in STRATEGIES:
            if w != window:
                continue
            for experiment, pool in pools.items():
                for t, date in enumerate(dates):
                    eligible = pool[np.isfinite(ic[t, pool]) & np.isfinite(x[t, pool])
                                    & np.isfinite(means[t, pool])]
                    order = np.argsort(-np.round(np.abs(ic[t, eligible]), RANK_DECIMALS),
                                       kind="stable")
                    chosen = eligible[order[:k]]
                    votes = -np.sign(ic[t, chosen] * (x[t, chosen] - means[t, chosen])).astype(int)
                    details = [{"因子名": str(names[j]), "IC": float(ic[t, j]),
                                "有效样本数": int(counts[t, j]), "当日值": float(x[t, j]),
                                "历史均值": float(means[t, j]), "投票": int(votes[i]),
                                "入选顺序": i + 1,
                                "最新标签成熟日期": dates[int(last[t, j]) + 1]}
                               for i, j in enumerate(chosen)]
                    valid = bool(len(chosen))
                    status = ("有效" if valid else
                              "当日因子全部缺失" if not np.isfinite(x[t, pool]).any()
                              else "无合格因子（样本不足/常数/异常）")
                    rows.append({
                        "实验": experiment, "策略": strategy, "日期": date,
                        "lookback": w, "topK": k, "预测期限": "10Y", "预测窗口": 1,
                        "信号可用日期": date, "可用时点": "收盘后",
                        "最早执行日期": next_dates.iloc[t],
                        "预测起始日期": date, "预测结束日期": next_dates.iloc[t],
                        "目标变动bp": y.iloc[t], "目标有效": bool(np.isfinite(y.iloc[t])),
                        "训练窗口起始日": dates[max(0, t - w)] if t else pd.NaT,
                        "训练窗口结束日": dates[t - 1] if t else pd.NaT,
                        "最新训练标签成熟日期": max((d["最新标签成熟日期"] for d in details),
                                                    default=pd.NaT),
                        "因子池数量": len(pool), "合格因子数": len(eligible),
                        "实际入选数": len(chosen), "入选明细": details,
                        "票数合计": int(votes.sum()) if valid else None,
                        "仓位": int(np.sign(votes.sum())) if valid else None,
                        "信号有效": valid, "状态": status,
                    })
        if progress:
            print(f"完成 {window} 日窗口，两组因子池共用同一统计结果。", flush=True)
    out = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    for col in ["仓位", "票数合计", "合格因子数", "实际入选数", "因子池数量"]:
        out[col] = out[col].astype("Int16")
    return out


def _metadata(path):
    return json.loads((pq.read_schema(path).metadata or {})[b"strategy_config"])


def load_results():
    """两个文件分别原子替换；读取端拒绝不同批次混用。"""
    a, b = _metadata(OUT_POS), _metadata(OUT_MET)
    if a["run_id"] != b["run_id"]:
        raise ValueError("两张策略表批次不同，请重新完成 strategy.py 写出")
    return pd.read_parquet(OUT_POS), pd.read_parquet(OUT_MET), a


def load_legacy():
    """保留旧结果；只在首次迁移/重建旧参照时读取 Excel，主实验不使用 Excel 因子。"""
    if OUT_POS.exists() and "实验" in pq.read_schema(OUT_POS).names:
        pos, met, config = load_results()
        legacy = pos.loc[pos["实验"].eq(LEGACY)].copy()
        if legacy.empty:
            raise ValueError("既有策略表缺少旧版参照，不能静默覆盖")
        return legacy, config["legacy_reference"]
    from legacy_strategy import load_all, compute_strategy, metrics, SRC
    fac, tgt, dates = load_all()
    if OUT_POS.exists():
        old = pd.read_parquet(OUT_POS)
        old_metrics = pd.read_parquet(OUT_MET)
        origin = {"positions_sha256": sha256(OUT_POS), "metrics_sha256": sha256(OUT_MET)}
    else:
        old = pd.concat([pd.DataFrame({"日期": dates, "策略": name,
                                      "仓位": compute_strategy(fac, tgt, w, k).astype(int)})
                         for name, w, k in STRATEGIES], ignore_index=True)
        old_metrics = pd.DataFrame([
            dict(zip(["策略", "lookback", "topK", "对bp", "错bp", "胜率"],
                     [name, w, k, *metrics(old.loc[old["策略"].eq(name), "仓位"].to_numpy(), tgt)]))
            for name, w, k in STRATEGIES])
        origin = {"reconstructed_from_excel": True}
    blocks = []
    for name, w, k in STRATEGIES:
        g = old.loc[old["策略"].eq(name)].sort_values("日期").copy()
        if not pd.DatetimeIndex(g["日期"]).equals(pd.DatetimeIndex(dates)):
            raise ValueError("旧仓位日期与历史 Excel 输入不一致，拒绝猜测目标对齐")
        got = metrics(g["仓位"].to_numpy(), tgt)
        expected = old_metrics.loc[old_metrics["策略"].eq(name), ["对bp", "错bp", "胜率"]].iloc[0]
        if not np.allclose(got, expected.to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError("旧结果与当前 Excel 目标不匹配，拒绝改写历史参照")
        g["实验"], g["lookback"], g["topK"] = LEGACY, w, k
        g["预测期限"], g["预测窗口"] = "10Y", 1
        g["目标变动bp"] = tgt
        g["目标有效"] = np.isfinite(tgt)
        g["信号有效"] = g["仓位"].notna()
        g["可用时点"] = "旧版未认证"
        g["预测起始日期"] = g["日期"]
        for col in ["信号可用日期", "最早执行日期", "预测结束日期", "训练窗口起始日",
                    "训练窗口结束日", "最新训练标签成熟日期"]:
            g[col] = pd.NaT
        g["因子池数量"] = fac.shape[1]
        for col in ["合格因子数", "实际入选数", "票数合计"]:
            g[col] = pd.array([pd.NA] * len(g), dtype="Int16")
        g["入选明细"] = [[] for _ in range(len(g))]
        g["状态"] = "历史参照：执行时点及入选明细未认证"
        blocks.append(g)
    origin.update({"excel_sha256": sha256(SRC), "metrics": old_metrics.to_dict("records"),
                   "note": "保留旧仓位与汇总，不与新实验混作同口径因果对照"})
    return pd.concat(blocks, ignore_index=True), origin


def aggregate_metrics(positions):
    """从逐日结果聚合；无效目标不会填成零，共同样本只比较两组新实验。"""
    new = positions.loc[positions["实验"].isin(EXPERIMENTS)].copy()
    new["可参评"] = new["信号有效"] & new["目标有效"]
    grid = new.pivot(index=["策略", "日期"], columns="实验", values="可参评")
    common = grid.reindex(columns=list(EXPERIMENTS)).fillna(False).all(axis=1)
    rows = []
    for (experiment, strategy), group in positions.groupby(["实验", "策略"], sort=True):
        g = group.sort_values("日期").reset_index(drop=True)
        years = sorted(g["日期"].dt.year.unique())
        periods = [("全期", pd.Series(True, index=g.index))]
        periods += [(str(year), g["日期"].dt.year.eq(year)) for year in years]
        for period, period_mask in periods:
            scopes = ["历史原口径"] if experiment == LEGACY else ["各自有效", "共同有效"]
            for scope in scopes:
                h = g.loc[period_mask]
                eligible = (g["信号有效"] & g["目标有效"]).to_numpy(bool, copy=True)
                if scope == "共同有效":
                    idx = pd.MultiIndex.from_arrays([[strategy] * len(g), g["日期"]])
                    eligible &= common.reindex(idx, fill_value=False).to_numpy(bool)
                mask = eligible & period_mask.to_numpy(bool)
                s = g["仓位"].to_numpy(dtype=float, na_value=np.nan)
                y = g["目标变动bp"].to_numpy(dtype=float)
                right = mask & (s * y < 0)
                wrong = mask & (s * y > 0)
                directional = int(right.sum() + wrong.sum())
                # 不跨缺失日期计算换手；每个年度首日可与上一日连接，归入当前日期。
                change_ok = mask & np.r_[False, eligible[:-1]]
                diff = np.abs(s - np.r_[np.nan, s[:-1]])
                row = {
                    "实验": experiment, "策略": strategy,
                    "lookback": int(g["lookback"].iloc[0]), "topK": int(g["topK"].iloc[0]),
                    "期间": period, "起始日期": h["日期"].min(), "结束日期": h["日期"].max(),
                    "样本口径": scope, "候选日期数": len(h),
                    "目标有效日数": int(h["目标有效"].sum()),
                    "有效信号日数": int(h["信号有效"].sum()), "参评日数": int(mask.sum()),
                    "有方向样本数": directional, "正确日数": int(right.sum()),
                    "错误日数": int(wrong.sum()),
                    "对bp": float(np.abs(y[right]).sum()), "错bp": float(np.abs(y[wrong]).sum()),
                    "胜率": float(right.sum() / directional) if directional else np.nan,
                    "信号覆盖率": float(h["信号有效"].mean()) if len(h) else np.nan,
                    "仓位变动次数": int((change_ok & (diff > 0)).sum()),
                    "仓位变动幅度": float(np.nansum(diff[change_ok])),
                }
                row["对减错bp"] = row["对bp"] - row["错bp"]
                rows.append(row)
    return pd.DataFrame(rows)


def validate_results(pos, met):
    if pos.empty or pos.duplicated(KEYS).any() or pos[KEYS].isna().any().any():
        raise ValueError("逐日表主键错误")
    if not pos["仓位"].dropna().isin([-1, 0, 1]).all():
        raise ValueError("仓位值非法")
    if not np.isfinite(pos["目标变动bp"]).eq(pos["目标有效"]).all():
        raise ValueError("目标有效状态错误")
    new = pos.loc[pos["实验"].isin(EXPERIMENTS)]
    if not new["仓位"].notna().eq(new["信号有效"]).all():
        raise ValueError("把无效信号填成空仓")
    for row in new.itertuples(index=False):
        d = row._asdict()
        details = d["入选明细"]
        if len(details) != d["实际入选数"] or len(details) != min(d["合格因子数"], d["topK"]):
            raise ValueError("入选数量错误")
        if not pd.isna(d["最早执行日期"]) and d["最早执行日期"] <= d["信号可用日期"]:
            raise ValueError("执行日期不晚于收盘后信号")
        if len(details):
            if d["最新训练标签成熟日期"] > d["日期"]:
                raise ValueError("训练使用未成熟标签")
            if d["训练窗口结束日"] >= d["日期"]:
                raise ValueError("训练包含当日因子")
            if d["票数合计"] != sum(item["投票"] for item in details):
                raise ValueError("投票合计错误")
            if d["仓位"] != np.sign(d["票数合计"]):
                raise ValueError("仓位与票数不一致")
            for item in details:
                if not np.isfinite(item["IC"]) or abs(item["IC"]) > 1 or item["有效样本数"] < MIN_PAIRS:
                    raise ValueError("无效 IC 入选")
    metric_keys = ["实验", "策略", "期间", "样本口径"]
    if met.duplicated(metric_keys).any() or np.isinf(met.select_dtypes("number")).any().any():
        raise ValueError("汇总表主键或有限性错误")


DETAIL_TYPE = pa.list_(pa.struct([
    ("因子名", pa.string()), ("IC", pa.float64()), ("有效样本数", pa.int16()),
    ("当日值", pa.float64()), ("历史均值", pa.float64()), ("投票", pa.int8()),
    ("入选顺序", pa.int8()), ("最新标签成熟日期", pa.timestamp("ns")),
]))


def make_table(frame, config):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if "入选明细" in frame:
        at = table.schema.get_field_index("入选明细")
        table = table.set_column(at, pa.field("入选明细", DETAIL_TYPE),
                                 pa.array(frame["入选明细"].tolist(), type=DETAIL_TYPE))
    meta = dict(table.schema.metadata or {})
    meta[b"strategy_config"] = json.dumps(config, ensure_ascii=False).encode("utf-8")
    return table.replace_schema_metadata(meta)


def write_results(pos, met, config):
    """先写完并验证两份临时表，再分别替换；发生异常时恢复原文件。"""
    DATA.mkdir(parents=True, exist_ok=True)
    temporary, backups, replaced = {}, {}, []
    try:
        for path, frame in [(OUT_POS, pos), (OUT_MET, met)]:
            with tempfile.NamedTemporaryFile(dir=DATA, suffix=".tmp", delete=False) as handle:
                temporary[path] = Path(handle.name)
            pq.write_table(make_table(frame, config), temporary[path])
        validate_results(pd.read_parquet(temporary[OUT_POS]), pd.read_parquet(temporary[OUT_MET]))
        for path in (OUT_POS, OUT_MET):
            if path.exists():
                with tempfile.NamedTemporaryFile(dir=DATA, suffix=".bak", delete=False) as handle:
                    backups[path] = Path(handle.name)
                shutil.copy2(path, backups[path])
        for path in (OUT_POS, OUT_MET):
            temporary[path].replace(path)
            replaced.append(path)
    except BaseException:
        for path in replaced:
            if path in backups:
                backups[path].replace(path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        for path in [*temporary.values(), *backups.values()]:
            path.unlink(missing_ok=True)


def print_comparison(metrics, period="全期"):
    shown = metrics.loc[metrics["样本口径"].eq("共同有效") & metrics["期间"].eq(period)]
    print(shown[["策略", "实验", "参评日数", "胜率", "对bp", "错bp", "对减错bp",
                 "信号覆盖率", "仓位变动次数"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("以上为次日方向预测统计，不是按当日收盘成交的投资收益。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="仅用该日及以前输入；覆盖本批次的新实验结果，历史参照保留")
    args = parser.parse_args(argv)
    input_hashes = {"factor_sha256": sha256(FACTORS), "market_sha256": sha256(MARKET)}
    wide, market, baseline = load_inputs(args.as_of)
    legacy, legacy_info = load_legacy()
    print(f"修正基准 {len(baseline)} 个，扩展池 {len(wide.columns)} 个；共同日期 {len(wide)}。", flush=True)
    new = build_positions(wide, market, baseline, progress=True)
    pos = pd.concat([legacy.reindex(columns=new.columns), new], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    for col in ["仓位", "票数合计", "合格因子数", "实际入选数", "因子池数量"]:
        pos[col] = pos[col].astype("Int16")
    met = aggregate_metrics(pos)
    # 迁移后全期旧指标必须保持，不把新日历/新目标强套到旧仓位。
    for old in legacy_info["metrics"]:
        saved = met.loc[(met["实验"] == LEGACY) & (met["策略"] == old["策略"]) & (met["期间"] == "全期")]
        if len(saved) != 1 or not np.allclose(saved[["对bp", "错bp", "胜率"]].iloc[0].to_numpy(float),
                                             [old[k] for k in ["对bp", "错bp", "胜率"]], atol=1e-9, rtol=0):
            raise ValueError("历史参照指标未保真")
    validate_results(pos, met)
    if input_hashes != {"factor_sha256": sha256(FACTORS), "market_sha256": sha256(MARKET)}:
        raise ValueError("运行中输入发生变化，拒绝保存混合版本")
    missing_dates = wide.index[wide.isna().all(axis=1)]
    config = {
        "version": VERSION, "run_id": str(uuid.uuid4()), **input_hashes,
        "strategy_code_sha256": sha256(__file__), "as_of": str(market.index.max().date()),
        "strategies": STRATEGIES, "min_pairs": MIN_PAIRS,
        "calendar": "market.parquet observation dates; not officially certified trading calendar",
        "all_factors_missing_dates": [str(d.date()) for d in missing_dates],
        "release": "same sample day after close",
        "target": "100 * (10Y[t+1] - 10Y[t]); prediction only, not executable close-to-close PnL",
        "training": "[t-W,t-1], labels mature no later than t close; mean uses finite historical factor values",
        "ties": f"abs(IC) rounded {RANK_DECIMALS} decimals, factor name ascending, exactly at most K",
        "near_constant_rtol": NEAR_CONSTANT_RTOL,
        "pools": {BASE: baseline, EXPANDED: wide.columns.tolist()},
        "baseline_mapping": baseline_mapping(),
        "baseline_removed": ["保险_国债≦1Y"],
        "legacy_reference": legacy_info,
        "sample_note": "fixed full-history-cleaned factor universe; historical comparison is not strict out-of-sample evidence",
        "turnover": "model position changes on adjacent eligible observation dates; no fill across missing signals",
    }
    write_results(pos, met, config)
    print(f"写出逐日结果 {len(pos):,} 行、汇总 {len(met):,} 行；同批次 {config['run_id']}")
    print_comparison(met)
    return pos, met


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
