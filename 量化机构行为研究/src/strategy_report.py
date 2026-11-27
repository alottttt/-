# -*- coding: utf-8 -*-
"""接收同批次内存结果并输出 JPG；不依赖 strategy 模块，不改写运行记录。"""
import json
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

OUTPUT = PROJECT / "reports" / "strategy_report.jpg"


def common_positions(positions, experiments):
    new = positions.loc[positions["实验"].isin(experiments)].copy()
    new["可参评"] = new["信号有效"] & new["目标有效"]
    grid = new.pivot(index=["策略", "日期"], columns="实验", values="可参评")
    common = grid.reindex(columns=list(experiments)).fillna(False).all(axis=1)
    keys = pd.MultiIndex.from_frame(new[["策略", "日期"]])
    return new.loc[common.reindex(keys, fill_value=False).to_numpy()].copy()


def factor_contributions(common):
    """固定入选集合、移除单票不补位。边际贡献不具可加性。"""
    totals = {}
    for row in common.to_dict("records"):
        total = int(row["票数合计"])
        signal, target = int(row["仓位"]), float(row["目标变动bp"])
        details = row["入选明细"]
        if sum(int(d["投票"]) for d in details) != total or np.sign(total) != signal:
            raise ValueError("入选投票、票数合计与仓位不一致")
        for d in details:
            key = (row["实验"], row["策略"], d["因子名"])
            item = totals.setdefault(key, {"边际贡献bp": 0.0, "入选次数": 0, "改变方向次数": 0})
            without = int(np.sign(total - int(d["投票"])))
            item["边际贡献bp"] += -(signal - without) * target
            item["入选次数"] += 1
            item["改变方向次数"] += int(signal != without)
    return pd.DataFrame([dict(zip(["实验", "策略", "因子名"], key), **value)
                         for key, value in totals.items()])


def prepare_report(positions, metrics, config, experiments):
    summary = metrics.loc[metrics["实验"].isin(experiments)
                          & metrics["期间"].eq("全期")
                          & metrics["样本口径"].eq("共同有效")].copy()
    common = common_positions(positions, experiments)
    if common.empty:
        raise ValueError("没有共同有效样本，无法生成比较报告")
    for row in summary.to_dict("records"):
        g = common.loc[common["策略"].eq(row["策略"]) & common["实验"].eq(row["实验"])]
        score = -(g["仓位"].astype(float) * g["目标变动bp"])
        if len(g) != row["参评日数"] or not np.isclose(score.sum(), row["对减错bp"], atol=1e-9, rtol=0):
            raise ValueError("图表共同样本与汇总口径不一致")
    contributions = factor_contributions(common)
    return summary, common, contributions, config


def render_report(summary, common, contributions, config, experiments, output=OUTPUT):
    """用 Windows 内置 System.Drawing 出图，无额外绘图库依赖。"""
    if sys.platform != "win32":
        raise RuntimeError("当前图片渲染器使用 Windows System.Drawing")
    if len(config["strategies"]) != 4:
        raise ValueError("当前一页布局适用四个策略，请同步调整报告布局")
    pool_sizes = common.groupby("实验")["因子池数量"].first().astype(int).to_dict()
    rows, panels, limit_notes = [], [], []
    for name, w, k in config["strategies"]:
        for experiment in experiments:
            r = summary.loc[summary["策略"].eq(name) & summary["实验"].eq(experiment)].iloc[0]
            g = common.loc[common["策略"].eq(name) & common["实验"].eq(experiment)]
            rows.append({
                "expanded": experiment == experiments[1],
                "cells": [f"{name}  {w}/{k}", ("基准 " if experiment == experiments[0] else "扩展 ") + str(pool_sizes[experiment]),
                          f'{r["对bp"]:.2f}', f'{r["错bp"]:.2f}', f'{r["对减错bp"]:+.2f}',
                          f'{r["胜率"]:.2%}', f'{r["正确日数"]}/{r["有方向样本数"]}',
                          str(int(g["仓位"].eq(0).sum()))],
            })
            if k > pool_sizes[experiment]:
                limit_notes.append(f'{name} Top{k} 超过{experiment}的{pool_sizes[experiment]}个因子；实际入选{int(g["实际入选数"].min())}–{int(g["实际入选数"].max())}个。')
        group = contributions.loc[contributions["实验"].eq(experiments[1])
                                  & contributions["策略"].eq(name)]
        best = group.loc[group["边际贡献bp"] > 1e-9].sort_values(
            ["边际贡献bp", "因子名"], ascending=[False, True]).head(5)
        worst = group.sort_values(["边际贡献bp", "因子名"]).head(1)
        panels.append({"title": f"{name}  ·  窗口 {w} / Top{k}",
                       "top": json.loads(best.to_json(orient="records", force_ascii=False)),
                       "worst": json.loads(worst.to_json(orient="records", force_ascii=False))})
    counts = "/".join(str(int(v)) for v in summary.loc[summary["实验"].eq(experiments[0]), "参评日数"])
    payload = {"start": str(common["日期"].min().date()), "end": str(common["日期"].max().date()),
               "as_of": config["as_of"], "run_id": config["run_id"],
               "days": counts, "rows": rows, "panels": panels,
               "pool_note": f'修正基准 {pool_sizes[experiments[0]]} 因子 / 扩展池 {pool_sizes[experiments[1]]} 因子',
               "limit_note": " ".join(limit_notes) if limit_notes else "TopK 为入选上限；实际数量受当日合格因子数限制。偶数票可能抵消为零仓位。"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.jpg")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                     dir=output.parent, delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = Path(f.name)
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
                        str(PROJECT / "scripts" / "render_strategy_report.ps1"),
                        "-DataPath", str(payload_path.resolve()), "-OutputPath", str(temporary.resolve())],
                       check=True)
        if temporary.stat().st_size == 0:
            raise ValueError("图片为空")
        temporary.replace(output)
    finally:
        payload_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return output


def write_report(positions, metrics, config, experiments, output=OUTPUT):
    """供 strategy 主流程及独立补图入口共用。"""
    prepared = prepare_report(positions, metrics, config, experiments)
    return render_report(*prepared, experiments, output=output)
