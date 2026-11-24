# -*- coding: utf-8 -*-
"""
legacy_strategy.py — 历史 Excel 参照（仅供迁移或重建历史参照）\n\n旧 strategy.py 复现「策略仓位」sheet 的 4 个策略（复现 + 对账）。

逻辑（已从 Excel 公式解码）：
  - 目标 y = 预测值!P列 = 未来1日10Y累计变动（1 日，非因子评估的 5 日）
  - 因子池 = 买债强度(B..AJ 35列) + 分券种(Q..AA 11列) + 加速度(B..E 4列)，共 50 列（空/非数值列自动过滤）
  - 每策略 (lookback W, topK K)，逐日 t（因果，只用 t 之前的数据）：
      ic[j]  = CORREL(因子_j[t-W:t-1], y[t-W:t-1])   （有效对，≥2 点）
      x[j]   = 因子_j[t]（当日值）
      mu[j]  = MEAN(因子_j[t-W:t-1])
      vote[j]= -SIGN(ic[j] * (x[j] - mu[j]))         （+1买入 / -1卖出）
      仓位    = SIGN(Σ 按 |ic| 降序取前 K 的 vote)
  - 效果评估：对bp=Σ|y|·(s·y<0)，错bp=Σ|y|·(s·y>0)，胜率=#(s·y<0)/#(s≠0且y≠0)

输出：
  - strategy_positions.parquet（日期 × 策略1..4 的 -1/0/+1 仓位）
  - strategy_metrics.parquet（策略 × lookback/topK/对bp/错bp/胜率）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SRC = ROOT / "机构行为研究 轻量公式版本.xlsx"
DATA = PROJECT / "data"
OUT_POS = DATA / "strategy_positions.parquet"
OUT_MET = DATA / "strategy_metrics.parquet"

STRATEGIES = [("策略1", 120, 5), ("策略2", 180, 5), ("策略3", 120, 7), ("策略4", 60, 7)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_cols(wb, sheet, c0, c1, r0, r1):
    ws = wb[sheet]
    return np.array([[_f(v) for v in row] for row in ws.iter_rows(
        min_row=r0, max_row=r1, min_col=c0, max_col=c1, values_only=True)], dtype=float)


def load_all():
    wb = load_workbook(SRC, read_only=True, data_only=True)
    buy = read_cols(wb, "买债强度因子", 2, 36, 14, 1464)   # B..AJ 35列
    div = read_cols(wb, "分券种因子", 17, 27, 14, 1464)     # Q..AA 11列（实际 6 列有值）
    acc = read_cols(wb, "机构买入加速度", 2, 5, 14, 1464)   # B..E 4列（实际 2 列有值）
    fac = np.hstack([buy, div, acc])                        # (1451, 50)
    tgt = read_cols(wb, "预测值", 16, 16, 3, 1453).ravel()  # P列 未来1日10Y变化
    dates = [r[0] for r in wb["买债强度因子"].iter_rows(
        min_row=14, max_row=1464, min_col=1, max_col=1, values_only=True)]
    wb.close()
    return fac, tgt, pd.to_datetime(dates)


def compute_strategy(fac, tgt, W, K):
    N, F = fac.shape
    votes = np.zeros(N)
    for t in range(1, N):                      # t=当日；只用 t 之前的数据（因果）
        s0 = max(0, t - W)
        if t - s0 < 2:
            continue
        fw, tw = fac[s0:t], tgt[s0:t]          # 历史窗口 [t-W, t-1]
        x, mu = fac[t], np.nanmean(fw, axis=0)
        ic = np.full(F, np.nan)
        for j in range(F):
            fj = fw[:, j]
            m = ~np.isnan(fj) & ~np.isnan(tw)
            if m.sum() < 2:
                continue
            ic[j] = np.corrcoef(fj[m], tw[m])[0, 1]
        ok = ~np.isnan(ic) & ~np.isnan(x) & ~np.isnan(mu)
        if not ok.any():
            continue
        v = -np.sign(ic[ok] * (x[ok] - mu[ok]))
        a = np.abs(ic[ok])
        order = np.argsort(-a, kind="stable")          # 按 |ic| 降序，稳定
        top = v[order[: min(K, len(order))]]
        votes[t] = np.sign(top.sum())
    return votes


def metrics(votes, tgt):
    s, y = votes, np.where(np.isnan(tgt), 0.0, tgt)
    right = (s * y < 0)
    wrong = (s * y > 0)
    bp_right = float(np.sum(right * np.abs(y)))
    bp_wrong = float(np.sum(wrong * np.abs(y)))
    denom = int(np.sum((s != 0) & (y != 0)))
    win = float(np.sum(right) / denom) if denom else np.nan
    return bp_right, bp_wrong, win



if __name__ == "__main__":
    raise SystemExit("这是历史参照模块；主流程请运行 src/strategy.py。")
