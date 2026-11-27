# -*- coding: utf-8 -*-
"""仅在需要补图时运行；正常运行 strategy.py 已自动生成 JPG。"""
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import strategy
from strategy_report import write_report


def main():
    positions, saved_metrics, config = strategy.load_results()
    for name, path in [("factor_sha256", strategy.FACTORS),
                       ("market_sha256", strategy.MARKET),
                       ("strategy_code_sha256", PROJECT / "src" / "strategy.py")]:
        if strategy.sha256(path) != config[name]:
            raise ValueError("输入或 strategy.py 已变更，请先重跑策略，避免给旧结果贴新参数")
    metrics = strategy.aggregate_metrics(positions)
    keys = ["实验", "策略", "期间", "样本口径"]
    pd.testing.assert_frame_equal(
        metrics.sort_values(keys).reset_index(drop=True),
        saved_metrics.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=0, atol=1e-9,
    )
    output = write_report(positions, metrics, config, strategy.EXPERIMENTS)
    print(f"JPG：{output}\n运行批次：{config['run_id']}")


if __name__ == "__main__":
    main()
