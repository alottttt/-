# -*- coding: utf-8 -*-
"""因子边界回归测试，只用内存数据，不改 Parquet。

运行：python -X utf8 -B "量化机构行为研究/scripts/test_factor_regression.py"
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import factors


class FactorBoundaryTests(unittest.TestCase):
    def test_missing_components_are_not_zero(self):
        frame = pd.DataFrame({
            "组": ["全缺", "全缺", "部分缺", "部分缺", "真零", "真零", "有值", "有值"],
            "金额": [np.nan, np.nan, 2, np.nan, 0, 0, 2, 3],
        })
        total = factors._sum_complete(frame, ["组"], ["金额"])["金额"]
        self.assertTrue(pd.isna(total["全缺"]))
        self.assertTrue(pd.isna(total["部分缺"]))
        self.assertEqual(total["真零"], 0)
        self.assertEqual(total["有值"], 5)

    def test_warmup_counts_observed_days(self):
        values = pd.Series([np.nan] * 20 + [10.0] * 20)
        ratio = factors._volume_ratio(values)
        self.assertTrue(ratio.iloc[:29].isna().all())
        np.testing.assert_allclose(ratio.iloc[29:], 1)
        # 窗口足够，但当前观测缺失，不能输出信号。
        values.iloc[-1] = np.nan
        self.assertTrue(pd.isna(factors._volume_ratio(values).iloc[-1]))

    def test_real_zero_and_real_twenty_are_preserved(self):
        # 19 天的真实零不是缺失，第 20 天成交量是均量的 20 倍。
        ratio = factors._volume_ratio(pd.Series([0.0] * 19 + [10.0]))
        self.assertTrue(ratio.iloc[:-1].isna().all())
        self.assertEqual(ratio.iloc[-1], 20)
        self.assertEqual(factors._volume_ratio(pd.Series([10.0] * 19 + [0.0])).iloc[-1], 0)
        self.assertTrue(factors._volume_ratio(pd.Series([0.0] * 30)).isna().all())

    def test_constant_window_after_large_history_is_undefined(self):
        values = pd.Series([10000.1, -1700.2, 0.01, 9000.3] * 30 + [0.0] * 10)
        result = factors._rolling_standardize(values, 10, 5)
        self.assertTrue(pd.isna(result.iloc[-1]))
        self.assertTrue(factors._rolling_standardize(pd.Series([3.3] * 30), 10, 5).isna().all())

    def test_real_small_variation_is_not_discarded(self):
        values = pd.Series(np.arange(1.0, 11.0) * 1e-12)
        expected = (values.iloc[-1] - values.mean()) / values.std(ddof=1)
        result = factors._rolling_standardize(values, 10, 5)
        self.assertAlmostEqual(result.iloc[-1], expected, places=12)
        self.assertTrue(result.iloc[:4].isna().all())
        # 同样形状的小量级窗口应保持相同 zscore，而非被绝对阈值全部过滤。
        tiny = factors._rolling_standardize(values * 1e-100, 10, 5)
        self.assertAlmostEqual(tiny.iloc[-1], expected, places=12)

    def test_forms_do_not_read_future_data(self):
        dates = pd.date_range("2024-01-01", periods=300)
        buy = pd.Series(10 + np.sin(np.arange(300)), index=dates)
        sell = pd.Series(9 + np.cos(np.arange(300)), index=dates)
        net = buy - sell
        complete = factors._recognition_forms(net, buy, sell)
        prefix = factors._recognition_forms(net.iloc[:200], buy.iloc[:200], sell.iloc[:200])
        for name in factors.RECOG_FORMS:
            pd.testing.assert_series_equal(complete[name].iloc[:200], prefix[name])
        zero = pd.Series(0.0, index=dates)
        forms = factors._recognition_forms(zero, zero, zero)
        self.assertTrue(forms["净买入占比"].isna().all())
        self.assertTrue(forms["成交量"].eq(0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
