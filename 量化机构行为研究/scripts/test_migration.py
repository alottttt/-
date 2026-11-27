# -*- coding: utf-8 -*-
"""迁移因子的手算、缺失、未来隔离及投票边界检查；不写任何结果表。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import factors as f
import strategy as s


class MigrationChecks(unittest.TestCase):
    def amounts(self):
        return pd.DataFrame([[20., 80.]] * 30, index=pd.date_range("2024-01-01", periods=30),
                            columns=["A", "B"])

    def test_weighted_share_and_disjoint_windows(self):
        a = self.amounts()
        a.iloc[20:24] = [10., 90.]
        a.iloc[24] = [900., 100.]
        result = f.migration_values(a)
        self.assertTrue(result.iloc[:24].isna().all().all())
        self.assertAlmostEqual(result.iloc[24, 0], 100 * (940 / 1400 - .2), places=12)
        self.assertAlmostEqual(result.iloc[24].sum(), 0., places=12)

    def test_scaling_does_not_create_migration(self):
        a = self.amounts()
        a.iloc[20:] *= 100
        np.testing.assert_allclose(f.migration_values(a).iloc[24:], 0, atol=1e-12)

    def test_zero_day_vs_zero_window(self):
        a = self.amounts()
        a.iloc[20:24] = 0
        self.assertTrue(f.migration_values(a).iloc[24].notna().all())
        a.iloc[24] = 0
        self.assertTrue(f.migration_values(a).iloc[24].isna().all())
        a = self.amounts()
        a.iloc[:20] = 0
        self.assertTrue(f.migration_values(a).iloc[24].isna().all())

    def test_bad_input_invalidates_entire_vector_and_window(self):
        for bad in [np.nan, np.inf, -1.]:
            a = self.amounts()
            a.iloc[2, 0] = bad
            result = f.migration_values(a)
            self.assertTrue(result.iloc[24:27].isna().all().all())
            self.assertTrue(result.iloc[27].notna().all())

    def test_no_future_data(self):
        a = self.amounts()
        a.iloc[:, 0] += np.arange(len(a))
        first = f.migration_values(a.iloc[:26])
        a.iloc[26:] = [1e9, .01]
        pd.testing.assert_frame_equal(first, f.migration_values(a).iloc[:26])

    def raw(self):
        tenors = [t for group in f.MIGRATION_TENORS.values() for t in group]
        rows = []
        for date in pd.date_range("2024-01-01", periods=30):
            for bond in [b for bs in f.MIGRATION_BONDS.values() for b in bs]:
                for tenor in [*tenors, "合计"]:
                    rows.append((date, "基金公司及产品", bond, tenor, 9. if tenor == "合计" else 1.))
        return pd.DataFrame(rows, columns=["日期", "机构", "券种", "期限桶标签", "买入（亿元）"])

    def test_bond_merge_and_totals_not_double_counted(self):
        raw = self.raw()
        b = f.migration_buy_matrix(raw, "基金公司及产品", "券种迁移")
        t = f.migration_buy_matrix(raw, "基金公司及产品", "期限迁移")
        self.assertEqual(b.iloc[0]["国债"], 18.)
        self.assertEqual(b.iloc[0]["政金债"], 18.)
        self.assertEqual(b.iloc[0]["其他"], 9.)
        self.assertEqual(b.iloc[0].sum(), 99.)
        self.assertEqual(t.iloc[0]["10Y+"], 8.)
        self.assertEqual(t.iloc[0].sum(), 18.)

    def test_absent_source_row_is_not_zero(self):
        raw = self.raw()
        mask = (raw["日期"].eq(raw["日期"].iloc[0]) & raw["券种"].eq("国债-新债")
                & raw["期限桶标签"].eq("10-15Y"))
        result = f.migration_buy_matrix(raw.loc[~mask], "基金公司及产品", "期限迁移")
        self.assertTrue(result.iloc[0].isna().all())
        self.assertTrue(f.migration_values(result).iloc[24].isna().all())
        self.assertTrue(f.migration_values(result).iloc[25].notna().all())
        missing_date = raw.loc[raw["日期"].ne(raw["日期"].iloc[0])]
        result = f.migration_buy_matrix(missing_date, "基金公司及产品", "券种迁移",
                                        calendar=pd.date_range("2024-01-01", periods=30))
        self.assertTrue(result.iloc[0].isna().all())

    def test_missing_category_total_invalidates_bond_shares(self):
        raw = self.raw()
        raw = raw.loc[~(raw["券种"].eq("其他") & raw["期限桶标签"].eq("合计"))]
        b = f.migration_buy_matrix(raw, "基金公司及产品", "券种迁移")
        self.assertTrue(b.isna().all().all())

    def test_one_vote_per_institution_per_family_with_backfill(self):
        names = [x["name"] for x in f.migration_specs()
                 if x["inst"] == "基金公司及产品" and x["family"] == "期限迁移"][:2]
        names += [next(x["name"] for x in f.migration_specs()
                       if x["inst"] == "基金公司及产品" and x["family"] == "券种迁移"), "基础因子"]
        names = np.array(names)
        selected = s.select_candidates(np.arange(4), np.array([.9, .8, .7, .6]), names, 3)
        self.assertEqual(selected.tolist(), [0, 2, 3])
        # 没有迁移族时，旧TopK行为保留。
        self.assertEqual(s.select_candidates(np.arange(3), np.array([.1, -.5, .3]),
                                            np.array(["A", "B", "C"]), 2).tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
