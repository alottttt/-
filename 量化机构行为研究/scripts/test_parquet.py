# -*- coding: utf-8 -*-
"""查看 data/ 下所有 parquet 的格式：表头(schema)、行数、列数、列名、前几行。"""
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

files = sorted(DATA.glob("*.parquet"))

for fp in files:
    pf = pq.ParquetFile(fp)
    sz = fp.stat().st_size / 1e6
    print("=" * 78)
    print(f"【{fp.name}】  {sz:.2f} MB")
    print(f"  行数 {pf.metadata.num_rows:,} | 列数 {pf.metadata.num_columns} | Row Groups {pf.metadata.num_row_groups}")
    print("  列名与类型：")
    for f in pf.schema_arrow:
        print(f"    {f.name}: {f.type}")
    df = pf.read().to_pandas()
    if "日期" in df.columns:
        d = pd.to_datetime(df["日期"], errors="coerce")
        print(f"  日期范围 {d.min().date()} ~ {d.max().date()}")
    print("  前 5 行：")
    print(df.head(5).to_string(index=False))
    print()
