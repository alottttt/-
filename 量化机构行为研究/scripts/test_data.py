# -*- coding: utf-8 -*-
from pathlib import Path
import pyarrow.parquet as pq
pf = pq.ParquetFile(r"D:\浙商资管\银行间市场机构行为研报清单\量化机构行为研究\data\daily_detail.parquet")
print(pf.schema_arrow)
print("行数", pf.metadata.num_rows)
print("列数", pf.metadata.num_columns)
print("Row Groups：", pf.metadata.num_row_groups)

batch = next(pf.iter_batches(batch_size=10), None)
if batch is not None:
    df = batch.to_pandas()
    print(df.to_string(index=False))

df = pf.read().to_pandas()

print(df.iloc[1000:1010].to_string(index=False))
print(df[df["机构"] == "大型银行"].head(10).to_string(index=False))
print(df["机构"].unique())
print(df["机构"].value_counts())

pf2=pq.ParquetFile(r"D:\浙商资管\银行间市场机构行为研报清单\量化机构行为研究\data\market.parquet")
print(pf2.schema_arrow)
print("行数", pf2.metadata.num_rows)
print("列数", pf2.metadata.num_columns)
print("Row Groups：", pf2.metadata.num_row_groups)
batch = next(pf2.iter_batches(batch_size=10), None)
if batch is not None:
    df = batch.to_pandas()
    print(df.to_string(index=False))