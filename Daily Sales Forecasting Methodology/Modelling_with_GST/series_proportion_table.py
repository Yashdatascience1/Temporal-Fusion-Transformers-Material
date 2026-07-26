# =============================================================================
# PER-SERIES SELECTION TABLE
#
# Output columns:
#   PARENT_DEALER_CODE_MODEL_FAMILY
#   PROPORTION_OF_NON_ZERO_SALES   non-zero days / total days in window
#   PROP_NONZERO_SINCE_FIRST_SALE  non-zero days / days since first sale
#   TOTAL_SALES_PER_SERIES         sum of clipped NET_SALES
#   PCT_OF_TOTAL_SALES             this series' share of all sales (%)
#   CUM_PCT_OF_TOTAL_SALES         running coverage, series ranked by sales
#
# Negatives clipped to 0 first: a day that nets negative counts as a zero day,
# and clipped values feed the totals -- matching how the model is trained.
# Window is the training range only.
# =============================================================================

import os
import pandas as pd

from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
import sys
sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import Snowflake_configuration

# ── config ───────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\G0004878\Desktop\TFT_Data"       # <-- adjust
OUT_DIR   = os.path.join(DATA_ROOT, "series_selection")

TABLE_NAME = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'

group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
date_col   = 'CAL_DATE'
first_col  = 'FIRST_SALE_DATE'

TRAIN_START = '2023-04-01'
TRAIN_END   = '2025-12-31'

SANITIZE_GROUP_KEY = True
os.makedirs(OUT_DIR, exist_ok=True)

# ── query ─────────────────────────────────────────────────────────────────────
print("connecting...")
session = Session.builder.configs(Snowflake_configuration.ds1_role_json).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')

sdf = session.table(TABLE_NAME).filter(
    (F.col(date_col) >= TRAIN_START) & (F.col(date_col) <= TRAIN_END)
)

if SANITIZE_GROUP_KEY:
    sdf = sdf.with_column(
        group_col, F.replace(F.col(group_col), F.lit('<>'), F.lit('_'))
    )

# clip negatives BEFORE deriving anything
sdf = sdf.with_column('SALES_CLIPPED', F.greatest(F.col(target_col), F.lit(0)))

agg = sdf.group_by(F.col(group_col)).agg(
    F.count(F.lit(1)).alias('N_DAYS'),
    F.sum(F.when(F.col('SALES_CLIPPED') > 0, 1).otherwise(0)).alias('N_NONZERO'),
    F.sum('SALES_CLIPPED').alias('TOTAL_SALES_PER_SERIES'),
    F.min(F.col(first_col)).alias('FIRST_SALE_DATE'),
)

print("running aggregation...")
df = agg.to_pandas()
session.close()
print("done. rows:", len(df))

df.columns = [c.strip('"').upper() for c in df.columns]
df = df.rename(columns={group_col.upper(): group_col})
df['FIRST_SALE_DATE'] = pd.to_datetime(df['FIRST_SALE_DATE'])

# ── metrics ───────────────────────────────────────────────────────────────────
train_end = pd.Timestamp(TRAIN_END)

df['PROPORTION_OF_NON_ZERO_SALES'] = df['N_NONZERO'] / df['N_DAYS']

days_alive = (train_end - df['FIRST_SALE_DATE']).dt.days + 1
df['PROP_NONZERO_SINCE_FIRST_SALE'] = (df['N_NONZERO'] / days_alive).clip(upper=1.0)

total_all = df['TOTAL_SALES_PER_SERIES'].sum()
df['PCT_OF_TOTAL_SALES'] = df['TOTAL_SALES_PER_SERIES'] / total_all * 100

df = df.sort_values('TOTAL_SALES_PER_SERIES', ascending=False).reset_index(drop=True)
df['CUM_PCT_OF_TOTAL_SALES'] = df['PCT_OF_TOTAL_SALES'].cumsum()

out = df[[
    group_col,
    'PROPORTION_OF_NON_ZERO_SALES',
    'PROP_NONZERO_SINCE_FIRST_SALE',
    'TOTAL_SALES_PER_SERIES',
    'PCT_OF_TOTAL_SALES',
    'CUM_PCT_OF_TOTAL_SALES',
]].round(6)

# ── save ──────────────────────────────────────────────────────────────────────
p = os.path.join(OUT_DIR, "series_proportion_table.parquet")
c = os.path.join(OUT_DIR, "series_proportion_table.csv")
out.to_parquet(p, index=False)
out.to_csv(c, index=False)

print(f"\nseries        : {len(out):,}")
print(f"total sales   : {total_all/1e5:,.2f} lacs")
print(f"saved: {p}")
print(f"saved: {c}")

# quick coverage read
print("\ncoverage if ranked by TOTAL_SALES_PER_SERIES:")
for n in [30000, 40000, 50000, 60000, 70000]:
    if n <= len(out):
        print(f"  top {n:>6,}: {out.loc[n-1,'CUM_PCT_OF_TOTAL_SALES']:.2f}% of sales")
print("\nseries needed to reach:")
for pct in [90, 95, 98, 99]:
    hit = out['CUM_PCT_OF_TOTAL_SALES'] >= pct
    if hit.any():
        print(f"  {pct}%: {int(hit.idxmax())+1:,} series")
