# =============================================================================
# SERIES FILTERING — final selection pipeline
#
# Order matters. Negatives are clipped FIRST, because an early negative row
# would otherwise define FIRST_SALE and make a new series look old enough to
# survive the cutoff.
#
#   1. clip negative NET_SALES to 0
#   2. drop series whose first (clipped) sale falls after 2025-06-30
#      -> those have an all-zero encoder input in EVERY training window,
#         since 2025-06-30 is the latest date any input window reaches
#   3. compute non-zero density over the train window
#   4. select — Section 6 compares the options before you commit
#
# Only one Snowflake query runs, returning ~117K rows (a few MB).
# =============================================================================

import os
import numpy as np
import pandas as pd

from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
import sys
sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import Snowflake_configuration

# =============================================================================
# SECTION 1: CONFIG
# =============================================================================

DATA_ROOT = r"C:\Users\G0004878\Desktop\TFT_Data"      # <-- adjust
OUT_DIR   = os.path.join(DATA_ROOT, "series_selection")

TABLE_NAME = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'

group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
date_col   = 'CAL_DATE'

TRAIN_START = '2023-04-01'
TRAIN_END   = '2025-12-31'

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 184

# Latest date any encoder input window reaches:
#   train window = 1006 days, i+o = 549 -> last input ends 2025-06-30
LAST_INPUT_END = pd.Timestamp('2025-06-30')

SANITIZE_GROUP_KEY = True

# ---- selection (set after reading Section 6) --------------------------------
SELECTION_METHOD = "compare_only"
#   "compare_only"        -> print the options, select nothing
#   "top_n_density"       -> your original plan
#   "top_n_sales"         -> rank by total sales
#   "floor_then_sales"    -> density floor, then top-N by sales   [recommended]
#   "coverage_target"     -> density floor, then enough series to hit X% of sales

TOP_N               = 30000
MIN_DENSITY         = 0.05     # used by floor_then_sales / coverage_target
TARGET_COVERAGE_PCT = 95.0     # used by coverage_target

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 2: PER-SERIES STATS FROM SNOWFLAKE  (clip -> aggregate)
# =============================================================================

print("="*60)
print("SECTION 2: SNOWFLAKE AGGREGATION")
print("="*60)

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

# STEP 2 of the plan — clip BEFORE anything else is derived
sdf = sdf.with_column('SALES_CLIPPED', F.greatest(F.col(target_col), F.lit(0)))

stats_sdf = sdf.group_by(F.col(group_col)).agg(
    F.sum('SALES_CLIPPED').alias('TOTAL_SALES'),
    F.count(F.lit(1)).alias('N_DAYS'),
    F.sum(F.when(F.col('SALES_CLIPPED') > 0, 1).otherwise(0)).alias('N_NONZERO'),
    F.min(F.when(F.col('SALES_CLIPPED') > 0, F.col(date_col))).alias('FIRST_SALE'),
    F.max(F.when(F.col('SALES_CLIPPED') > 0, F.col(date_col))).alias('LAST_SALE'),
    F.max('SALES_CLIPPED').alias('MAX_DAILY'),
    F.sum(F.when(F.col(target_col) < 0, 1).otherwise(0)).alias('N_NEGATIVE_DAYS'),
)

print("running aggregation...")
stats = stats_sdf.to_pandas()
session.close()
print("Snowflake session closed.")

stats.columns = [c.strip('"').upper() for c in stats.columns]
stats = stats.rename(columns={group_col.upper(): group_col})
stats['FIRST_SALE'] = pd.to_datetime(stats['FIRST_SALE'])
stats['LAST_SALE']  = pd.to_datetime(stats['LAST_SALE'])

TOTAL_SALES_ALL = stats['TOTAL_SALES'].sum()

print(f"\nseries              : {len(stats):,}")
print(f"total sales         : {TOTAL_SALES_ALL/1e5:,.2f} lacs")
print(f"negative rows       : {stats['N_NEGATIVE_DAYS'].sum():,} (clipped to 0)")
print(f"series w/ zero sales: {(stats['N_NONZERO']==0).sum():,}")

# =============================================================================
# SECTION 3: STEP 1 — FIRST-SALE CUTOFF
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: FIRST-SALE CUTOFF")
print("="*60)
print(f"latest date any input window reaches: {LAST_INPUT_END.date()}")

never_sold = stats['FIRST_SALE'].isna()
too_new    = stats['FIRST_SALE'] > LAST_INPUT_END

print(f"\nnever sold at all      : {never_sold.sum():,} "
      f"({stats.loc[never_sold,'TOTAL_SALES'].sum()/1e5:,.2f} lacs)")
print(f"first sale after cutoff: {too_new.sum():,} "
      f"({stats.loc[too_new,'TOTAL_SALES'].sum()/1e5:,.2f} lacs, "
      f"{stats.loc[too_new,'TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:.2f}% of sales)")

elig = stats[~never_sold & ~too_new].copy()
print(f"\neligible: {len(elig):,} series | "
      f"{elig['TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:.2f}% of sales retained")

# =============================================================================
# SECTION 4: DENSITY METRICS
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: DENSITY")
print("="*60)

train_end_ts = pd.Timestamp(TRAIN_END)

# your step 3: non-zero share over the whole train window
elig['WINDOW_DENSITY'] = elig['N_NONZERO'] / elig['N_DAYS']

# how much of the last input window the series was actually alive for
elig['INPUT_COVERAGE'] = (
    (LAST_INPUT_END - elig['FIRST_SALE']).dt.days.clip(lower=0)
    / INPUT_CHUNK_LENGTH
).clip(upper=1.0)

elig['DAYS_SINCE_LAST_SALE'] = (train_end_ts - elig['LAST_SALE']).dt.days
elig['AVG_DAILY_WHEN_SELLING'] = elig['TOTAL_SALES'] / elig['N_NONZERO']

print("WINDOW_DENSITY percentiles:")
for q in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{q:<3}: {elig['WINDOW_DENSITY'].quantile(q/100):.4f}")

print("\nINPUT_COVERAGE distribution:")
for thr in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
    k = elig['INPUT_COVERAGE'] >= thr
    print(f"  >= {thr:>4.0%}: {k.sum():>7,} series | "
          f"{elig.loc[k,'TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:>6.2f}% of sales")

# =============================================================================
# SECTION 5: THE DENSITY-vs-SALES TENSION, MADE CONCRETE
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: WHAT RANKING BY DENSITY COSTS")
print("="*60)

d_rank = set(elig.nlargest(TOP_N, 'WINDOW_DENSITY')[group_col])
s_rank = set(elig.nlargest(TOP_N, 'TOTAL_SALES')[group_col])

only_sales = elig[elig[group_col].isin(s_rank - d_rank)]
print(f"in top {TOP_N//1000}K by SALES but NOT by density: {len(only_sales):,}")
print(f"  sales they hold : {only_sales['TOTAL_SALES'].sum()/1e5:,.2f} lacs "
      f"({only_sales['TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:.2f}%)")
if len(only_sales):
    print(f"  median density  : {only_sales['WINDOW_DENSITY'].median():.4f}")
    print(f"  median avg sale : {only_sales['AVG_DAILY_WHEN_SELLING'].median():,.1f} "
          f"per selling day")

# =============================================================================
# SECTION 6: COMPARISON TABLE — read this before choosing
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: SELECTION OPTIONS")
print("="*60)
print(f"{'method':<34}{'series':>9}{'% sales':>10}{'med.density':>13}")
print("-"*66)

def summarise(df, label):
    if not len(df):
        print(f"{label:<34}{0:>9,}{0:>9.2f}%{'-':>13}")
        return
    print(f"{label:<34}{len(df):>9,}"
          f"{df['TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:>9.2f}%"
          f"{df['WINDOW_DENSITY'].median():>13.4f}")

for n in [20000, 30000, 40000, 50000]:
    if n > len(elig):
        continue
    summarise(elig.nlargest(n, 'WINDOW_DENSITY'), f"top {n//1000}K by density")
    summarise(elig.nlargest(n, 'TOTAL_SALES'),    f"top {n//1000}K by sales")
    print()

for thr in [0.02, 0.05, 0.10, 0.20]:
    summarise(elig[elig['WINDOW_DENSITY'] >= thr], f"density floor >= {thr:.0%}")

print()
for thr in [0.05]:
    base = elig[elig['WINDOW_DENSITY'] >= thr]
    for n in [20000, 30000, 40000]:
        if n > len(base):
            continue
        summarise(base.nlargest(n, 'TOTAL_SALES'),
                  f"floor {thr:.0%} + top {n//1000}K sales")

# =============================================================================
# SECTION 7: APPLY SELECTION
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: APPLYING SELECTION")
print("="*60)
print(f"method: {SELECTION_METHOD}")

if SELECTION_METHOD == "compare_only":
    print("\nNo selection applied. Read Section 6, set SELECTION_METHOD, rerun.")
    kept = None

elif SELECTION_METHOD == "top_n_density":
    kept = elig.nlargest(TOP_N, 'WINDOW_DENSITY')

elif SELECTION_METHOD == "top_n_sales":
    kept = elig.nlargest(TOP_N, 'TOTAL_SALES')

elif SELECTION_METHOD == "floor_then_sales":
    kept = elig[elig['WINDOW_DENSITY'] >= MIN_DENSITY].nlargest(TOP_N, 'TOTAL_SALES')

elif SELECTION_METHOD == "coverage_target":
    base = elig[elig['WINDOW_DENSITY'] >= MIN_DENSITY].sort_values(
        'TOTAL_SALES', ascending=False).reset_index(drop=True)
    base['CUM_PCT'] = base['TOTAL_SALES'].cumsum() / TOTAL_SALES_ALL * 100
    hit = base['CUM_PCT'] >= TARGET_COVERAGE_PCT
    n = int(hit.idxmax()) + 1 if hit.any() else len(base)
    kept = base.iloc[:n]
    if not hit.any():
        print(f"!! {TARGET_COVERAGE_PCT}% unreachable after the density floor; "
              f"keeping all {len(base):,}")

else:
    raise ValueError(f"unknown SELECTION_METHOD: {SELECTION_METHOD}")

if kept is not None:
    print(f"\nkept        : {len(kept):,} of {len(stats):,} series "
          f"({len(kept)/len(stats)*100:.1f}%)")
    print(f"sales kept  : {kept['TOTAL_SALES'].sum()/1e5:,.2f} lacs "
          f"({kept['TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100:.2f}%)")
    print(f"sales lost  : "
          f"{(TOTAL_SALES_ALL-kept['TOTAL_SALES'].sum())/1e5:,.2f} lacs")
    print(f"\nkept density: min {kept['WINDOW_DENSITY'].min():.4f} | "
          f"median {kept['WINDOW_DENSITY'].median():.4f}")
    print(f"kept history: earliest first sale "
          f"{kept['FIRST_SALE'].min().date()} | "
          f"latest {kept['FIRST_SALE'].max().date()}")

    # =========================================================================
    # SECTION 8: SAVE
    # =========================================================================
    print("\n" + "="*60)
    print("SECTION 8: SAVING")
    print("="*60)

    stats.to_parquet(os.path.join(OUT_DIR, "series_stats_all.parquet"), index=False)
    print(f"saved: series_stats_all.parquet   ({len(stats):,} rows)")

    keep_list = kept[[group_col]].reset_index(drop=True)
    keep_list.to_parquet(os.path.join(OUT_DIR, "series_keep_list.parquet"), index=False)
    keep_list.to_csv(os.path.join(OUT_DIR, "series_keep_list.csv"), index=False)
    print(f"saved: series_keep_list.parquet / .csv  ({len(keep_list):,} series)")

    dropped = stats[~stats[group_col].isin(set(kept[group_col]))]
    dropped[[group_col, 'TOTAL_SALES', 'N_NONZERO', 'FIRST_SALE']].to_parquet(
        os.path.join(OUT_DIR, "series_dropped.parquet"), index=False)
    print(f"saved: series_dropped.parquet     ({len(dropped):,} series, "
          f"{dropped['TOTAL_SALES'].sum()/1e5:,.2f} lacs needing a fallback forecast)")

    x = os.path.join(OUT_DIR, "series_selection_summary.xlsx")
    with pd.ExcelWriter(x, engine='openpyxl') as xl:
        pd.DataFrame({
            'metric': ['total series', 'eligible after cutoff', 'kept', 'dropped',
                       'total sales (lacs)', 'kept sales (lacs)', 'coverage %',
                       'selection method', 'top_n', 'min_density'],
            'value':  [len(stats), len(elig), len(kept), len(dropped),
                       round(TOTAL_SALES_ALL/1e5, 2),
                       round(kept['TOTAL_SALES'].sum()/1e5, 2),
                       round(kept['TOTAL_SALES'].sum()/TOTAL_SALES_ALL*100, 2),
                       SELECTION_METHOD, TOP_N, MIN_DENSITY],
        }).to_excel(xl, sheet_name='Summary', index=False)
        kept.head(2000).to_excel(xl, sheet_name='Kept (top 2000)', index=False)
        dropped.nlargest(2000, 'TOTAL_SALES').to_excel(
            xl, sheet_name='Dropped (top 2000)', index=False)
    print(f"saved: {x}")

print("\nDone.")
