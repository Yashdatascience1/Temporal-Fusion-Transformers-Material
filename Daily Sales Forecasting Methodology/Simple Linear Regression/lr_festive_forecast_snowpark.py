# =============================================================================
# LR FESTIVE FORECAST (Snowpark source) -> DATE-KEYED -> BLEND WITH DAILY TFT
#
# Pipeline:
#   1. Aggregate NET_SALES by (series, lag_day, year) INSIDE SNOWFLAKE using
#      conditional aggregation -- avoids pulling 117K series of daily rows
#   2. Vectorised OLS on 2023-2025 -> extrapolate 2026
#   3. Map each lag_day to its 2026 CAL_DATE via the festive-feature file
#   4. Left-join onto the daily TFT forecast, keyed by (series, CAL_DATE)
#   5. Blend on festive dates; TFT alone everywhere else
#
# Why the aggregation runs server-side:
#   raw daily for 117K series ~ 200+ GB in pandas
#   aggregated result: 117K series x 3 years = ~351K rows x 46 cols (~130 MB)
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

DATA_ROOT    = r"C:\Users\G0004878\Desktop\TFT_Data"          # <-- adjust
FEATURE_XLSX = os.path.join(DATA_ROOT, "2026_festive_feature.xlsx")
TFT_PATH     = os.path.join(DATA_ROOT, "predictions_jul_dec_2026.parquet")
OUT_DIR      = os.path.join(DATA_ROOT, "lr_blend_output")

TABLE_NAME = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'

group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
year_col   = 'YEAR'
date_col   = 'CAL_DATE'
tft_col    = 'PREDICTED_NET_SALES'      # column name in the TFT parquet

HISTORICAL_YEARS = [2023, 2024, 2025]
FORECAST_YEAR    = 2026

BLEND_WEIGHT_LR = 0.5       # 0.5 = simple average (Sheet4 method)
CLIP_NEGATIVE   = True

SANITIZE_GROUP_KEY = True   # replace '<>' with '_' to match the TFT output

penalty_cols = [
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 2: SNOWFLAKE SESSION + COLUMN NAME RESOLUTION
# =============================================================================
# Lag-day names contain '-' and '+', so Snowflake stores them as quoted
# identifiers. df.columns may return them as '"N-16"' rather than 'N-16'.
# Resolve the real names before building any expression.

print("="*60)
print("SECTION 2: SNOWFLAKE CONNECTION")
print("="*60)

session = Session.builder.configs(Snowflake_configuration.ds1_role_json).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')

sdf = session.table(TABLE_NAME)
sf_cols = sdf.columns
print(f"table columns: {len(sf_cols)}")

def resolve(name, available):
    """Find how Snowflake actually spells this column."""
    for cand in (name, f'"{name}"', name.upper(), f'"{name.upper()}"'):
        if cand in available:
            return cand
    return None

col_map, missing = {}, []
for c in penalty_cols:
    r = resolve(c, sf_cols)
    if r is None:
        missing.append(c)
    else:
        col_map[c] = r

if missing:
    print(f"\n!! lag-day columns not found in Snowflake: {missing}")
    print("   sample of actual column names:", sf_cols[:8])
    raise ValueError("Resolve the column names above before continuing.")

print(f"resolved all {len(col_map)} lag-day columns")

grp_sf  = resolve(group_col,  sf_cols) or group_col
yr_sf   = resolve(year_col,   sf_cols) or year_col
tgt_sf  = resolve(target_col, sf_cols) or target_col

# =============================================================================
# SECTION 3: SERVER-SIDE CONDITIONAL AGGREGATION
# =============================================================================
# One query. For every lag-day column, sum NET_SALES over the days it flags.
#   SUM(CASE WHEN "N-16" != 0 THEN NET_SALES ELSE 0 END) AS "N-16"
# NULL flags fall to the ELSE branch, so they contribute 0.
# Result: one row per (series, year), one column per lag-day.

print("\n" + "="*60)
print("SECTION 3: AGGREGATING IN SNOWFLAKE")
print("="*60)

years_present = sorted(
    r[0] for r in sdf.select(F.col(yr_sf)).distinct().collect()
)
print(f"years in table: {years_present}")

# YEAR may be stored as 23/24/25 or 2023/2024/2025
two_digit = max(years_present) < 100
wanted = [y - 2000 for y in HISTORICAL_YEARS] if two_digit else HISTORICAL_YEARS
print(f"filtering to  : {wanted}")

agg_exprs = [
    F.sum(
        F.when(F.col(col_map[c]).isNotNull() & (F.col(col_map[c]) != 0),
               F.col(tgt_sf)).otherwise(0)
    ).alias(f'"{c}"')
    for c in penalty_cols
]

sdf_hist = sdf.filter(F.col(yr_sf).isin(wanted))

if SANITIZE_GROUP_KEY:
    sdf_hist = sdf_hist.with_column(
        grp_sf, F.replace(F.col(grp_sf), F.lit('<>'), F.lit('_'))
    )

agg_sdf = sdf_hist.group_by(F.col(grp_sf), F.col(yr_sf)).agg(*agg_exprs)

print("running query (this is the only Snowflake pull)...")
hist_wide = agg_sdf.to_pandas()
session.close()
print("Snowflake session closed.")

# Normalise the pandas column names back to the plain forms
rename = {}
for c in hist_wide.columns:
    plain = c.strip('"')
    rename[c] = plain
hist_wide = hist_wide.rename(columns=rename)

grp_pd = resolve(group_col, list(hist_wide.columns)) or group_col
yr_pd  = resolve(year_col,  list(hist_wide.columns)) or year_col
hist_wide = hist_wide.rename(columns={grp_pd: group_col, yr_pd: year_col})

if two_digit:
    hist_wide[year_col] = hist_wide[year_col] + 2000

print(f"\nrows pulled : {len(hist_wide):,}")
print(f"series      : {hist_wide[group_col].nunique():,}")
print(f"memory      : {hist_wide.memory_usage(deep=True).sum()/1e6:,.0f} MB")

# =============================================================================
# SECTION 4: WIDE -> LONG -> PIVOT BY YEAR
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: RESHAPING")
print("="*60)

long_df = hist_wide.melt(
    id_vars=[group_col, year_col],
    value_vars=penalty_cols,
    var_name='LAG_DAY',
    value_name=target_col,
)
long_df[target_col] = long_df[target_col].astype(float).fillna(0.0)
print(f"long rows: {len(long_df):,}")
del hist_wide

wide = (
    long_df.pivot_table(index=[group_col, 'LAG_DAY'], columns=year_col,
                        values=target_col, aggfunc='sum')
           .reindex(columns=HISTORICAL_YEARS)
)
print(f"pairs (series x lag-day): {len(wide):,}")
print("\ncoverage per year:")
for y in HISTORICAL_YEARS:
    print(f"  {y}: {wide[y].notna().sum():,} non-null "
          f"({wide[y].notna().mean()*100:.1f}%)")
wide = wide.fillna(0.0)
del long_df

# =============================================================================
# SECTION 5: VECTORISED OLS -> 2026
# =============================================================================
# slope    = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
# forecast = ybar + slope * (2026 - xbar)
# Identical to sklearn LinearRegression per row, computed for all rows at once.

print("\n" + "="*60)
print("SECTION 5: OLS EXTRAPOLATION")
print("="*60)

Y = wide[HISTORICAL_YEARS].to_numpy(dtype=np.float64)
x = np.asarray(HISTORICAL_YEARS, dtype=np.float64)

xbar  = x.mean()
ybar  = Y.mean(axis=1)
slope = ((x - xbar) * (Y - ybar[:, None])).sum(axis=1) / ((x - xbar) ** 2).sum()
lr    = ybar + slope * (FORECAST_YEAR - xbar)

n_neg = int((lr < 0).sum())
if CLIP_NEGATIVE:
    lr = np.clip(lr, 0, None)

lr_df = wide.reset_index()
lr_df.columns = [group_col, 'LAG_DAY'] + [f'Y{y}' for y in HISTORICAL_YEARS]
lr_df['SLOPE_PER_YEAR'] = slope
lr_df['LR_FORECAST']    = lr

print(f"pairs            : {len(lr_df):,}")
print(f"negative clipped : {n_neg:,} ({n_neg/len(lr_df)*100:.2f}%)")
print(f"declining pairs  : {(slope<0).sum():,} ({(slope<0).mean()*100:.1f}%)")

hist_tot = {y: lr_df[f'Y{y}'].sum() for y in HISTORICAL_YEARS}
print("\nfestive-window totals (lacs):")
for y in HISTORICAL_YEARS:
    print(f"  {y}       : {hist_tot[y]/1e5:>9,.2f}")
print(f"  {FORECAST_YEAR} (LR) : {lr_df['LR_FORECAST'].sum()/1e5:>9,.2f}")
print(f"  implied YoY vs 2025: "
      f"{(lr_df['LR_FORECAST'].sum()/hist_tot[2025]-1)*100:+.1f}%")

# =============================================================================
# SECTION 6: LAG_DAY -> 2026 DATE (with validation)
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: MAPPING LAG DAYS TO 2026 DATES")
print("="*60)

feat = pd.read_excel(FEATURE_XLSX)
feat['Date'] = pd.to_datetime(feat['Date'])
print(f"feature file: {feat['Date'].min().date()} → {feat['Date'].max().date()}")

map_rows, no_date, multi_date = [], [], []
for c in penalty_cols:
    if c not in feat.columns:
        no_date.append(c); continue
    hit = feat.loc[feat[c].fillna(0) != 0, 'Date']
    if len(hit) == 0:
        no_date.append(c)
    else:
        if len(hit) > 1:
            multi_date.append((c, [d.date().isoformat() for d in hit]))
        for d in hit:
            map_rows.append({'LAG_DAY': c, date_col: d})

lag_map = pd.DataFrame(map_rows)
print(f"lag-days mapped: {lag_map['LAG_DAY'].nunique()} / {len(penalty_cols)}")
print(f"distinct dates : {lag_map[date_col].nunique()}")

if no_date:
    print(f"\n!! lag-days with NO 2026 date: {no_date}")
    print("   Their LR forecast cannot be placed and will be DROPPED.")
if multi_date:
    print(f"\n!! lag-days spanning multiple dates: {multi_date}")

dup = lag_map[lag_map.duplicated(date_col, keep=False)].sort_values(date_col)
if len(dup):
    print("\n!! dates claimed by more than one lag-day (values will be SUMMED):")
    for d, g in dup.groupby(date_col):
        print(f"   {d.date()}: {sorted(g['LAG_DAY'].tolist())}")

lag_map.to_csv(os.path.join(OUT_DIR, "lag_day_date_map_2026.csv"), index=False)

lr_dated = lr_df.merge(lag_map, on='LAG_DAY', how='inner')
dropped = lr_df[lr_df['LAG_DAY'].isin(no_date)]
if len(dropped):
    print(f"\ndropped rows (unmapped lag-days): {len(dropped):,} "
          f"worth {dropped['LR_FORECAST'].sum()/1e5:,.2f} lacs")

lr_by_date = (
    lr_dated.groupby([group_col, date_col], observed=True)
            .agg(LR_FORECAST=('LR_FORECAST', 'sum'),
                 LAG_DAYS=('LAG_DAY', lambda s: '+'.join(sorted(s))))
            .reset_index()
)
print(f"rows keyed by (series, date): {len(lr_by_date):,}")

# =============================================================================
# SECTION 7: JOIN TO DAILY TFT + BLEND
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: JOINING TFT AND BLENDING")
print("="*60)

tft = pd.read_parquet(TFT_PATH)
tft[date_col] = pd.to_datetime(tft[date_col])
tft = tft.rename(columns={tft_col: 'TFT_FORECAST'})
print(f"TFT rows   : {len(tft):,}")
print(f"TFT range  : {tft[date_col].min().date()} → {tft[date_col].max().date()}")
print(f"TFT series : {tft[group_col].nunique():,}")

# Key overlap check -- catches sanitisation mismatches before they silently
# produce an all-NaN join
lr_keys  = set(lr_by_date[group_col].unique())
tft_keys = set(tft[group_col].unique())
print(f"\nseries in both: {len(lr_keys & tft_keys):,}")
print(f"LR only       : {len(lr_keys - tft_keys):,}")
print(f"TFT only      : {len(tft_keys - lr_keys):,}")
if not (lr_keys & tft_keys):
    print("!! NO overlapping series keys — check SANITIZE_GROUP_KEY.")
    print("   LR sample :", list(lr_keys)[:3])
    print("   TFT sample:", list(tft_keys)[:3])

out = tft.merge(lr_by_date, on=[group_col, date_col], how='left')
has_lr = out['LR_FORECAST'].notna()
print(f"\ndaily rows with an LR value: {has_lr.sum():,} ({has_lr.mean()*100:.1f}%)")

w = BLEND_WEIGHT_LR
out['BLENDED_FORECAST'] = np.where(
    has_lr,
    w * out['LR_FORECAST'].fillna(0) + (1 - w) * out['TFT_FORECAST'],
    out['TFT_FORECAST'],                 # non-festive days: TFT alone
)
out['IS_FESTIVE'] = has_lr

print("\ntotals across Jul-Dec 2026 (lacs):")
print(f"  TFT only : {out['TFT_FORECAST'].sum()/1e5:,.2f}")
print(f"  Blended  : {out['BLENDED_FORECAST'].sum()/1e5:,.2f}")
print("\non festive dates only (lacs):")
print(f"  TFT      : {out.loc[has_lr,'TFT_FORECAST'].sum()/1e5:,.2f}")
print(f"  LR       : {out.loc[has_lr,'LR_FORECAST'].sum()/1e5:,.2f}")
print(f"  Blended  : {out.loc[has_lr,'BLENDED_FORECAST'].sum()/1e5:,.2f}")

# =============================================================================
# SECTION 8: SAVE
# =============================================================================

print("\n" + "="*60)
print("SECTION 8: SAVING")
print("="*60)

cols = [group_col, date_col, 'LAG_DAYS', 'IS_FESTIVE',
        'TFT_FORECAST', 'LR_FORECAST', 'BLENDED_FORECAST']
out = out[cols].sort_values([group_col, date_col])

p = os.path.join(OUT_DIR, "blended_forecast_daily_2026.parquet")
out.to_parquet(p, index=False)
print(f"saved: {p}  ({len(out):,} rows)")

# Excel: long format is ~21.6M rows and Excel caps at 1,048,576 -> go wide
wide_x = out.pivot_table(index=group_col, columns=date_col,
                         values='BLENDED_FORECAST', aggfunc='sum')
wide_x.columns = [c.strftime('%Y-%m-%d') for c in wide_x.columns]
wide_x['TOTAL'] = wide_x.sum(axis=1)
wide_x = wide_x.sort_values('TOTAL', ascending=False).reset_index()

daily_summary = (
    out.groupby([date_col, 'IS_FESTIVE'], observed=True)
       .agg(TFT=('TFT_FORECAST','sum'), LR=('LR_FORECAST','sum'),
            BLENDED=('BLENDED_FORECAST','sum'))
       .reset_index()
)

x = os.path.join(OUT_DIR, "blended_forecast_2026.xlsx")
print(f"writing Excel: {len(wide_x):,} rows x {len(wide_x.columns)} cols ...")
with pd.ExcelWriter(x, engine='openpyxl') as xl:
    wide_x.to_excel(xl, sheet_name='Blended by Series', index=False)
    daily_summary.to_excel(xl, sheet_name='Daily Totals', index=False)
    lag_map.to_excel(xl, sheet_name='Lag Day Map', index=False)
print(f"saved: {x}")
print("\nDone.")
