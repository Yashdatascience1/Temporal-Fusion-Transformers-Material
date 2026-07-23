# =============================================================================
# LR FESTIVE FORECAST -> DATE-KEYED -> BLEND WITH DAILY TFT FORECAST
#
# Pipeline:
#   1. Aggregate historical NET_SALES by (series, lag_day, year)
#   2. Vectorised OLS on 2023-2025 -> extrapolate 2026
#   3. Map each lag_day to its 2026 CAL_DATE via the festive-feature file
#   4. Left-join onto the daily TFT forecast so both are keyed by
#      (series, CAL_DATE)
#   5. Blend on festive dates; TFT alone everywhere else
#
# Scale: 117K series x 44 lag-days = ~5.15M regressions.
# OLS has a closed form, so this is vectorised numpy, not a sklearn loop.
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# SECTION 1: CONFIG
# =============================================================================

DATA_ROOT   = r"C:\Users\G0004878\Desktop\TFT_Data"          # <-- adjust
HIST_PATH   = os.path.join(DATA_ROOT, "festive_daily_data.parquet")   # historical daily + flag cols
FEATURE_XLSX= os.path.join(DATA_ROOT, "2026_festive_feature.xlsx")    # lag_day -> 2026 date
TFT_PATH    = os.path.join(DATA_ROOT, "predictions_jul_dec_2026.parquet")
OUT_DIR     = os.path.join(DATA_ROOT, "lr_blend_output")

group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
year_col   = 'YEAR'
date_col   = 'CAL_DATE'
tft_col    = 'PREDICTED_NET_SALES'      # column name in the TFT parquet

HISTORICAL_YEARS = [2023, 2024, 2025]
FORECAST_YEAR    = 2026

BLEND_WEIGHT_LR = 0.5       # 0.5 = simple average (Sheet4 method)
CLIP_NEGATIVE   = True

penalty_cols = [
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 2: LAG_DAY -> 2026 DATE MAPPING (with validation)
# =============================================================================

print("="*60)
print("SECTION 2: BUILDING LAG_DAY -> CAL_DATE MAP")
print("="*60)

feat = pd.read_excel(FEATURE_XLSX)
feat['Date'] = pd.to_datetime(feat['Date'])
print(f"feature file: {feat['Date'].min().date()} → {feat['Date'].max().date()} "
      f"({len(feat)} rows)")

map_rows = []
no_date, multi_date = [], []
for c in penalty_cols:
    if c not in feat.columns:
        no_date.append(c)
        continue
    hit = feat.loc[feat[c].fillna(0) != 0, 'Date']
    if len(hit) == 0:
        no_date.append(c)
    else:
        if len(hit) > 1:
            multi_date.append((c, [d.date().isoformat() for d in hit]))
        for d in hit:
            map_rows.append({'LAG_DAY': c, date_col: d})

lag_map = pd.DataFrame(map_rows)

print(f"\nlag-days mapped   : {lag_map['LAG_DAY'].nunique()} / {len(penalty_cols)}")
print(f"distinct dates    : {lag_map[date_col].nunique()}")

if no_date:
    print(f"\n!! WARNING - lag-days with NO 2026 date: {no_date}")
    print("   Their LR forecast cannot be placed on a date and will be DROPPED.")

if multi_date:
    print(f"\n!! WARNING - lag-days spanning multiple dates: {multi_date}")

dup = lag_map[lag_map.duplicated(date_col, keep=False)].sort_values(date_col)
if len(dup):
    print(f"\n!! WARNING - dates claimed by more than one lag-day:")
    for d, g in dup.groupby(date_col):
        print(f"   {d.date()}: {sorted(g['LAG_DAY'].tolist())}")
    print("   Their LR values will be SUMMED onto that date.")

lag_map.to_csv(os.path.join(OUT_DIR, "lag_day_date_map_2026.csv"), index=False)

# =============================================================================
# SECTION 3: HISTORICAL -> (series, lag_day, year)
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: AGGREGATING HISTORY BY LAG DAY")
print("="*60)

df = pd.read_parquet(HIST_PATH)
if df[year_col].max() < 100:
    df[year_col] = df[year_col] + 2000

print(f"rows   : {len(df):,}")
print(f"series : {df[group_col].nunique():,}")
print(f"years  : {sorted(df[year_col].unique())}")

usable = [c for c in penalty_cols if c in df.columns]
blocks = []
for i, col in enumerate(usable, 1):
    mask = df[col].fillna(0).to_numpy() != 0
    if not mask.any():
        continue
    agg = (
        df.loc[mask, [group_col, year_col, target_col]]
          .groupby([group_col, year_col], observed=True)[target_col]
          .sum().reset_index()
    )
    agg['LAG_DAY'] = col
    blocks.append(agg)
    if i % 10 == 0 or i == len(usable):
        print(f"  [{i}/{len(usable)}] processed")

long_df = pd.concat(blocks, ignore_index=True)
del blocks, df
print(f"long rows: {len(long_df):,}")

# =============================================================================
# SECTION 4: VECTORISED OLS -> 2026
# =============================================================================
# slope    = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
# forecast = ybar + slope * (2026 - xbar)
# Matches sklearn LinearRegression exactly, for all rows at once.

print("\n" + "="*60)
print("SECTION 4: OLS EXTRAPOLATION")
print("="*60)

wide = (
    long_df.pivot_table(index=[group_col, 'LAG_DAY'], columns=year_col,
                        values=target_col, aggfunc='sum')
           .reindex(columns=HISTORICAL_YEARS)
)

print("coverage per year:")
for y in HISTORICAL_YEARS:
    print(f"  {y}: {wide[y].notna().sum():,} non-null "
          f"({wide[y].notna().mean()*100:.1f}%)")
wide = wide.fillna(0.0)

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

print(f"\npairs            : {len(lr_df):,}")
print(f"negative clipped : {n_neg:,} ({n_neg/len(lr_df)*100:.2f}%)")
print(f"declining pairs  : {(slope<0).sum():,} ({(slope<0).mean()*100:.1f}%)")

hist_tot = {y: lr_df[f'Y{y}'].sum() for y in HISTORICAL_YEARS}
print("\nfestive-window totals (lacs):")
for y in HISTORICAL_YEARS:
    print(f"  {y}      : {hist_tot[y]/1e5:>9,.2f}")
print(f"  {FORECAST_YEAR} (LR): {lr_df['LR_FORECAST'].sum()/1e5:>9,.2f}")
print(f"  implied YoY vs 2025: "
      f"{(lr_df['LR_FORECAST'].sum()/hist_tot[2025]-1)*100:+.1f}%")

# =============================================================================
# SECTION 5: ATTACH DATES
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: ATTACHING 2026 DATES")
print("="*60)

before = len(lr_df)
lr_dated = lr_df.merge(lag_map, on='LAG_DAY', how='inner')
print(f"rows before map : {before:,}")
print(f"rows after map  : {len(lr_dated):,}")
if no_date:
    dropped = before - len(lr_dated[lr_dated['LAG_DAY'].isin(lr_df['LAG_DAY'])].drop_duplicates([group_col,'LAG_DAY']))
    print(f"dropped (unmapped lag-days {no_date}): "
          f"{lr_df[lr_df['LAG_DAY'].isin(no_date)].shape[0]:,} rows")

# Collapse to one row per (series, date). Colliding lag-days sum.
lr_by_date = (
    lr_dated.groupby([group_col, date_col], observed=True)
            .agg(LR_FORECAST=('LR_FORECAST', 'sum'),
                 LAG_DAYS=('LAG_DAY', lambda s: '+'.join(sorted(s))))
            .reset_index()
)
print(f"rows keyed by (series, date): {len(lr_by_date):,}")
print(f"festive dates covered       : {lr_by_date[date_col].nunique()}")

# =============================================================================
# SECTION 6: JOIN TO DAILY TFT + BLEND
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: JOINING TFT AND BLENDING")
print("="*60)

tft = pd.read_parquet(TFT_PATH)
tft[date_col] = pd.to_datetime(tft[date_col])
tft = tft.rename(columns={tft_col: 'TFT_FORECAST'})
print(f"TFT rows   : {len(tft):,}")
print(f"TFT range  : {tft[date_col].min().date()} → {tft[date_col].max().date()}")
print(f"TFT series : {tft[group_col].nunique():,}")

out = tft.merge(lr_by_date, on=[group_col, date_col], how='left')

has_lr = out['LR_FORECAST'].notna()
print(f"\nrows with an LR value: {has_lr.sum():,} "
      f"({has_lr.mean()*100:.1f}% of daily rows)")

w = BLEND_WEIGHT_LR
out['BLENDED_FORECAST'] = np.where(
    has_lr,
    w * out['LR_FORECAST'].fillna(0) + (1 - w) * out['TFT_FORECAST'],
    out['TFT_FORECAST'],                 # non-festive days: TFT alone
)
out['IS_FESTIVE'] = has_lr

print(f"\ntotals across Jul-Dec 2026 (lacs):")
print(f"  TFT only      : {out['TFT_FORECAST'].sum()/1e5:,.2f}")
print(f"  Blended       : {out['BLENDED_FORECAST'].sum()/1e5:,.2f}")
print(f"\non festive dates only (lacs):")
print(f"  TFT     : {out.loc[has_lr,'TFT_FORECAST'].sum()/1e5:,.2f}")
print(f"  LR      : {out.loc[has_lr,'LR_FORECAST'].sum()/1e5:,.2f}")
print(f"  Blended : {out.loc[has_lr,'BLENDED_FORECAST'].sum()/1e5:,.2f}")

# LR pairs that found no matching TFT row
orphan = lr_by_date.merge(tft[[group_col, date_col]], on=[group_col, date_col],
                          how='left', indicator=True)
n_orphan = (orphan['_merge'] == 'left_only').sum()
if n_orphan:
    print(f"\n!! {n_orphan:,} LR rows had no matching TFT (series, date) — not blended.")

# =============================================================================
# SECTION 7: SAVE
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: SAVING")
print("="*60)

cols = [group_col, date_col, 'LAG_DAYS', 'IS_FESTIVE',
        'TFT_FORECAST', 'LR_FORECAST', 'BLENDED_FORECAST']
out = out[cols].sort_values([group_col, date_col])

p = os.path.join(OUT_DIR, "blended_forecast_daily_2026.parquet")
out.to_parquet(p, index=False)
print(f"saved: {p}  ({len(out):,} rows)")

# Excel: long format is ~21.6M rows, Excel caps at 1,048,576 -> go wide.
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
