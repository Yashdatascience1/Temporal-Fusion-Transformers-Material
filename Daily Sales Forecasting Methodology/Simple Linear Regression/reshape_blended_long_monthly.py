# =============================================================================
# RESHAPE BLENDED FORECAST -> LONG (dates as rows) + MONTHLY AGGREGATION
#
# Input : blended_forecast_daily_2026.parquet   (already long, extra columns)
# Output: 1. daily long   -> PARENT_DEALER_CODE_MODEL_FAMILY | DATE | BLENDED_SALES
#         2. monthly long -> PARENT_DEALER_CODE_MODEL_FAMILY | MONTH | BLENDED_SALES
#
# Note on Excel: daily long is ~21.6M rows and Excel caps at 1,048,576, so the
# daily file is parquet/CSV only. Monthly (~703K rows) fits and is written to xlsx.
# =============================================================================

import os
import pandas as pd

# ── config ───────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\G0004878\Desktop\TFT_Data"       # <-- adjust
OUT_DIR   = os.path.join(DATA_ROOT, "lr_blend_output")
SRC       = os.path.join(OUT_DIR, "blended_forecast_daily_2026.parquet")

group_col = 'PARENT_DEALER_CODE_MODEL_FAMILY'
date_col  = 'CAL_DATE'

WRITE_CSV = True        # daily long as CSV too (large file)

# ── 1. daily long ────────────────────────────────────────────────────────────
print("Loading...")
df = pd.read_parquet(SRC)
df[date_col] = pd.to_datetime(df[date_col])

daily = (
    df[[group_col, date_col, 'BLENDED_FORECAST']]
    .rename(columns={date_col: 'DATE', 'BLENDED_FORECAST': 'BLENDED_SALES'})
    .sort_values([group_col, 'DATE'])
    .reset_index(drop=True)
)

print(f"daily rows : {len(daily):,}")
print(f"series     : {daily[group_col].nunique():,}")
print(f"dates      : {daily['DATE'].min().date()} → {daily['DATE'].max().date()}")
print(f"total      : {daily['BLENDED_SALES'].sum()/1e5:,.2f} lacs")

p_daily = os.path.join(OUT_DIR, "blended_daily_long_2026.parquet")
daily.to_parquet(p_daily, index=False)
print(f"saved: {p_daily}")

if WRITE_CSV:
    c_daily = os.path.join(OUT_DIR, "blended_daily_long_2026.csv")
    daily.to_csv(c_daily, index=False)
    print(f"saved: {c_daily}  ({os.path.getsize(c_daily)/1e6:,.0f} MB)")

# ── 2. monthly aggregation ───────────────────────────────────────────────────
daily['MONTH']       = daily['DATE'].dt.to_period('M').dt.to_timestamp()
daily['MONTH_LABEL'] = daily['DATE'].dt.strftime('%Y-%b')

monthly = (
    daily.groupby([group_col, 'MONTH', 'MONTH_LABEL'], observed=True)['BLENDED_SALES']
         .sum()
         .reset_index()
         .sort_values([group_col, 'MONTH'])
         .reset_index(drop=True)
)

print(f"\nmonthly rows: {len(monthly):,}")

p_month = os.path.join(OUT_DIR, "blended_monthly_2026.parquet")
monthly.to_parquet(p_month, index=False)
print(f"saved: {p_month}")

# grand total by month
totals = (
    monthly.groupby(['MONTH', 'MONTH_LABEL'], observed=True)['BLENDED_SALES']
           .sum().reset_index().sort_values('MONTH')
)
totals['LACS'] = totals['BLENDED_SALES'] / 1e5
print("\nmonthly grand totals:")
for _, r in totals.iterrows():
    print(f"  {r['MONTH_LABEL']}: {r['LACS']:>10,.2f} lacs")
print(f"  {'TOTAL':>8}: {totals['LACS'].sum():>10,.2f} lacs")

# ── 3. Excel: monthly long + wide + totals ───────────────────────────────────
month_order = totals['MONTH_LABEL'].tolist()

wide = (
    monthly.pivot_table(index=group_col, columns='MONTH_LABEL',
                        values='BLENDED_SALES', aggfunc='sum', fill_value=0)
           .reindex(columns=month_order)
           .reset_index()
)
wide['TOTAL'] = wide[month_order].sum(axis=1)
wide = wide.sort_values('TOTAL', ascending=False)

x = os.path.join(OUT_DIR, "blended_monthly_2026.xlsx")
print(f"\nwriting Excel ({len(monthly):,} long rows, {len(wide):,} wide rows)...")

try:
    writer_kw = dict(engine='xlsxwriter',
                     engine_kwargs={'options': {'constant_memory': True}})
    with pd.ExcelWriter(x, **writer_kw) as xl:
        monthly[[group_col, 'MONTH_LABEL', 'BLENDED_SALES']].to_excel(
            xl, sheet_name='Monthly Long', index=False)
        wide.to_excel(xl, sheet_name='Monthly Wide', index=False)
        totals.to_excel(xl, sheet_name='Grand Total', index=False)
except ImportError:
    with pd.ExcelWriter(x, engine='openpyxl') as xl:
        monthly[[group_col, 'MONTH_LABEL', 'BLENDED_SALES']].to_excel(
            xl, sheet_name='Monthly Long', index=False)
        wide.to_excel(xl, sheet_name='Monthly Wide', index=False)
        totals.to_excel(xl, sheet_name='Grand Total', index=False)

print(f"saved: {x}")
print("\nDone.")
