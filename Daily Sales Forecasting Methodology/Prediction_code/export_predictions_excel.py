# =============================================================================
# EXPORT PREDICTIONS TO EXCEL
#
# Input : predictions_jul_dec_2026.parquet  (long format, ~21.6M rows)
# Output: Excel files, shaped to fit Excel's limits
#
# Excel hard limits:  1,048,576 rows  |  16,384 columns  per sheet
# Long format is 21.6M rows -> impossible. So:
#     series as ROWS (117K, fits), dates as COLUMNS (184, fits)
#
# MODES:
#   "monthly"     -> 1 file.  Series x 6 months. Opens instantly. Best for review.
#   "dealer"      -> 1 file.  Dealer x 184 days. ~1,176 rows. Best for management.
#   "daily_split" -> N files. Series x 184 days, split so each file opens.
#   "csv"         -> 1 file.  Full daily long format. No row limit, no formatting.
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT  = r"C:\Users\G0004878\Desktop\TFT_Data"     # <-- adjust
PRED_PATH  = os.path.join(DATA_ROOT, "predictions_jul_dec_2026.parquet")
OUT_DIR    = os.path.join(DATA_ROOT, "excel_output")

MODE = "monthly"          # "monthly" | "dealer" | "daily_split" | "csv"

SERIES_PER_FILE = 15000   # only used by "daily_split"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
value_col  = 'PREDICTED_NET_SALES'

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD
# =============================================================================

print("Loading predictions...")
df = pd.read_parquet(PRED_PATH)
df[time_col] = pd.to_datetime(df[time_col])

print(f"  rows   : {len(df):,}")
print(f"  series : {df[group_col].nunique():,}")
print(f"  dates  : {df[time_col].min().date()} → {df[time_col].max().date()}")
print(f"  total  : {df[value_col].sum()/1e5:,.2f} lacs")

# Dealer code is the part of the key before the first underscore-separated
# model family. Adjust the split if your key format differs.
df['PARENT_DEALER_CODE'] = df[group_col].str.split('_').str[0]

# =============================================================================
# MODE: monthly  — series x months
# =============================================================================

if MODE == "monthly":
    print("\nBuilding monthly summary...")

    df['MONTH'] = df[time_col].dt.strftime('%Y-%b')
    month_order = (
        df[[time_col, 'MONTH']].drop_duplicates()
        .sort_values(time_col)['MONTH'].unique().tolist()
    )

    wide = (
        df.pivot_table(index=group_col, columns='MONTH',
                       values=value_col, aggfunc='sum', fill_value=0)
        .reindex(columns=month_order)
        .reset_index()
    )
    wide['TOTAL'] = wide[month_order].sum(axis=1)
    wide = wide.sort_values('TOTAL', ascending=False)

    out = os.path.join(OUT_DIR, "predictions_monthly_by_series.xlsx")
    print(f"Writing {len(wide):,} rows x {len(wide.columns)} cols ...")

    with pd.ExcelWriter(out, engine='openpyxl') as xl:
        wide.to_excel(xl, sheet_name='Monthly by Series', index=False)

        # Grand total row per month, as its own sheet
        totals = pd.DataFrame({
            'MONTH': month_order,
            'PREDICTED_NET_SALES': [df.loc[df['MONTH'] == m, value_col].sum()
                                    for m in month_order],
        })
        totals['LACS'] = totals['PREDICTED_NET_SALES'] / 1e5
        totals.to_excel(xl, sheet_name='Grand Total', index=False)

    print(f"Saved: {out}")

# =============================================================================
# MODE: dealer  — dealer x daily
# =============================================================================

elif MODE == "dealer":
    print("\nBuilding dealer-level daily...")

    wide = (
        df.pivot_table(index='PARENT_DEALER_CODE', columns=time_col,
                       values=value_col, aggfunc='sum', fill_value=0)
    )
    wide.columns = [c.strftime('%Y-%m-%d') for c in wide.columns]
    wide = wide.reset_index()
    wide['TOTAL'] = wide.iloc[:, 1:].sum(axis=1)
    wide = wide.sort_values('TOTAL', ascending=False)

    out = os.path.join(OUT_DIR, "predictions_daily_by_dealer.xlsx")
    print(f"Writing {len(wide):,} rows x {len(wide.columns)} cols ...")

    with pd.ExcelWriter(out, engine='openpyxl') as xl:
        wide.to_excel(xl, sheet_name='Daily by Dealer', index=False)

    print(f"Saved: {out}")

# =============================================================================
# MODE: daily_split  — series x daily, across several files
# =============================================================================

elif MODE == "daily_split":
    print("\nBuilding daily wide format (split across files)...")

    keys = sorted(df[group_col].unique())
    n_files = int(np.ceil(len(keys) / SERIES_PER_FILE))
    print(f"{len(keys):,} series -> {n_files} files of up to {SERIES_PER_FILE:,}")

    for fi in range(n_files):
        chunk_keys = keys[fi*SERIES_PER_FILE : (fi+1)*SERIES_PER_FILE]
        sub = df[df[group_col].isin(chunk_keys)]

        wide = sub.pivot_table(index=group_col, columns=time_col,
                               values=value_col, aggfunc='sum', fill_value=0)
        wide.columns = [c.strftime('%Y-%m-%d') for c in wide.columns]
        wide = wide.reset_index()

        out = os.path.join(OUT_DIR, f"predictions_daily_part_{fi+1:02d}.xlsx")
        print(f"  part {fi+1}/{n_files}: {len(wide):,} rows -> writing ...")
        with pd.ExcelWriter(out, engine='openpyxl') as xl:
            wide.to_excel(xl, sheet_name='Daily Predictions', index=False)
        print(f"  saved: {out}")

        del sub, wide

    print(f"\nAll {n_files} files written to {OUT_DIR}")

# =============================================================================
# MODE: csv  — everything, long format
# =============================================================================

elif MODE == "csv":
    out = os.path.join(OUT_DIR, "predictions_daily_full.csv")
    print(f"\nWriting {len(df):,} rows to CSV ...")
    df[[group_col, time_col, value_col]].to_csv(out, index=False)
    size_mb = os.path.getsize(out) / 1e6
    print(f"Saved: {out}  ({size_mb:,.0f} MB)")
    print("Note: too many rows for Excel. Use Power Query, pandas, or a database.")

else:
    raise ValueError(f"Unknown MODE: {MODE}")

print("\nDone.")
