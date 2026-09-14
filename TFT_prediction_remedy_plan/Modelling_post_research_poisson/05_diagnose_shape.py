"""
Diagnose WHERE the festive peak is landing, in the forecast vs in history.

Answers three questions:
  1. On which calendar dates does the festive peak fall in each year?
     (read from your own D/N/C covariate blocks -- no assumed dates)
  2. Is the forecast peak aligned with the 2026 festival, or with the
     CALENDAR MONTH the peak fell in during training years?
  3. Is the Sep-over / Nov-under pattern a shape error or a level error?

Run:  python 05_diagnose_shape.py
"""

import os, json, glob
import numpy as np
import pandas as pd

MODEL_NAME = "PUT_YOUR_MODEL_NAME_HERE"
BASE       = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
OUT_DIR    = os.path.join(os.getcwd(), "predictions_2026")

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)
time_col, group_col, target_col = ROLES["time_col"], ROLES["group_col"], ROLES["target_col"]


# ---------------------------------------------------------------- 1. festival dates
print("=" * 70)
print("1. WHERE THE FESTIVE ANCHOR FALLS EACH YEAR (from your own covariates)")
print("=" * 70)

cal = pd.read_parquet(os.path.join(BASE, "shared_calendar.parquet"))
cal[time_col] = pd.to_datetime(cal[time_col])

for anchor in ["N", "D", "C"]:
    if anchor not in cal.columns:
        print(f"  '{anchor}' not in calendar, skipping")
        continue
    hits = cal.loc[cal[anchor] != 0, time_col]
    print(f"\n  {anchor} (day 0) falls on:")
    for d in sorted(hits):
        print(f"      {d.date()}  ({d.strftime('%B')})")

print("\n  READ THIS: if the anchor sat in OCTOBER for most training years but")
print("  falls in NOVEMBER for 2026, any month-of-year feature will pull the")
print("  predicted peak into October. That is a calendar artifact, not demand.")


# ---------------------------------------------------------------- 2. forecast shape
print("\n" + "=" * 70)
print("2. FORECAST SHAPE: daily total across the horizon")
print("=" * 70)

pred = pd.read_parquet(os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet"))
pred[time_col] = pd.to_datetime(pred[time_col])

daily = pred.groupby(time_col)["PRED_MEAN"].sum().sort_index()
monthly = daily.resample("MS").sum()

print("\n  Monthly predicted totals:")
for d, v in monthly.items():
    print(f"    {d.strftime('%b %Y')}: {v:>12,.0f}  ({v/1e5:>6.2f} lacs)")
print(f"    {'TOTAL':<8}: {daily.sum():>12,.0f}  ({daily.sum()/1e5:>6.2f} lacs)")

peak_date = daily.idxmax()
print(f"\n  Predicted PEAK DAY: {peak_date.date()} ({peak_date.strftime('%B')}), "
      f"{daily.max():,.0f} units")

# where is the 2026 D anchor?
if "D" in cal.columns:
    d26 = cal.loc[(cal[anchor := "D"] != 0) & (cal[time_col].dt.year == 2026), time_col]
    if len(d26):
        d26 = d26.iloc[0]
        gap = (peak_date - d26).days
        print(f"  2026 'D' anchor   : {d26.date()}")
        print(f"  Peak is {gap:+d} days from the anchor.")
        if abs(gap) > 10:
            print("  >>> PEAK IS MISALIGNED WITH THE FESTIVAL. The model is not")
            print("      following the D/N/C relative-day blocks. Suspect the")
            print("      month-of-year features are overriding them.")
        else:
            print("  >>> Peak is aligned with the festival. The shape error is")
            print("      NOT a timing problem -- look at level anchoring instead.")


# ---------------------------------------------------------------- 3. vs history
print("\n" + "=" * 70)
print("3. SAME-WINDOW ACTUALS FROM PRIOR YEARS (level check)")
print("=" * 70)

files = sorted(glob.glob(os.path.join(BASE, "local_train_scooter_data", "chunk_*.parquet")))
hist = pd.concat(
    [pd.read_parquet(f, columns=[time_col, target_col]) for f in files],
    ignore_index=True,
)
hist[time_col] = pd.to_datetime(hist[time_col])
hist_daily = hist.groupby(time_col)[target_col].sum().sort_index()

print("\n  Sep 1 - Dec 7 actual totals by year:")
for yr in [2023, 2024, 2025]:
    w = hist_daily[(hist_daily.index >= f"{yr}-09-01") & (hist_daily.index <= f"{yr}-12-07")]
    if len(w) == 0:
        continue
    pk = w.idxmax()
    print(f"    {yr}: {w.sum():>12,.0f} ({w.sum()/1e5:>6.2f} lacs) | "
          f"peak {pk.date()} ({pk.strftime('%b')}) = {w.max():,.0f}")
    for m in ["09", "10", "11"]:
        mv = w[w.index.strftime("%m") == m].sum()
        print(f"         {pd.Timestamp(f'{yr}-{m}-01').strftime('%b')}: "
              f"{mv:>10,.0f} ({mv/1e5:.2f} lacs)")

print(f"\n  2026 PREDICTED: {daily.sum():,.0f} ({daily.sum()/1e5:.2f} lacs)")
print("\n  READ THIS: compare the MONTHLY SPLIT, not just the totals. If 2026's")
print("  total is reasonable but Sep/Oct/Nov are distributed differently from")
print("  the years whose Diwali fell in the same month, the problem is shape.")
