"""
Turn the raw inference output into a clean forecast table:
    series | date | predicted_sales

Run:  python 04_prediction_output.py
"""

import os, json
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_NAME = "PUT_YOUR_MODEL_NAME_HERE"

BASE    = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
OUT_DIR = os.path.join(os.getcwd(), "predictions_2026")

# which column to report as the forecast.
#   PRED_MEAN -> the distribution mean, unbiased point forecast
#   PRED_Q60  -> biased upward; use if you want to lean against under-forecasting
FORECAST_COL = "PRED_MEAN"

ROUND_TO_INTEGER = True     # sales are unit counts, not fractions

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)

time_col, group_col = ROLES["time_col"], ROLES["group_col"]


# ---------------- LOAD ----------------
src = os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet")
df = pd.read_parquet(src)
print(f"Loaded {len(df):,} rows from {os.path.basename(src)}")


# ---------------- RESHAPE ----------------
out = df[[group_col, time_col, FORECAST_COL]].copy()
out.columns = ["series", "date", "predicted_sales"]

out["date"] = pd.to_datetime(out["date"])

if ROUND_TO_INTEGER:
    out["predicted_sales"] = out["predicted_sales"].round().astype("int32")

out = out.sort_values(["series", "date"]).reset_index(drop=True)


# ---------------- CHECK ----------------
n_series = out["series"].nunique()
n_dates  = out["date"].nunique()

print(f"\nseries : {n_series:,}")
print(f"dates  : {n_dates} ({out['date'].min().date()} -> {out['date'].max().date()})")
print(f"rows   : {len(out):,}  (expected {n_series * n_dates:,})")

if len(out) != n_series * n_dates:
    print("WARNING: not every series has every date. Some series are incomplete.")

if out["predicted_sales"].isna().any():
    print(f"WARNING: {out['predicted_sales'].isna().sum():,} null predictions.")

print(f"\ntotal predicted: {out['predicted_sales'].sum():,} "
      f"({out['predicted_sales'].sum()/1e5:.2f} lacs)")


# ---------------- SAVE ----------------
dst = os.path.join(OUT_DIR, f"{MODEL_NAME}_forecast.parquet")
out.to_parquet(dst, index=False)
out.to_csv(dst.replace(".parquet", ".csv"), index=False)

print(f"\nWritten -> {dst}")
print(f"          {dst.replace('.parquet', '.csv')}")
print("\nSample:")
print(out.head(10).to_string(index=False))
