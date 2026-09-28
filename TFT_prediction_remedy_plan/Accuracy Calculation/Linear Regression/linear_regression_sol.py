"""
forecast_2026_linear_regression.py
-----------------------------------
Goal
====
For every unique PARENT_DEALER_CODE_MODEL_FAMILY, and for every calendar day
in the Sep-01 -> Dec-10 window, we have up to 3 historical NET_SALES values
(one from 2023, one from 2024, one from 2025) for that exact day-of-year.

We fit a simple linear regression:

        NET_SALES = slope * YEAR + intercept

across those (up to 3) points, and use it to predict NET_SALES for 2026 on
that same calendar day (e.g. Sep-01-2023, Sep-01-2024, Sep-01-2025 ->
predict Sep-01-2026).

Because there can be ~31,893 series x ~101 days = ~3.2 million tiny
regressions, we do NOT loop and call sklearn per group (way too slow).
Instead we reshape the data so each row = one (series, month-day) group
with up to 3 columns (2023/2024/2025), and compute the OLS slope/intercept
for ALL rows at once using vectorized numpy arithmetic (closed-form OLS).

Input
=====
A CSV/text file with (at least) these columns:
    PARENT_DEALER_CODE_MODEL_FAMILY   (string key)
    CAL_DATE                          (string, format DD-MM-YYYY)
    NET_SALES                         (numeric)

Output
======
A CSV with columns:
    PARENT_DEALER_CODE_MODEL_FAMILY
    CAL_DATE            (DD-MM-2026)
    PREDICTED_NET_SALES
    N_HISTORICAL_POINTS (how many of 2023/2024/2025 had data -> quality flag)

Usage
=====
    python forecast_2026_linear_regression.py \\
        --input path/to/your_data.csv \\
        --output path/to/predictions_2026.csv \\
        --sep "\\t"      # or "," depending on your file
"""

import argparse
import sys
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(input_path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=sep, dtype={"PARENT_DEALER_CODE_MODEL_FAMILY": str})

    required_cols = {"PARENT_DEALER_CODE_MODEL_FAMILY", "CAL_DATE", "NET_SALES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Parse dates as DD-MM-YYYY (matches "01-09-2023" == 1st Sep 2023)
    df["CAL_DATE_PARSED"] = pd.to_datetime(df["CAL_DATE"], format="%d-%m-%Y", errors="coerce")

    bad_dates = df["CAL_DATE_PARSED"].isna().sum()
    if bad_dates:
        print(f"[WARN] {bad_dates} rows had unparseable CAL_DATE and will be dropped.", file=sys.stderr)
        df = df.dropna(subset=["CAL_DATE_PARSED"])

    df["YEAR"] = df["CAL_DATE_PARSED"].dt.year
    # month-day key, e.g. "09-01" for Sep 1st -> uniquely identifies the
    # "day slot" we're forecasting across years, regardless of which year.
    df["MONTH_DAY"] = df["CAL_DATE_PARSED"].dt.strftime("%m-%d")

    df["NET_SALES"] = pd.to_numeric(df["NET_SALES"], errors="coerce")

    return df


# --------------------------------------------------------------------------
# 2. RESHAPE: one row per (series, month-day), one column per year
# --------------------------------------------------------------------------
def build_year_matrix(df: pd.DataFrame, hist_years):
    # Collapse any duplicate (series, year, month-day) rows by averaging.
    grouped = (
        df.groupby(["PARENT_DEALER_CODE_MODEL_FAMILY", "MONTH_DAY", "YEAR"])["NET_SALES"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["PARENT_DEALER_CODE_MODEL_FAMILY", "MONTH_DAY"],
        columns="YEAR",
        values="NET_SALES",
    )

    # Make sure all historical year columns exist even if some are entirely absent
    for y in hist_years:
        if y not in pivot.columns:
            pivot[y] = np.nan

    pivot = pivot[hist_years]  # fixed column order: e.g. [2023, 2024, 2025]
    return pivot


# --------------------------------------------------------------------------
# 3. VECTORIZED CLOSED-FORM OLS ACROSS ALL ROWS AT ONCE
# --------------------------------------------------------------------------
def vectorized_linear_forecast(pivot: pd.DataFrame, hist_years, predict_year: int) -> pd.DataFrame:
    Y = pivot.to_numpy(dtype=float)                 # shape (n_rows, n_years)
    x = np.array(hist_years, dtype=float)            # shape (n_years,)

    mask = ~np.isnan(Y)                               # which years actually have data
    n = mask.sum(axis=1)                              # valid point count per row

    Y_filled = np.where(mask, Y, 0.0)
    X_bcast = np.broadcast_to(x, Y.shape)
    X_filled = np.where(mask, X_bcast, 0.0)

    sum_x = X_filled.sum(axis=1)
    sum_y = Y_filled.sum(axis=1)
    sum_xy = (X_filled * Y_filled).sum(axis=1)
    sum_x2 = (X_filled * X_filled).sum(axis=1)

    denom = n * sum_x2 - sum_x ** 2

    slope = np.zeros(len(pivot))
    intercept = np.zeros(len(pivot))

    # Case A: n >= 2 and denom != 0 -> standard closed-form OLS
    valid = (n >= 2) & (denom != 0)
    slope[valid] = (n[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]
    intercept[valid] = (sum_y[valid] - slope[valid] * sum_x[valid]) / n[valid]

    # Case B: exactly 1 valid point -> flat-line forecast = that single value
    one_point = n == 1
    # sum_y / n gives the single non-nan value directly
    single_val = np.divide(sum_y, n, out=np.zeros_like(sum_y), where=n > 0)
    intercept[one_point] = single_val[one_point]
    slope[one_point] = 0.0

    # Case C: n == 0 -> no history at all, leave as NaN
    no_point = n == 0

    prediction = slope * predict_year + intercept
    prediction[no_point] = np.nan

    # Sales can't be negative -> clip
    prediction = np.where(np.isnan(prediction), prediction, np.clip(prediction, 0, None))

    result = pivot.reset_index()[["PARENT_DEALER_CODE_MODEL_FAMILY", "MONTH_DAY"]].copy()
    result["PREDICTED_NET_SALES"] = prediction
    result["N_HISTORICAL_POINTS"] = n.astype(int)

    return result


# --------------------------------------------------------------------------
# 4. BUILD FINAL DATE STRINGS & FILTER TO THE TARGET WINDOW
# --------------------------------------------------------------------------
def finalize_output(result: pd.DataFrame, predict_year: int, window_start: str, window_end: str) -> pd.DataFrame:
    # MONTH_DAY is like "09-01" -> build a real 2026 date to filter & format
    result["CAL_DATE_2026"] = pd.to_datetime(
        str(predict_year) + "-" + result["MONTH_DAY"], format="%Y-%m-%d", errors="coerce"
    )

    start = pd.to_datetime(f"{predict_year}-{window_start}", format="%Y-%m-%d")
    end = pd.to_datetime(f"{predict_year}-{window_end}", format="%Y-%m-%d")
    result = result[(result["CAL_DATE_2026"] >= start) & (result["CAL_DATE_2026"] <= end)]

    result["CAL_DATE"] = result["CAL_DATE_2026"].dt.strftime("%d-%m-%Y")

    out = result[
        ["PARENT_DEALER_CODE_MODEL_FAMILY", "CAL_DATE", "PREDICTED_NET_SALES", "N_HISTORICAL_POINTS"]
    ].sort_values(["PARENT_DEALER_CODE_MODEL_FAMILY", "CAL_DATE"])

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Per-series linear regression forecast for 2026 NET_SALES.")
    parser.add_argument("--input", required=True, help="Path to input data file (CSV/TSV).")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV.")
    parser.add_argument("--sep", default=",", help="Field separator of input file, e.g. ',' or '\\t' (default ',').")
    parser.add_argument("--hist_years", default="2023,2024,2025", help="Comma-separated historical years to fit on.")
    parser.add_argument("--predict_year", type=int, default=2026, help="Year to forecast.")
    parser.add_argument("--window_start", default="09-01", help="Forecast window start, MM-DD.")
    parser.add_argument("--window_end", default="12-10", help="Forecast window end, MM-DD.")
    args = parser.parse_args()

    hist_years = [int(y) for y in args.hist_years.split(",")]

    print(f"[1/4] Loading data from {args.input} ...")
    df = load_data(args.input, args.sep)
    print(f"      -> {len(df):,} rows, {df['PARENT_DEALER_CODE_MODEL_FAMILY'].nunique():,} unique series")

    print("[2/4] Building (series x month-day) x year matrix ...")
    pivot = build_year_matrix(df, hist_years)
    print(f"      -> {len(pivot):,} (series, day) groups to fit")

    print("[3/4] Fitting vectorized linear regression for every group ...")
    result = vectorized_linear_forecast(pivot, hist_years, args.predict_year)

    print("[4/4] Formatting output and writing CSV ...")
    out = finalize_output(result, args.predict_year, args.window_start, args.window_end)
    out.to_csv(args.output, index=False)

    n_no_history = out["PREDICTED_NET_SALES"].isna().sum()
    print(f"      -> Wrote {len(out):,} predictions to {args.output}")
    if n_no_history:
        print(f"      -> NOTE: {n_no_history:,} rows had zero historical data points and are NaN.")


if __name__ == "__main__":
    main()