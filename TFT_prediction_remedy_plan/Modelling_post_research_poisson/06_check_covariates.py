"""
Confirm the festive covariates actually reach the model for the forecast
window. If they are all zero across Sep-Dec 2026, every other diagnosis is
irrelevant -- you have a plumbing bug, not a modelling problem.

Checks the SAME artifact inference uses (shared_cov.pkl), not just the
source parquet, because those can diverge.

Run:  python 06_check_covariates.py
"""

import os, json
import numpy as np
import pandas as pd
from darts import TimeSeries

BASE      = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
CACHE_DIR = os.path.join(os.getcwd(), "series_cache_scooter")

FC_START = pd.Timestamp("2026-09-01")
FC_END   = pd.Timestamp("2026-12-07")

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)
time_col = ROLES["time_col"]

N_BLOCK = [f"N-{i}" for i in range(16, 0, -1)] + ["N"] + [f"N+{i}" for i in range(1, 11)]
D_BLOCK = [f"D-{i}" for i in range(3, 0, -1)] + ["D"] + [f"D+{i}" for i in range(1, 7)]


def frame_from_pickle(path):
    ts = TimeSeries.from_pickle(path)
    df = ts.to_dataframe().reset_index()
    df.columns = [time_col] + list(ts.components)
    return df


# ------------------------------------------------------------------ load both
sources = {}

pkl = os.path.join(CACHE_DIR, "shared_cov.pkl")
if os.path.exists(pkl):
    sources["shared_cov.pkl (what inference uses)"] = frame_from_pickle(pkl)
else:
    print("shared_cov.pkl NOT FOUND -- inference would have rebuilt from parquet.")

pq = os.path.join(BASE, "shared_calendar.parquet")
if os.path.exists(pq):
    d = pd.read_parquet(pq)
    d[time_col] = pd.to_datetime(d[time_col])
    sources["shared_calendar.parquet (source)"] = d


for label, cal in sources.items():
    print("=" * 70)
    print(label)
    print("=" * 70)

    cal[time_col] = pd.to_datetime(cal[time_col])
    print(f"Date range : {cal[time_col].min().date()} -> {cal[time_col].max().date()}")
    print(f"Columns    : {len(cal.columns) - 1}")

    if cal[time_col].max() < FC_END:
        print(f"\n*** FAIL: calendar ends before {FC_END.date()}. "
              f"Covariates are missing for part of the forecast. ***")

    win = cal[(cal[time_col] >= FC_START) & (cal[time_col] <= FC_END)]
    print(f"Rows in Sep 1 - Dec 7 2026: {len(win)} (expected 98)")

    if len(win) == 0:
        print("\n*** FAIL: no calendar rows in the forecast window at all. ***\n")
        continue

    # --- NaN check: NaNs silently poison the whole forecast ------------------
    nan_cols = [c for c in win.columns if c != time_col and win[c].isna().any()]
    if nan_cols:
        print(f"\n*** FAIL: {len(nan_cols)} columns contain NaN in the window: "
              f"{nan_cols[:8]} ***")
    else:
        print("NaN check : clean")

    # --- the festive blocks --------------------------------------------------
    for name, block, expect in [("N block", N_BLOCK, 27), ("D block", D_BLOCK, 10)]:
        present = [c for c in block if c in win.columns]
        missing = [c for c in block if c not in win.columns]

        print(f"\n{name}: {len(present)}/{len(block)} columns present")
        if missing:
            print(f"  MISSING COLUMNS: {missing}")

        live = {c: int((win[c] != 0).sum()) for c in present}
        n_live = sum(1 for v in live.values() if v > 0)
        print(f"  columns with a non-zero day in the window: {n_live}/{expect}")

        if n_live == 0:
            print("  *** FAIL: the entire block is zero across the forecast window. ***")
            print("      The model is flying blind on this festival. This is the bug.")
        elif n_live < expect:
            dead = [c for c, v in live.items() if v == 0]
            print(f"  *** PARTIAL: these are all-zero: {dead}")
        else:
            print("  OK: every column fires at least once.")

        # each flag should fire on exactly one day
        multi = {c: v for c, v in live.items() if v > 1}
        if multi:
            print(f"  NOTE: fire on >1 day (expected 1 each): {multi}")

    # --- where does each anchor land -----------------------------------------
    for anchor in ["N", "D"]:
        if anchor in win.columns:
            hits = win.loc[win[anchor] != 0, time_col]
            print(f"\n'{anchor}' day-0 in window: "
                  f"{[d.date().isoformat() for d in hits] or 'NONE'}")

    # --- days with no festive signal at all ----------------------------------
    fest = [c for c in (N_BLOCK + D_BLOCK) if c in win.columns]
    if fest:
        blank = win.loc[(win[fest] == 0).all(axis=1), time_col]
        print(f"\nDays in the window with NO festive flag: {len(blank)}/98")
        if len(blank):
            print(f"  first: {blank.min().date()}   last: {blank.max().date()}")
            sep = blank[blank.dt.month == 9]
            print(f"  of those, {len(sep)} fall in September "
                  f"({sep.min().date() if len(sep) else '-'} to "
                  f"{sep.max().date() if len(sep) else '-'})")
            print("  On these days the model has no festive signal and can only")
            print("  extrapolate the baseline from the encoder.")
    print()


# ------------------------------------------------------------------ compare
if len(sources) == 2:
    print("=" * 70)
    print("DO THE TWO ARTIFACTS AGREE?")
    print("=" * 70)
    a, b = list(sources.values())
    ca = set(a.columns) - {time_col}
    cb = set(b.columns) - {time_col}
    if ca != cb:
        print(f"*** Column sets differ. only in pkl: {sorted(ca-cb)[:8]} | "
              f"only in parquet: {sorted(cb-ca)[:8]} ***")
    else:
        wa = a[(a[time_col] >= FC_START) & (a[time_col] <= FC_END)].set_index(time_col).sort_index()
        wb = b[(b[time_col] >= FC_START) & (b[time_col] <= FC_END)].set_index(time_col).sort_index()
        cols = sorted(ca)
        same = np.allclose(wa[cols].to_numpy(float), wb[cols].to_numpy(float), equal_nan=True)
        print("Values identical in the forecast window:", same)
        if not same:
            print("*** The pickle inference uses differs from the source parquet.")
            print("    Rebuild shared_cov.pkl. ***")
