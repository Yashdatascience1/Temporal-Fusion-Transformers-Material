"""
Pitra Paksha / Navratri relative features.

Replaces the 27 sparse binaries N-16 .. N+10 with 4 dense columns:

    days_to_navratri  signed offset to Navratri day 1, clipped to +/- WINDOW
    is_pitra_paksha   1.0 on the 16 days before Navratri, else 0.0
    pitra_progress    0.0 -> 1.0 ramp across Pitra Paksha, 0.0 outside
    is_amavasya       1.0 on Sarva Pitru Amavasya (the day before Navratri)

All four are date-derived, so they are IDENTICAL across every series and belong
in the shared future-covariate TimeSeries, never in the per-series cache.
"""

import numpy as np
import pandas as pd

WINDOW = 20        # clip range for days_to_navratri
PITRA_LEN = 16     # length of the Pitra Paksha period


def get_navratri_anchors(cal_df, time_col="CAL_DATE", n_col="N"):
    """
    Derive Navratri day-1 dates from your EXISTING 'N' flag column.

    Strongly preferred over hardcoding dates: it stays consistent with whatever
    convention your festive table already uses, and it cannot drift out of sync
    with the flags the model was previously trained on.
    """
    dates = cal_df.loc[cal_df[n_col] == 1, time_col]
    anchors = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(anchors) == 0:
        raise ValueError(f"no rows with {n_col}==1 -- check the column name/encoding")
    return anchors


def build_pitra_features(dates, anchors, window=WINDOW, pitra_len=PITRA_LEN):
    """
    Parameters
    ----------
    dates   : date-like sequence -- every date you need covariates for
              (must span training AND the full forecast horizon)
    anchors : Navratri day-1 dates, one per year, covering the same span

    Returns
    -------
    DataFrame indexed 0..n-1 with columns:
        CAL_DATE, days_to_navratri, is_pitra_paksha, pitra_progress, is_amavasya
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    anchors = pd.DatetimeIndex(pd.to_datetime(anchors))

    # signed offset to the NEAREST anchor (handles year boundaries correctly:
    # January dates resolve to the previous autumn, not the coming one)
    diffs = np.stack([(dates - a).days.to_numpy() for a in anchors])   # (n_anchors, n_dates)
    nearest = np.argmin(np.abs(diffs), axis=0)
    raw = diffs[nearest, np.arange(len(dates))]

    d2n = np.clip(raw, -window, window).astype(np.float32)

    is_pp = ((d2n >= -pitra_len) & (d2n < 0)).astype(np.float32)

    # 0.0 on the first day of Pitra Paksha, 1.0 on Amavasya, 0.0 outside
    progress = np.where(
        is_pp == 1.0,
        (d2n + pitra_len) / (pitra_len - 1),
        0.0,
    ).astype(np.float32)

    is_amavasya = (d2n == -1).astype(np.float32)

    return pd.DataFrame({
        "CAL_DATE": dates,
        "days_to_navratri": d2n,
        "is_pitra_paksha": is_pp,
        "pitra_progress": progress,
        "is_amavasya": is_amavasya,
    })


# ---------------------------------------------------------------------------
# How this slots into your existing Section 4
# ---------------------------------------------------------------------------
#
#   anchors  = get_navratri_anchors(cal, time_col=time_col, n_col="N")
#   pitra_df = build_pitra_features(cal[time_col], anchors)
#   cal      = cal.merge(pitra_df, on=time_col, how="left")
#
#   NEW_COLS = ["days_to_navratri","is_pitra_paksha","pitra_progress","is_amavasya"]
#   DROP     = [f"N{s}{i}" for s,i in
#               [("-",k) for k in range(1,17)] + [("+",k) for k in range(1,11)]] + ["N"]
#
#   future_covariates = [c for c in future_covariates if c not in DROP] + NEW_COLS
#
#   # the festive flag feeding component 1 of the target must be redefined too,
#   # since penalty_cols loses the N block:
#   #   flag = is_pitra_paksha | (abs(days_to_navratri) <= 10) | <D block> | <C block>


if __name__ == "__main__":
    # standalone demo using explicit anchors; in production derive them from 'N'
    anchors = pd.to_datetime(["2023-10-15", "2024-10-03", "2025-09-22"])
    cal = pd.date_range("2023-04-01", "2026-12-31", freq="D")

    f = build_pitra_features(cal, anchors)

    print("Around Navratri 2024 (anchor 2024-10-03):")
    print(f[(f.CAL_DATE >= "2024-09-14") & (f.CAL_DATE <= "2024-10-08")].to_string(index=False))

    print("\nDensity check (fraction of all days non-zero):")
    for c in ["is_pitra_paksha", "pitra_progress", "is_amavasya"]:
        print(f"  {c:18s} {(f[c] != 0).mean()*100:5.2f}%")
    print(f"  {'a single N-k flag':18s} {1/365*100:5.2f}%   <- what you have today")
