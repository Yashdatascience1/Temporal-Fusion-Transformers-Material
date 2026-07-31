"""
Diwali / Dhanteras relative features.

Replaces the 10 sparse binaries D-3 .. D+6 with 5 columns.

DESIGN NOTE -- why this differs from the Pitra Paksha treatment
--------------------------------------------------------------
Pitra Paksha is one homogeneous 16-day dip, so a smooth ramp captures it well.
The Diwali cluster is NOT homogeneous: Dhanteras, Lakshmi Puja and the days
around them are distinct events with different -- and for two-wheelers, often
opposite-signed -- effects. Dhanteras is the auspicious vehicle-buying peak;
the days right after are typically a slump. A single smooth curve cannot
represent that, so the named days keep their own binaries here and the
continuous feature only carries position.

Compression is therefore modest (10 -> 5), unlike Pitra Paksha (27 -> 4). That
is the correct trade, not a failure to optimise.
"""

import numpy as np
import pandas as pd

WINDOW = 25        # clip range for days_to_dhanteras
RUNUP_LEN = 14     # length of the pre-Dhanteras booking build-up
DIWALI_OFFSET = 2  # Lakshmi Puja falls this many days after Dhanteras
WIN_LO, WIN_HI = -3, 6   # matches your existing D-3 .. D+6 block


def get_anchors_from_flag(cal_df, time_col="CAL_DATE", flag_col="D"):
    """Derive anchor dates from an existing binary flag column (rows where ==1)."""
    dates = cal_df.loc[cal_df[flag_col] == 1, time_col]
    anchors = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(anchors) == 0:
        raise ValueError(f"no rows with {flag_col}==1 -- check the column name/encoding")
    return anchors


def build_diwali_features(
    dates,
    dhanteras_anchors,
    window=WINDOW,
    runup_len=RUNUP_LEN,
    diwali_offset=DIWALI_OFFSET,
    win_lo=WIN_LO,
    win_hi=WIN_HI,
):
    """
    Parameters
    ----------
    dates             : every date needing covariates (train AND forecast horizon)
    dhanteras_anchors : Dhanteras dates, one per year, covering the same span
    diwali_offset     : days from Dhanteras to Lakshmi Puja (2 in most years --
                        verify against your festive table, it is not invariant)

    Returns
    -------
    DataFrame with columns:
        CAL_DATE
        days_to_dhanteras   signed offset, clipped to +/- window
        dhanteras_runup     0.0 -> 1.0 ramp over the runup_len days BEFORE Dhanteras
        is_dhanteras        1.0 on Dhanteras itself
        is_diwali           1.0 on Lakshmi Puja
        is_diwali_window    1.0 across the whole D-3 .. D+6 block
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    anchors = pd.DatetimeIndex(pd.to_datetime(dhanteras_anchors))

    # signed offset to the NEAREST anchor
    diffs = np.stack([(dates - a).days.to_numpy() for a in anchors])
    nearest = np.argmin(np.abs(diffs), axis=0)
    raw = diffs[nearest, np.arange(len(dates))]

    d2d = np.clip(raw, -window, window).astype(np.float32)

    # build-up: 0.0 at runup_len days out, 1.0 on the eve of Dhanteras
    in_runup = (d2d >= -runup_len) & (d2d < 0)
    runup = np.where(in_runup, (d2d + runup_len) / (runup_len - 1), 0.0).astype(np.float32)

    is_dhanteras = (d2d == 0).astype(np.float32)
    is_diwali = (d2d == diwali_offset).astype(np.float32)
    is_window = ((d2d >= win_lo) & (d2d <= win_hi)).astype(np.float32)

    return pd.DataFrame({
        "CAL_DATE": dates,
        "days_to_dhanteras": d2d,
        "dhanteras_runup": runup,
        "is_dhanteras": is_dhanteras,
        "is_diwali": is_diwali,
        "is_diwali_window": is_window,
    })


# ---------------------------------------------------------------------------
# Integration into Section 4
# ---------------------------------------------------------------------------
#   d_anchors = get_anchors_from_flag(cal, time_col, flag_col="D")
#   diwali_df = build_diwali_features(cal[time_col], d_anchors)
#   cal       = cal.merge(diwali_df, on=time_col, how="left")
#
#   DROP = ["D"] + [f"D-{k}" for k in (1,2,3)] + [f"D+{k}" for k in range(1,7)]
#   NEW  = ["days_to_dhanteras","dhanteras_runup","is_dhanteras",
#           "is_diwali","is_diwali_window"]
#   future_covariates = [c for c in future_covariates if c not in DROP] + NEW


if __name__ == "__main__":
    # demo anchors -- in production derive these from your 'D' column
    dhanteras = pd.to_datetime(["2023-11-10", "2024-10-29", "2025-10-18"])
    cal = pd.date_range("2023-04-01", "2026-12-31", freq="D")

    f = build_diwali_features(cal, dhanteras)

    print("Around Dhanteras 2024 (anchor 2024-10-29):")
    sl = f[(f.CAL_DATE >= "2024-10-20") & (f.CAL_DATE <= "2024-11-06")]
    print(sl.to_string(index=False))

    print("\nDensity check:")
    for c in ["dhanteras_runup", "is_dhanteras", "is_diwali", "is_diwali_window"]:
        print(f"  {c:20s} {(f[c] != 0).mean()*100:5.2f}%")
    print(f"  {'a single D-k flag':20s} {1/365*100:5.2f}%   <- what you have today")
