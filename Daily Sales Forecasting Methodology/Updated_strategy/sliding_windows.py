"""
Sliding-window arithmetic for Darts-style (input_chunk, output_chunk) training samples.

THE ONE RULE EVERYTHING FOLLOWS
-------------------------------
A training sample is ICL + OCL *contiguous* days. The input chunk is the first
ICL days, the output chunk is the next OCL days. Windows slide one day at a
time, so:

    window i (i = 0 is the OLDEST window):
        input_start  = start_date + i
        input_end    = input_start + ICL - 1
        output_start = input_end + 1
        output_end   = output_start + OCL - 1

    n_windows = n_days - (ICL + OCL) + 1        (0 if negative)

The newest window's output_end always lands exactly on end_date. That is the
sanity check to reach for when a number looks wrong.

WHY `most_recent_only` MATTERS
------------------------------
Darts' `max_samples_per_ts` draws from the most recent past. So capping it does
not thin windows evenly across history -- it *truncates the oldest ones*, which
is how a festival that sits early in your training range can silently stop being
something the model is ever scored on predicting.
"""

from __future__ import annotations

import pandas as pd


def sliding_windows(
    start_date,
    end_date,
    icl: int,
    ocl: int,
    max_samples_per_ts: int | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Enumerate every (input_chunk, output_chunk) window in a series.

    Parameters
    ----------
    start_date, end_date : str or Timestamp
        First and last day of the series, both INCLUSIVE.
    icl, ocl : int
        input_chunk_length and output_chunk_length, in days.
    max_samples_per_ts : int, optional
        Keep only the N most recent windows (mirrors Darts' behaviour).
        None keeps all of them.
    freq : str
        Only 'D' is supported by this implementation.

    Returns
    -------
    DataFrame with one row per window, oldest first, columns:
        window_idx      position from the oldest window (0-based)
        recency_idx     position from the newest window (0 = most recent)
        input_start, input_end, output_start, output_end
        kept            False if dropped by max_samples_per_ts
    """
    if freq != "D":
        raise NotImplementedError("only daily freq is implemented")
    if icl < 1 or ocl < 1:
        raise ValueError("icl and ocl must both be >= 1")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    if end_date < start_date:
        raise ValueError("end_date is before start_date")

    n_days = (end_date - start_date).days + 1
    sample_len = icl + ocl
    n_windows = max(n_days - sample_len + 1, 0)

    if n_windows == 0:
        return pd.DataFrame(
            columns=["window_idx", "recency_idx", "input_start", "input_end",
                     "output_start", "output_end", "kept"]
        )

    idx = pd.RangeIndex(n_windows)
    offset = pd.to_timedelta(idx, unit="D")

    input_start = start_date + offset
    input_end = input_start + pd.Timedelta(days=icl - 1)
    output_start = input_end + pd.Timedelta(days=1)
    output_end = output_start + pd.Timedelta(days=ocl - 1)

    df = pd.DataFrame({
        "window_idx": idx,
        "recency_idx": n_windows - 1 - idx,   # 0 == most recent
        "input_start": input_start,
        "input_end": input_end,
        "output_start": output_start,
        "output_end": output_end,
    })

    if max_samples_per_ts is None:
        df["kept"] = True
    else:
        df["kept"] = df["recency_idx"] < max_samples_per_ts

    # invariant: the newest window must finish exactly on end_date
    assert df["output_end"].iloc[-1] == end_date, "window arithmetic is off"
    return df


def summarise(
    start_date,
    end_date,
    icl: int,
    ocl: int,
    max_samples_per_ts: int | None = None,
    events: dict | None = None,
    label: str = "",
) -> pd.DataFrame:
    """
    Print a human-readable summary and, if `events` is given, report which of
    them are covered by a KEPT output chunk (i.e. actually learnable targets)
    versus only ever visible inside an input chunk (context only).

    events : {name: date-like}, e.g. {"Diwali 2024": "2024-11-01"}
    """
    df = sliding_windows(start_date, end_date, icl, ocl, max_samples_per_ts)
    n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1

    print("=" * 72)
    if label:
        print(label)
    print(f"series      : {pd.Timestamp(start_date).date()} .. {pd.Timestamp(end_date).date()}  ({n_days} days)")
    print(f"icl + ocl   : {icl} + {ocl} = {icl + ocl} days needed per sample")
    print(f"formula     : {n_days} - {icl + ocl} + 1 = {max(n_days - icl - ocl + 1, 0)} windows")

    if df.empty:
        print("RESULT      : series too short -- 0 windows, this series is dropped")
        print("=" * 72)
        return df

    kept = df[df["kept"]]
    print(f"windows     : {len(df)} total, {len(kept)} kept"
          + (f" (max_samples_per_ts={max_samples_per_ts})" if max_samples_per_ts else ""))
    print()
    print(f"  oldest kept window : input {kept.iloc[0].input_start.date()} .. {kept.iloc[0].input_end.date()}"
          f" | output {kept.iloc[0].output_start.date()} .. {kept.iloc[0].output_end.date()}")
    print(f"  newest kept window : input {kept.iloc[-1].input_start.date()} .. {kept.iloc[-1].input_end.date()}"
          f" | output {kept.iloc[-1].output_start.date()} .. {kept.iloc[-1].output_end.date()}")
    print()
    print(f"  TARGET coverage (union of kept output chunks) : "
          f"{kept['output_start'].min().date()} .. {kept['output_end'].max().date()}")
    print(f"  CONTEXT-only span (input chunks reach back to): "
          f"{kept['input_start'].min().date()}")

    if events:
        print()
        print("  event coverage:")
        for name, when in events.items():
            d = pd.Timestamp(when)
            in_target = ((kept["output_start"] <= d) & (d <= kept["output_end"])).sum()
            in_input = ((kept["input_start"] <= d) & (d <= kept["input_end"])).sum()
            if in_target:
                verdict = f"TARGET in {in_target} window(s)  <-- learnable"
            elif in_input:
                verdict = f"context only ({in_input} window(s)) <-- never a target"
            else:
                verdict = "NOT SEEN AT ALL"
            print(f"    {name:<16} {d.date()}  {verdict}")
    print("=" * 72)
    return df


def val_series_bounds(val_start, val_end, icl: int, ocl: int, train_end=None):
    """
    Work out the warm-up slice a validation series needs, and flag the case
    where the validation OUTPUT chunk runs back into the training set.

    A validation series must be at least icl + ocl long, so it needs warm-up
    history before val_start. If (val_end - val_start + 1) < ocl, the output
    chunk is longer than the validation window itself and necessarily spills
    backwards into training data -- that is leakage, and the fix is to make the
    validation window exactly ocl days long.
    """
    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    val_days = (val_end - val_start).days + 1
    warmup_days = icl + ocl - val_days
    warmup_start = val_start - pd.Timedelta(days=warmup_days)

    out_start = warmup_start + pd.Timedelta(days=icl)
    out_end = out_start + pd.Timedelta(days=ocl - 1)

    print("=" * 72)
    print(f"validation window : {val_start.date()} .. {val_end.date()}  ({val_days} days)")
    print(f"ocl               : {ocl} days")
    print(f"warm-up needed    : {warmup_days} days  ->  series starts {warmup_start.date()}")
    print(f"series length     : {warmup_days + val_days} (must equal icl+ocl = {icl + ocl})")
    print(f"val OUTPUT chunk  : {out_start.date()} .. {out_end.date()}")

    overlap = 0
    if train_end is not None:
        train_end = pd.Timestamp(train_end)
        overlap = max((train_end - out_start).days + 1, 0)
        if overlap > 0:
            print(f"LEAKAGE           : {overlap} days of the val output chunk are <= train_end "
                  f"({train_end.date()})")
            print(f"FIX               : set the validation window to exactly {ocl} days")
        else:
            print("LEAKAGE           : none -- val output chunk sits entirely after train_end")
    print("=" * 72)
    return {"warmup_start": warmup_start, "warmup_days": warmup_days,
            "output_start": out_start, "output_end": out_end, "overlap_days": overlap}


if __name__ == "__main__":
    DIWALI = {
        "Diwali 2023": "2023-11-12",
        "Diwali 2024": "2024-11-01",
        "Diwali 2025": "2025-10-20",
    }

    summarise("2023-04-01", "2025-06-30", 365, 184, None, DIWALI,
              label="STAGE 1 (select): train Apr-23 .. Jun-25, all windows")

    summarise("2023-04-01", "2026-06-30", 365, 184, None, DIWALI,
              label="STAGE 2 (refit): train Apr-23 .. Jun-26, all windows")

    summarise("2023-04-01", "2025-12-31", 365, 184, 50, DIWALI,
              label="max_samples_per_ts=50 -- shows how the cap eats old festivals")

    val_series_bounds("2025-08-01", "2025-12-31", 365, 184, train_end="2025-07-31")
    val_series_bounds("2025-07-01", "2025-12-31", 365, 184, train_end="2025-06-30")
