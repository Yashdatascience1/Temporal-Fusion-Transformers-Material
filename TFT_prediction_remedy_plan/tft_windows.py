from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta


def _advance(date, n, unit):
    """Move `date` forward by n units ('D' or 'M'). n may be negative."""
    if unit == 'M':
        return date + relativedelta(months=n)
    elif unit == 'D':
        return date + timedelta(days=n)
    raise ValueError("unit must be 'D' or 'M'")


def get_tft_windows(start_date_str, end_date_str, icl, ocl, unit='M', stride=1):
    """
    Return EVERY TFT input/output window that fits between start_date_str
    and end_date_str. Always returns a DataFrame with one row per window --
    if only one window fits, you get one row; if none fit, you get an empty
    DataFrame (and a printed reason, since that's usually a config mistake
    rather than intent).

    Parameters
    ----------
    start_date_str, end_date_str : 'YYYY-MM-DD'
        The two dates you actually know. For TRAINING this is
        TRAIN_START/TRAIN_END. For VALIDATION under the warmup scheme,
        this is warmup_start/VAL_END (see get_validation_window below --
        VAL_START itself is not one of these two dates; it cancels out).
    icl, ocl : int
        Input / output chunk length, in the given unit.
    unit : 'D' or 'M'
    stride : int
        How many units to slide forward each step.

    Returns
    -------
    pd.DataFrame: WINDOW_IDX, INPUT_START, INPUT_END, OUTPUT_START, OUTPUT_END
    """
    global_start = datetime.strptime(start_date_str, '%Y-%m-%d')
    global_end = datetime.strptime(end_date_str, '%Y-%m-%d')

    current_input_start = global_start
    window_idx = 0
    rows = []

    while True:
        input_end = _advance(current_input_start, icl - 1, unit)
        output_start = _advance(input_end, 1, unit)
        output_end = _advance(current_input_start, icl + ocl - 1, unit)

        if output_end > global_end:
            break

        rows.append({
            "WINDOW_IDX": window_idx,
            "INPUT_START": current_input_start.strftime('%Y-%m-%d'),
            "INPUT_END": input_end.strftime('%Y-%m-%d'),
            "OUTPUT_START": output_start.strftime('%Y-%m-%d'),
            "OUTPUT_END": output_end.strftime('%Y-%m-%d'),
        })

        current_input_start = _advance(current_input_start, stride, unit)
        window_idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        needed = icl + ocl
        have = (global_end - global_start).days if unit == 'D' else None
        print(f"No windows fit: range {start_date_str} to {end_date_str} is shorter "
              f"than icl+ocl={needed} {unit}. Widen the range or shrink icl/ocl.")
    return df


def get_validation_window(val_start_str, val_end_str, icl, ocl, unit='D'):
    """
    Given VAL_START and VAL_END (the two dates as typed into the config),
    return the actual single validation window under the warmup scheme --
    as a one-row DataFrame from get_tft_windows, so it's the same return
    shape as everything else.

    VAL_START only determines the size of the gap (val_window_days), which
    determines warmup_days, which determines how far back warmup_start
    reaches. It is NOT used as a boundary of the final window -- algebraically
    it cancels out, leaving warmup_start = VAL_END - icl - ocl + 1 unit.
    This function does that arithmetic for you and then calls
    get_tft_windows(warmup_start, VAL_END, ...), which will naturally return
    exactly one row.
    """
    val_start = datetime.strptime(val_start_str, '%Y-%m-%d')
    val_end = datetime.strptime(val_end_str, '%Y-%m-%d')

    if unit == 'D':
        val_window = (val_end - val_start).days + 1
    else:
        raise NotImplementedError("Monthly validation warmup not wired up yet -- "
                                   "tell me if you need it.")

    warmup_units = icl + ocl - val_window
    warmup_start = _advance(val_start, -warmup_units, unit)

    return get_tft_windows(warmup_start.strftime('%Y-%m-%d'), val_end_str, icl, ocl, unit=unit)


if __name__ == "__main__":
    # ---- TRAINING: give both known dates, get every window ----
    df_train = get_tft_windows('2023-04-01', '2025-12-01', 16, 2, unit='M')
    print("Training windows:", len(df_train))
    print(df_train.iloc[[0, -1]].to_string(index=False))
    print()

    # ---- VALIDATION: give VAL_START/VAL_END as configured, get the (one) real window ----
    df_val = get_validation_window('2026-05-01', '2026-07-31', 365, 184, unit='D')
    print("Validation windows:", len(df_val))
    print(df_val.to_string(index=False))
    print()

    # ---- sanity: too-short range returns empty, not silently wrong ----
    df_empty = get_tft_windows('2026-05-01', '2026-07-31', 365, 184, unit='D')
    print("Naive slide directly on VAL_START/VAL_END (should be empty):", len(df_empty))
