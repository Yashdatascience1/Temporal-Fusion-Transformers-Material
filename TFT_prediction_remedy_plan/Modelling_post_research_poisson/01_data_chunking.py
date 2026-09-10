"""
=============================================================================
STEP 1 OF 2 -- DATA CHUNKING (Snowflake -> local parquet chunks)
=============================================================================

Changes vs the previous version:

  1. Calendar columns (YEAR / MONTH / DAY_OF_THE_MONTH / DAY_OF_THE_WEEK)
     are now PULLED and converted into cyclic future covariates here, in the
     shared calendar, instead of being excluded and re-derived by
     Darts' add_encoders at fit time. Rationale in Section 6 below.

  2. No Scaler(), no add_encoders dict. The model no longer scales its
     target (count likelihood), and the calendar features are built by hand.

  3. NET_SALES is checked for negatives here, because Poisson / Negative
     Binomial likelihoods are undefined on negative support and a silent
     clamp downstream would be invisible.

Run this once. It writes:
    local_train_scooter_data/chunk_NNNN.parquet
    local_test_scooter_data/chunk_NNNN.parquet
    saved_group_keys_scooter/group_keys_scooter.json
    shared_calendar.parquet          <-- NEW: date -> all future covariates
"""

import os
import sys
import json
import glob
import gc

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import snowflake_utils
import Snowflake_configuration

from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import StringType


# =============================================================================
# SECTION 0: CONFIG
# =============================================================================

TABLE_NAME      = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'
SEGMENT_XLSX    = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Modelling_scaled_iteration_3107\Model segmentation.xlsx"

local_train_dir = "./local_train_scooter_data"
local_test_dir  = "./local_test_scooter_data"
group_keys_path = "./saved_group_keys_scooter/group_keys_scooter.json"
calendar_path   = "./shared_calendar.parquet"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
dealer_col = 'PARENT_DEALER_CODE'
target_col = 'NET_SALES'

CHUNK_SIZE = 100
FREQ       = 'D'

# The "train" parquet set must cover everything the modelling script reads as
# history, which includes the validation strip -- so it runs to VAL_END, not to
# TRAIN_END. The modelling script splits train vs val inside that range.
TRAIN_START   = '2023-04-01'
TRAIN_END     = '2026-04-30'
VAL_START     = '2026-05-01'
VAL_END       = '2026-08-31'
FORECAST_START = '2026-09-01'
FORECAST_END   = '2026-12-07'      # 98 days -> OCL = 98

TRAIN_CUTOFF = VAL_END             # last date written to local_train_dir
TEST_START   = FORECAST_START      # first date written to local_test_dir


# =============================================================================
# SECTION 1: SNOWFLAKE SESSION + READ
# =============================================================================

snowflake_conn_prop = Snowflake_configuration.ds1_role_json
session = Session.builder.configs(snowflake_conn_prop).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')


def remove_quotes_existing_columns(df):
    for old_col in df.columns:
        new_col = old_col.replace('"', '')
        df = df.rename(old_col, new_col)
    return df


data = session.table(TABLE_NAME)
data = remove_quotes_existing_columns(data)

# join model segmentation, keep scooters only
model_segment = pd.read_excel(SEGMENT_XLSX)
model_segment_sp = session.create_dataframe(model_segment)
model_segment_sp = remove_quotes_existing_columns(model_segment_sp)

data = data.join(
    model_segment_sp,
    on=(model_segment_sp["Model"] == data["MODEL_FAMILY"]),
    how='inner'
)
data = data.drop("MODEL")

df = data.filter(F.col("CATEGORY") == "Scooter").drop("CATEGORY")

print("Scooter series count:")
df.select(F.count_distinct(group_col).alias("N_SERIES")).show()


# =============================================================================
# SECTION 2: TARGET SANITY CHECK (matters now that we use a count likelihood)
# =============================================================================
# NET_SALES = INVOICED_SALES + CANCELLED_SALES + RETURNED_SALES.
# If cancellations/returns are stored as positive numbers this is wrong in a
# way that inflates volume; if stored as negatives, NET_SALES can go below 0.
# Poisson / NegBin are undefined on negative support, so we need to know the
# scale of the problem BEFORE the modelling script silently clamps it.

print("\n" + "=" * 60)
print("SECTION 2: TARGET SANITY CHECK")
print("=" * 60)

neg_check = df.select(
    F.min(F.col(target_col)).alias("MIN_NET_SALES"),
    F.sum(F.when(F.col(target_col) < 0, 1).otherwise(0)).alias("N_NEGATIVE_ROWS"),
    F.count(F.lit(1)).alias("N_ROWS"),
)
neg_check.show()
print("If N_NEGATIVE_ROWS > 0, the modelling script will clamp those to 0 and")
print("report how many. Decide whether that is acceptable before training.")


# =============================================================================
# SECTION 3: COLUMN ROLES
# =============================================================================

id_cols = [time_col, group_col, target_col]

# calendar columns are handled separately in Section 6 -- they are NOT raw
# future covariates, because raw integer MONTH=12 / MONTH=1 are adjacent in
# reality but 11 apart numerically.
calendar_raw_cols = ['YEAR', 'MONTH', 'DAY_OF_THE_MONTH', 'DAY_OF_THE_WEEK']

static_covariates = [
    f.name for f in df.schema
    if isinstance(f.datatype, StringType)
    and f.name not in ['MODEL_FAMILY_CODE', 'DAY_OF_THE_WEEK', group_col]
]

festive_covariates = [
    c.replace('"', '') for c in df.columns
    if c.replace('"', '') not in static_covariates
    and c.replace('"', '') not in id_cols
    and c.replace('"', '') not in calendar_raw_cols
    and c.replace('"', '') not in ['MODEL_FAMILY_CODE']
]

print("\nstatic_covariates :", static_covariates)
print("\nfestive_covariates:", len(festive_covariates), "columns")


# =============================================================================
# SECTION 4: TRAIN / TEST SPLIT AND GROUP-KEY SANITISATION
# =============================================================================

train_set = df.filter(
    (F.col(time_col) >= TRAIN_START) & (F.col(time_col) <= TRAIN_CUTOFF)
)
test_set = df.filter(
    (F.col(time_col) >= TEST_START) & (F.col(time_col) <= FORECAST_END)
)

print(f"\nTrain parquet range: {TRAIN_START} -> {TRAIN_CUTOFF} "
      f"(covers TRAIN {TRAIN_START}..{TRAIN_END} AND VAL {VAL_START}..{VAL_END})")
print(f"Test  parquet range: {TEST_START} -> {FORECAST_END} (forecast window, 98 days)")
print(f"\nTrain shape: {snowflake_utils.shape_of_snowpark_df(train_set)}")
print(f"Test  shape: {snowflake_utils.shape_of_snowpark_df(test_set)}")


def sanitize(sdf):
    return sdf.with_column(group_col, F.replace(F.col(group_col), F.lit('<>'), F.lit('_')))


train_set = sanitize(train_set)
test_set  = sanitize(test_set)


# =============================================================================
# SECTION 5: DEALER CHUNKS + WRITE PARQUET
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: WRITING PARQUET CHUNKS")
print("=" * 60)

all_dealers = [r[dealer_col] for r in train_set.select(dealer_col).distinct().collect()]
dealer_chunks = [all_dealers[i:i + CHUNK_SIZE] for i in range(0, len(all_dealers), CHUNK_SIZE)]

print(f"Total dealers: {len(all_dealers)} | chunk size: {CHUNK_SIZE} | chunks: {len(dealer_chunks)}")

os.makedirs(os.path.dirname(group_keys_path), exist_ok=True)
group_keys = [r[group_col] for r in train_set.select(group_col).distinct().collect()]
with open(group_keys_path, "w") as f:
    json.dump(group_keys, f)
print(f"Saved {len(group_keys)} group keys -> {group_keys_path}")

os.makedirs(local_train_dir, exist_ok=True)
os.makedirs(local_test_dir, exist_ok=True)


def write_chunks(snowpark_df, chunks, local_dir, label="train"):
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        out_path = os.path.join(local_dir, f"chunk_{idx:04d}.parquet")
        if os.path.exists(out_path):
            print(f"[{label}] chunk {idx+1}/{total} exists, skipping.")
            continue

        print(f"[{label}] pulling chunk {idx+1}/{total} ({len(chunk)} dealers)...")
        chunk_df = snowpark_df.filter(F.col(dealer_col).isin(chunk)).to_pandas()
        chunk_df[group_col] = chunk_df[group_col].str.replace("<>", "_", regex=False)

        # downcast before writing -- keeps the parquet small and the later
        # per-series cache build cheap
        for c in chunk_df.select_dtypes(include=['float64']).columns:
            chunk_df[c] = chunk_df[c].astype(np.float32)
        for c in chunk_df.select_dtypes(include=['int64']).columns:
            chunk_df[c] = pd.to_numeric(chunk_df[c], downcast='integer')

        chunk_df.to_parquet(out_path, index=False)
        print(f"[{label}] chunk {idx+1}/{total} -> {out_path} "
              f"({len(chunk_df):,} rows, {chunk_df[group_col].nunique()} groups)")

        del chunk_df
        gc.collect()

    n = len(glob.glob(os.path.join(local_dir, "chunk_*.parquet")))
    print(f"[{label}] done. {n} chunk files in {local_dir}")


write_chunks(train_set, dealer_chunks, local_train_dir, label="TRAIN")
write_chunks(test_set,  dealer_chunks, local_test_dir,  label="TEST")


# =============================================================================
# SECTION 6: SHARED CALENDAR -- FESTIVE + CYCLIC TIME FEATURES
# =============================================================================
# WHY BUILD THIS BY HAND INSTEAD OF USING add_encoders
# -----------------------------------------------------
# add_encoders={'cyclic': {'future': [...]}} is the idiomatic Darts way and it
# is correct. It is not used here for one specific reason: the modelling
# script hands ONE covariate TimeSeries to all ~116K series by reference
# (SharedCovSequence). Two extra float32 columns on that shared object cost a
# few KB once, for the whole run. add_encoders instead generates covariates
# per series from each series' own time index, which gives up the shared-object
# arrangement the whole DiskLazy design exists to preserve.
#
# Secondary benefit: this calendar is written to disk and reloaded at predict
# time, so training and forecasting provably read the same artifact and it can
# be inspected directly. add_encoders regenerates at predict time -- correct by
# design, but one more thing to trust rather than verify.
#
# WHAT GETS ENCODED AND WHY
# -------------------------
#   DAY_OF_THE_WEEK  -> sin/cos over 7    Sunday and Monday must be adjacent.
#                                          As a raw integer they are 6 apart.
#   MONTH            -> sin/cos over 12   December and January likewise.
#   DAY_OF_THE_MONTH -> sin/cos over 31   plus explicit month-end flags below,
#                                          because the salary-cycle effect is a
#                                          step change, not a smooth cycle.
#   IS_MONTH_END / IS_MONTH_START / DAYS_TO_MONTH_END
#                                          two-wheeler retail has a strong
#                                          month-end push that a smooth cycle
#                                          under-represents.
#   YEAR             -> DELIBERATELY DROPPED. It is a future covariate, so the
#                                          model would receive YEAR=2026 at
#                                          forecast time -- a value it may have
#                                          seen only in a partial year, or an
#                                          unseen value entirely for 2027. A
#                                          model that latches onto it will
#                                          extrapolate off a cliff. Year-level
#                                          drift belongs in an explicit growth
#                                          factor applied after the forecast,
#                                          not in a covariate.

print("\n" + "=" * 60)
print("SECTION 6: BUILDING SHARED CALENDAR")
print("=" * 60)

cal_cols = [time_col] + festive_covariates + calendar_raw_cols

train_chunk0 = sorted(glob.glob(os.path.join(local_train_dir, "chunk_*.parquet")))[0]
test_chunk0  = sorted(glob.glob(os.path.join(local_test_dir,  "chunk_*.parquet")))[0]

cal = pd.concat([
    pd.read_parquet(train_chunk0, columns=cal_cols),
    pd.read_parquet(test_chunk0,  columns=cal_cols),
])
cal[time_col] = pd.to_datetime(cal[time_col])
cal = cal.drop_duplicates(subset=time_col).sort_values(time_col).reset_index(drop=True)

# --- verify the calendar is gap-free; a missing day silently shifts every
# --- downstream relative-day feature by one
full_range = pd.date_range(cal[time_col].min(), cal[time_col].max(), freq=FREQ)
missing = full_range.difference(cal[time_col])
if len(missing):
    raise ValueError(
        f"Calendar has {len(missing)} missing dates, first few: {list(missing[:5])}. "
        "Fix upstream before training -- gaps corrupt the N/D/C relative-day blocks."
    )
print(f"Calendar continuous: {cal[time_col].min().date()} -> {cal[time_col].max().date()} "
      f"({len(cal)} days)")

# the calendar is a FUTURE covariate: it must reach the last forecast date or
# predict() will fail at the very end of the run, after training has completed
_need = pd.Timestamp(FORECAST_END)
if cal[time_col].max() < _need:
    raise ValueError(
        f"Calendar ends {cal[time_col].max().date()} but the forecast needs "
        f"{_need.date()}. Extend {TABLE_NAME} before training."
    )

dt = cal[time_col].dt

# --- cyclic encodings -------------------------------------------------------
def add_cyclic(frame, name, values, period):
    frame[f"{name}_SIN"] = np.sin(2 * np.pi * values / period).astype(np.float32)
    frame[f"{name}_COS"] = np.cos(2 * np.pi * values / period).astype(np.float32)
    return [f"{name}_SIN", f"{name}_COS"]

cyclic_cols = []
# dt.dayofweek is 0=Mon..6=Sun. Use it rather than the table's DAY_OF_THE_WEEK
# string so the encoding is reproducible without a category-order assumption.
cyclic_cols += add_cyclic(cal, "DOW",   dt.dayofweek.to_numpy(),  7)
cyclic_cols += add_cyclic(cal, "MONTH", dt.month.to_numpy(),      12)
cyclic_cols += add_cyclic(cal, "DOM",   dt.day.to_numpy(),        31)

# --- month-boundary features (step effects, not smooth cycles) ---------------
days_in_month = dt.days_in_month.to_numpy()
day_of_month  = dt.day.to_numpy()

cal["IS_MONTH_END"]     = (day_of_month >= days_in_month - 2).astype(np.float32)
cal["IS_MONTH_START"]   = (day_of_month <= 3).astype(np.float32)
cal["DAYS_TO_MONTH_END"] = ((days_in_month - day_of_month) / 31.0).astype(np.float32)

boundary_cols = ["IS_MONTH_END", "IS_MONTH_START", "DAYS_TO_MONTH_END"]

calendar_covariates = cyclic_cols + boundary_cols
future_covariates   = festive_covariates + calendar_covariates

for c in festive_covariates:
    cal[c] = cal[c].astype(np.float32)

cal_out = cal[[time_col] + future_covariates].copy()
cal_out.to_parquet(calendar_path, index=False)

print(f"Calendar written -> {calendar_path}")
print(f"  festive covariates : {len(festive_covariates)}")
print(f"  calendar covariates: {len(calendar_covariates)} -> {calendar_covariates}")
print(f"  TOTAL future covs  : {len(future_covariates)}")
print("  YEAR intentionally excluded (see Section 6 comment).")


# =============================================================================
# SECTION 7: SAVE COLUMN ROLES FOR THE MODELLING SCRIPT
# =============================================================================

roles = {
    "static_covariates":   static_covariates,
    "festive_covariates":  festive_covariates,
    "calendar_covariates": calendar_covariates,
    "future_covariates":   future_covariates,
    "train_start":         TRAIN_START,
    "train_end":           TRAIN_END,
    "val_start":           VAL_START,
    "val_end":             VAL_END,
    "forecast_start":      FORECAST_START,
    "forecast_end":        FORECAST_END,
    "time_col":            time_col,
    "group_col":           group_col,
    "target_col":          target_col,
    "freq":                FREQ,
}
with open("./column_roles.json", "w") as f:
    json.dump(roles, f, indent=2)
print("\nColumn roles written -> ./column_roles.json")

session.close()
print("\nSnowflake session closed.")
