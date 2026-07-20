# =============================================================================
# FULL PIPELINE: Snowflake → Local Per-Group Parquet → Darts TimeSeries
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
from snowflake.snowpark.functions import col, trim, split, lit
from snowflake.snowpark.functions import col, sum as _sum, when, is_null
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import StringType
from snowflake.snowpark.session import Session
import sys
sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import snowflake_utils
import Snowflake_configuration

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel

import pandas as pd
import numpy as np
import os, shutil, json, glob, math, gc
import collections.abc

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

snowflake_conn_prop = Snowflake_configuration.ds1_role_json
session = Session.builder.configs(snowflake_conn_prop).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')

TABLE_NAME       = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'
local_train_dir  = "./local_train_data"
local_test_dir   = "./local_test_data"
group_keys_path  = "./saved_group_keys/group_keys.json"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
dealer_col = 'PARENT_DEALER_CODE'
target_col = 'NET_SALES'
CHUNK_SIZE = 100   # dealers per Snowflake pull — safe for 64 GB RAM
FREQ       = 'D'

# =============================================================================
# SECTION 2: LOAD & SANITIZE DATA
# =============================================================================

data = session.table(TABLE_NAME)

id_cols = ['CAL_DATE', 'PARENT_DEALER_CODE_MODEL_FAMILY', 'NET_SALES']

static_covariates = [
    field.name for field in data.schema
    if isinstance(field.datatype, StringType)
    and field.name not in ['MODEL_FAMILY_CODE', 'DAY_OF_THE_WEEK', 'PARENT_DEALER_CODE_MODEL_FAMILY']
]

future_covariates = [
    c for c in data.columns
    if c not in static_covariates
    and c not in id_cols
    and c not in ['MODEL_FAMILY_CODE', 'DAY_OF_THE_WEEK', 'YEAR', 'MONTH', 'DAY_OF_THE_MONTH']
]

print(f"Static covariates : {static_covariates}")
print(f"Future covariates : {future_covariates}")

# Train / test split
train_set = data.filter(F.col("CAL_DATE") <= '2026-06-30')
test_set  = data.filter(F.col("CAL_DATE") >= '2026-07-01')

# Sanitize group key — replace <> with _
def sanitize(df):
    return df.with_column(
        group_col,
        F.replace(F.col(group_col), F.lit('<>'), F.lit('_'))
    )

train_set = sanitize(train_set)
test_set  = sanitize(test_set)

# =============================================================================
# SECTION 3: MEMORY CHECK — one chunk of CHUNK_SIZE dealers
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: MEMORY CHECK")
print("="*60)

all_dealers = [
    row[dealer_col]
    for row in train_set.select(dealer_col).distinct().collect()
]
print(f"Total unique dealers : {len(all_dealers)}")

dealer_chunks = [
    all_dealers[i:i + CHUNK_SIZE]
    for i in range(0, len(all_dealers), CHUNK_SIZE)
]
print(f"Chunks of {CHUNK_SIZE}        : {len(dealer_chunks)}")

print(f"\nPulling chunk 0 ({len(dealer_chunks[0])} dealers) to estimate RAM...")
sample_df = (
    train_set
    .filter(F.col(dealer_col).isin(dealer_chunks[0]))
    .to_pandas()
)

chunk_ram_gb = sample_df.memory_usage(deep=True).sum() / 1e9
n_series     = sample_df[group_col].nunique()

print(f"RAM for chunk 0          : {chunk_ram_gb:.2f} GB")
print(f"Series in chunk 0        : {n_series}")
print(f"Estimated total series   : {n_series * len(dealer_chunks):,}")
print(f"Estimated total RAM      : {chunk_ram_gb * len(dealer_chunks):.2f} GB  (never all in RAM at once)")

del sample_df
gc.collect()

# =============================================================================
# SECTION 4: DOWNLOAD — Snowflake → per-group parquet files on local disk
# =============================================================================
# One dealer chunk pulled at a time → split by group → saved as individual
# parquet files. Peak RAM = one chunk at a time. HDD holds all 117K files.

print("\n" + "="*60)
print("SECTION 4: DOWNLOADING TO LOCAL STORAGE")
print("="*60)

os.makedirs(local_train_dir, exist_ok=True)
os.makedirs(local_test_dir, exist_ok=True)

def safe_filename(key):
    """Convert group key to a safe filename."""
    return str(key).replace("/", "_").replace("\\", "_").replace(":", "_")

def download_and_partition(snowpark_df, dealer_chunks, dealer_col,
                            group_col, time_col, local_dir, label="train"):
    total = len(dealer_chunks)
    total_saved = 0
    total_skipped = 0

    for idx, chunk in enumerate(dealer_chunks):
        print(f"[{label}] Chunk {idx+1}/{total} — pulling {len(chunk)} dealers...")

        chunk_df = (
            snowpark_df
            .filter(F.col(dealer_col).isin(chunk))
            .to_pandas()
        )
        chunk_df[time_col] = pd.to_datetime(chunk_df[time_col])
        chunk_df = chunk_df.sort_values([group_col, time_col]).reset_index(drop=True)

        saved = skipped = 0
        for key, group_df in chunk_df.groupby(group_col):
            out_path = os.path.join(local_dir, f"group_{safe_filename(key)}.parquet")
            if os.path.exists(out_path):
                skipped += 1
                continue
            group_df.reset_index(drop=True).to_parquet(out_path, index=False)
            saved += 1

        total_saved   += saved
        total_skipped += skipped
        print(f"  → Saved: {saved} | Skipped (exist): {skipped}")

        del chunk_df
        gc.collect()

    print(f"\n[{label}] Done. Saved: {total_saved} | Skipped: {total_skipped}")
    print(f"[{label}] Total files on disk: "
          f"{len(glob.glob(os.path.join(local_dir, 'group_*.parquet')))}")

download_and_partition(train_set, dealer_chunks, dealer_col,
                       group_col, time_col, local_train_dir, label="TRAIN")
download_and_partition(test_set,  dealer_chunks, dealer_col,
                       group_col, time_col, local_test_dir,  label="TEST")

# =============================================================================
# SECTION 5: SAVE GROUP KEYS
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: SAVING GROUP KEYS")
print("="*60)

os.makedirs(os.path.dirname(group_keys_path), exist_ok=True)

group_keys = [
    row[group_col]
    for row in train_set.select(group_col).distinct().collect()
]

with open(group_keys_path, "w") as f:
    json.dump(group_keys, f)

print(f"Saved {len(group_keys)} group keys → {group_keys_path}")

session.close()
print("Snowflake session closed.")

# =============================================================================
# SECTION 6: DiskLazyTimeSeriesSequence
# =============================================================================
# Darts calls ts[0] to type-check, len(ts) for sizing, then indexes into the
# sequence once per series to build its training dataset. This class satisfies
# all three. Disk reads happen once at dataset construction — not per epoch.

print("\n" + "="*60)
print("SECTION 6: DISK-LAZY TIMESERIES SEQUENCE")
print("="*60)

class DiskLazyTimeSeriesSequence(collections.abc.Sequence):
    """
    A Sequence of TimeSeries that reads per-group parquet files from disk
    on demand. Darts' fit() indexes into this once per series to build its
    internal training dataset — after that, training runs purely in memory.

    Parameters
    ----------
    data_dir        : folder containing group_{key}.parquet files
    group_keys      : ordered list of group key strings (matches filenames)
    time_col        : datetime column name
    target_col      : target value column (str or list of str)
    future_cov_cols : list of future covariate column names
    static_cov_cols : list of static covariate column names (one row per series)
    freq            : pandas frequency string e.g. 'D'
    """

    def __init__(self, data_dir, group_keys, time_col, target_col,
                 future_cov_cols=None, static_cov_cols=None, freq='D'):
        self.data_dir        = data_dir
        self.group_keys      = group_keys
        self.time_col        = time_col
        self.target_col      = target_col
        self.future_cov_cols = future_cov_cols or []
        self.static_cov_cols = static_cov_cols or []
        self.freq            = freq

    def __len__(self):
        return len(self.group_keys)

    def _load(self, group_key):
        safe_key = safe_filename(group_key)
        path = os.path.join(self.data_dir, f"group_{safe_key}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing parquet for group: {group_key} → {path}")
        df = pd.read_parquet(path)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        return df.sort_values(self.time_col).reset_index(drop=True)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Darts occasionally slices — return a plain list
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for {len(self)} series")

        df = self._load(self.group_keys[idx])

        static_df = None
        if self.static_cov_cols:
            static_df = df[self.static_cov_cols].iloc[[0]].reset_index(drop=True)

        ts = TimeSeries.from_dataframe(
            df,
            time_col=self.time_col,
            value_cols=self.target_col,
            static_covariates=static_df,
            freq=self.freq,
            fill_missing_dates=False
        )
        return ts

    def get_future_covariates(self, idx):
        """Call this separately to get the matching future covariate series."""
        if idx < 0:
            idx = len(self) + idx
        df = self._load(self.group_keys[idx])
        return TimeSeries.from_dataframe(
            df,
            time_col=self.time_col,
            value_cols=self.future_cov_cols,
            freq=self.freq,
            fill_missing_dates=False
        )


class DiskLazyFutureCovSequence(collections.abc.Sequence):
    """
    Parallel sequence to DiskLazyTimeSeriesSequence for future covariates.
    Pass this as future_covariates= in TFTModel.fit().
    Must be indexed in the same order as DiskLazyTimeSeriesSequence.
    """

    def __init__(self, data_dir, group_keys, time_col,
                 future_cov_cols, freq='D'):
        self.data_dir        = data_dir
        self.group_keys      = group_keys
        self.time_col        = time_col
        self.future_cov_cols = future_cov_cols
        self.freq            = freq

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range")

        safe_key = safe_filename(self.group_keys[idx])
        path = os.path.join(self.data_dir, f"group_{safe_key}.parquet")
        df = pd.read_parquet(path)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        return TimeSeries.from_dataframe(
            df,
            time_col=self.time_col,
            value_cols=self.future_cov_cols,
            freq=self.freq,
            fill_missing_dates=False
        )


# =============================================================================
# SECTION 7: INSTANTIATE SEQUENCES & SANITY CHECK
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: INSTANTIATE & SANITY CHECK")
print("="*60)

with open(group_keys_path, "r") as f:
    group_keys = json.load(f)

print(f"Loaded {len(group_keys)} group keys")

train_target_seq = DiskLazyTimeSeriesSequence(
    data_dir        = local_train_dir,
    group_keys      = group_keys,
    time_col        = time_col,
    target_col      = target_col,
    future_cov_cols = future_covariates,
    static_cov_cols = static_covariates,
    freq            = FREQ
)

train_future_seq = DiskLazyFutureCovSequence(
    data_dir        = local_train_dir,
    group_keys      = group_keys,
    time_col        = time_col,
    future_cov_cols = future_covariates,
    freq            = FREQ
)

# Quick sanity check — read first and last series
print("\nSanity check — series[0]:")
ts0 = train_target_seq[0]
fc0 = train_future_seq[0]
print(f"  Target   : {ts0.start_time()} → {ts0.end_time()} | len={len(ts0)}")
print(f"  Future   : {fc0.start_time()} → {fc0.end_time()} | cols={fc0.components.tolist()}")
print(f"  Static   : {ts0.static_covariates}")

print(f"\nSanity check — series[-1]:")
ts_last = train_target_seq[-1]
print(f"  Target   : {ts_last.start_time()} → {ts_last.end_time()} | len={len(ts_last)}")

print(f"\nTotal series in sequence : {len(train_target_seq)}")
print("\nPipeline complete. Pass train_target_seq and train_future_seq to TFTModel.fit().")

# =============================================================================
# USAGE IN TFT FIT
# =============================================================================
# model.fit(
#     series           = train_target_seq,
#     future_covariates= train_future_seq,
#     val_series       = val_target_seq,        # if you have a val set
#     val_future_covariates = val_future_seq,
# )
