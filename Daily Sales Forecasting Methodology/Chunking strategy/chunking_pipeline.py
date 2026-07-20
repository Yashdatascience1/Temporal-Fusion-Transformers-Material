# =============================================================================
# FULL PIPELINE: Snowflake → Local Parquet → Darts TimeSeries
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
import os
import shutil
import json
import glob
import math
import gc

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

snowflake_conn_prop = Snowflake_configuration.ds1_role_json
session = Session.builder.configs(snowflake_conn_prop).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')

TABLE_NAME    = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'
local_train_dir  = "./local_train_data"
local_test_dir   = "./local_test_data"
group_keys_path  = "./saved_group_keys/group_keys.json"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
dealer_col = 'PARENT_DEALER_CODE'
target_col = 'NET_SALES'
CHUNK_SIZE = 300
FREQ       = 'D'

# =============================================================================
# SECTION 2: LOAD & SANITIZE DATA
# =============================================================================

data = session.table(TABLE_NAME)

# Derive column lists from schema
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

print(f"Static covariates  : {static_covariates}")
print(f"Future covariates  : {future_covariates}")

# Train / test split
train_set = data.filter(F.col("CAL_DATE") <= '2026-06-30')
test_set  = data.filter(F.col("CAL_DATE") >= '2026-07-01')

# Sanitize group key
def sanitize(df):
    return df.with_column(
        group_col,
        F.replace(F.col(group_col), F.lit('<>'), F.lit('_'))
    )

train_set = sanitize(train_set)
test_set  = sanitize(test_set)

# =============================================================================
# SECTION 3: MEMORY CHECK — one chunk of 300 dealers
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
print(f"Number of chunks     : {len(dealer_chunks)} (chunk size = {CHUNK_SIZE})")

# Pull one chunk to RAM to estimate memory
print("\nPulling chunk 0 to estimate memory footprint...")
sample_chunk_df = (
    train_set
    .filter(F.col(dealer_col).isin(dealer_chunks[0]))
    .to_pandas()
)

chunk_ram_gb   = sample_chunk_df.memory_usage(deep=True).sum() / 1e9
total_est_gb   = chunk_ram_gb * len(dealer_chunks)
n_series_chunk = sample_chunk_df[group_col].nunique()

print(f"Rows in chunk 0            : {len(sample_chunk_df):,}")
print(f"Series in chunk 0          : {n_series_chunk:,}")
print(f"RAM for chunk 0            : {chunk_ram_gb:.2f} GB")
print(f"Estimated RAM all chunks   : {total_est_gb:.2f} GB")
print(f"Estimated series total     : {n_series_chunk * len(dealer_chunks):,}")

# Clean up sample
del sample_chunk_df
gc.collect()

# =============================================================================
# SECTION 4: DOWNLOAD — Snowflake → per-group local parquet files
# =============================================================================
# Strategy: pull one dealer chunk at a time, then split into per-group parquet
# files inside local_train_dir. This avoids holding the full dataset in RAM.
# Per-group files let TimeSeries construction stream group-by-group cleanly.

print("\n" + "="*60)
print("SECTION 4: DOWNLOADING TO LOCAL STORAGE")
print("="*60)

for local_dir in [local_train_dir, local_test_dir]:
    os.makedirs(local_dir, exist_ok=True)

def download_and_partition(snowpark_df, dealer_chunks, dealer_col, group_col,
                            time_col, local_dir, label="train"):
    """
    For each dealer chunk:
      1. Pull from Snowflake to pandas
      2. Split by group_col and save one parquet per group
    Skips groups that already exist on disk (resume-safe).
    """
    total_chunks = len(dealer_chunks)

    for idx, chunk in enumerate(dealer_chunks):
        print(f"\n[{label}] Chunk {idx+1}/{total_chunks} — pulling {len(chunk)} dealers...")

        chunk_df = (
            snowpark_df
            .filter(F.col(dealer_col).isin(chunk))
            .to_pandas()
        )
        chunk_df[time_col] = pd.to_datetime(chunk_df[time_col])
        chunk_df = chunk_df.sort_values([group_col, time_col]).reset_index(drop=True)

        groups_in_chunk = chunk_df[group_col].unique()
        saved = 0
        skipped = 0

        for group_key in groups_in_chunk:
            # Sanitize key for use as filename
            safe_key = str(group_key).replace("/", "_").replace("\\", "_")
            out_path = os.path.join(local_dir, f"group_{safe_key}.parquet")

            if os.path.exists(out_path):
                skipped += 1
                continue

            group_df = chunk_df[chunk_df[group_col] == group_key].reset_index(drop=True)
            group_df.to_parquet(out_path, index=False)
            saved += 1

        print(f"  Saved: {saved} | Skipped (already exist): {skipped}")

        # Free memory before next chunk
        del chunk_df
        gc.collect()

    print(f"\n[{label}] Download complete. Files in {local_dir}: "
          f"{len(glob.glob(os.path.join(local_dir, 'group_*.parquet')))}")

download_and_partition(train_set, dealer_chunks, dealer_col, group_col,
                       time_col, local_train_dir, label="TRAIN")
download_and_partition(test_set,  dealer_chunks, dealer_col, group_col,
                       time_col, local_test_dir,  label="TEST")

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

print(f"Saved {len(group_keys)} group keys to {group_keys_path}")

session.close()
print("Snowflake session closed.")

# =============================================================================
# SECTION 6: BUILD DARTS TIMESERIES OBJECTS
# =============================================================================
# Load all per-group parquet files from disk and build TimeSeries lists.
# All series are loaded into RAM here — confirmed safe given 64 GB RAM.

print("\n" + "="*60)
print("SECTION 6: BUILDING DARTS TIMESERIES OBJECTS")
print("="*60)

with open(group_keys_path, "r") as f:
    group_keys = json.load(f)

print(f"Building TimeSeries for {len(group_keys)} groups...")

train_target_series   = []
train_future_cov_series = []
train_static_rows     = []

missing = []

for i, group_key in enumerate(group_keys):
    safe_key = str(group_key).replace("/", "_").replace("\\", "_")
    path = os.path.join(local_train_dir, f"group_{safe_key}.parquet")

    if not os.path.exists(path):
        missing.append(group_key)
        continue

    df = pd.read_parquet(path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # Target
    ts_target = TimeSeries.from_dataframe(
        df,
        time_col=time_col,
        value_cols=target_col,
        freq=FREQ,
        fill_missing_dates=False
    )

    # Future covariates
    ts_future = TimeSeries.from_dataframe(
        df,
        time_col=time_col,
        value_cols=future_covariates,
        freq=FREQ,
        fill_missing_dates=False
    )

    # Static covariates — one row per series, attached to the TimeSeries
    static_df = df[static_covariates].iloc[[0]].reset_index(drop=True)
    ts_target = ts_target.with_static_covariates(static_df)

    train_target_series.append(ts_target)
    train_future_cov_series.append(ts_future)

    if (i + 1) % 5000 == 0:
        print(f"  Built {i+1}/{len(group_keys)} series...")

if missing:
    print(f"\nWARNING: {len(missing)} group files missing from disk: {missing[:5]} ...")
else:
    print(f"\nAll {len(train_target_series)} target series built successfully.")
    print(f"All {len(train_future_cov_series)} future covariate series built successfully.")

# Quick sanity check on first series
print("\nSample series[0]:")
print(f"  Time range : {train_target_series[0].start_time()} → {train_target_series[0].end_time()}")
print(f"  Length     : {len(train_target_series[0])}")
print(f"  Components : {train_target_series[0].components.tolist()}")
print(f"  Static covs: {train_target_series[0].static_covariates}")

print("\nSample future_cov[0]:")
print(f"  Components : {train_future_cov_series[0].components.tolist()}")

print("\nPipeline complete. Ready for TFT training.")
