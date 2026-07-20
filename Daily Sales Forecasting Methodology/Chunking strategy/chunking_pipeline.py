# =============================================================================
# FULL PIPELINE: Snowflake → Local Chunk Parquet → Darts TimeSeries
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
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
import os, json, glob, gc
import collections.abc

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

snowflake_conn_prop = Snowflake_configuration.ds1_role_json
session = Session.builder.configs(snowflake_conn_prop).create()
session.use_database('MOP_DATABASE')
session.use_schema('SOQ')

TABLE_NAME      = 'MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT_APR_23_TO_DEC_26'
local_train_dir = "./local_train_data"
local_test_dir  = "./local_test_data"
group_keys_path = "./saved_group_keys/group_keys.json"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
dealer_col = 'PARENT_DEALER_CODE'
target_col = 'NET_SALES'
CHUNK_SIZE = 100
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

# Sanitize group key — replace <> with _ for clean filenames and consistent keys
def sanitize(df):
    return df.with_column(
        group_col,
        F.replace(F.col(group_col), F.lit('<>'), F.lit('_'))
    )

train_set = sanitize(train_set)
test_set  = sanitize(test_set)

# =============================================================================
# SECTION 3: GET DEALER LIST & BUILD CHUNKS
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: DEALER CHUNKS")
print("="*60)

all_dealers = [
    row[dealer_col]
    for row in train_set.select(dealer_col).distinct().collect()
]

dealer_chunks = [
    all_dealers[i:i + CHUNK_SIZE]
    for i in range(0, len(all_dealers), CHUNK_SIZE)
]

print(f"Total dealers : {len(all_dealers)}")
print(f"Chunk size    : {CHUNK_SIZE}")
print(f"Total chunks  : {len(dealer_chunks)}")

# =============================================================================
# SECTION 4: SAVE GROUP KEYS
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: SAVING GROUP KEYS")
print("="*60)

os.makedirs(os.path.dirname(group_keys_path), exist_ok=True)

group_keys = [
    row[group_col]
    for row in train_set.select(group_col).distinct().collect()
]

with open(group_keys_path, "w") as f:
    json.dump(group_keys, f)

print(f"Saved {len(group_keys)} group keys → {group_keys_path}")

# =============================================================================
# SECTION 5: WRITE CHUNK PARQUET FILES VIA SNOWPARK (no .to_pandas())
# =============================================================================
# Snowpark writes directly from Snowflake to local disk.
# No pandas conversion, no RAM spike.
# One parquet file per 100-dealer chunk.
# Resume-safe: skips chunks already on disk.

print("\n" + "="*60)
print("SECTION 5: WRITING PARQUET FILES TO LOCAL DISK")
print("="*60)

os.makedirs(local_train_dir, exist_ok=True)
os.makedirs(local_test_dir, exist_ok=True)

def to_file_uri(local_path):
    """Convert a local path to a file:/// URI (Windows-safe)."""
    abs_path = os.path.abspath(local_path).replace("\\", "/")
    return f"file:///{abs_path}"

def write_chunks(snowpark_df, dealer_chunks, dealer_col, local_dir, label="train"):
    total = len(dealer_chunks)
    for idx, chunk in enumerate(dealer_chunks):
        out_path = os.path.join(local_dir, f"chunk_{idx:04d}.parquet")

        if os.path.exists(out_path):
            print(f"[{label}] Chunk {idx+1}/{total} already exists, skipping.")
            continue

        print(f"[{label}] Writing chunk {idx+1}/{total} ({len(chunk)} dealers)...")

        snowpark_df \
            .filter(F.col(dealer_col).isin(chunk)) \
            .write.parquet(to_file_uri(out_path))

        print(f"[{label}] Chunk {idx+1}/{total} written → {out_path}")

    written = len(glob.glob(os.path.join(local_dir, "chunk_*.parquet")))
    print(f"\n[{label}] Done. {written} chunk files in {local_dir}")

write_chunks(train_set, dealer_chunks, dealer_col, local_train_dir, label="TRAIN")
write_chunks(test_set,  dealer_chunks, dealer_col, local_test_dir,  label="TEST")

session.close()
print("\nSnowflake session closed.")

# =============================================================================
# SECTION 6: BUILD GROUP → CHUNK INDEX
# =============================================================================
# Scan all chunk files once, record which chunk file each group key lives in.
# Saved to disk so it only needs to be built once.

print("\n" + "="*60)
print("SECTION 6: BUILD GROUP → CHUNK INDEX")
print("="*60)

def build_group_to_chunk_index(local_dir, group_col):
    index_path = os.path.join(local_dir, "group_to_chunk_index.json")

    if os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        with open(index_path, "r") as f:
            return json.load(f)

    print(f"Building index for {local_dir} (one-time scan)...")
    chunk_files = sorted(glob.glob(os.path.join(local_dir, "chunk_*.parquet")))
    index = {}

    for chunk_path in chunk_files:
        # Read only group_col column — minimal RAM
        df = pd.read_parquet(chunk_path, columns=[group_col])
        for key in df[group_col].unique():
            index[str(key)] = chunk_path
        del df
        gc.collect()

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Index built: {len(index)} groups across {len(chunk_files)} chunk files.")
    return index

group_to_chunk_train = build_group_to_chunk_index(local_train_dir, group_col)
group_to_chunk_test  = build_group_to_chunk_index(local_test_dir,  group_col)

# =============================================================================
# SECTION 7: DiskLazyTimeSeriesSequence
# =============================================================================
# Reads chunk parquet files on demand — one chunk at a time.
# Caches the last-read chunk file so consecutive groups from the same chunk
# don't trigger repeated disk reads.
#
# Darts indexes into this sequence once per series at fit() time to build
# its internal PyTorch Dataset. After that, training runs purely in memory.

class DiskLazyTimeSeriesSequence(collections.abc.Sequence):
    """
    Sequence of target TimeSeries backed by chunk parquet files on disk.

    Parameters
    ----------
    local_dir       : folder containing chunk_NNNN.parquet files
    group_keys      : ordered list of group key strings
    group_to_chunk  : dict mapping group_key → chunk file path
    time_col        : datetime column name
    group_col       : group identifier column name
    target_col      : target column name
    static_cov_cols : list of static covariate column names
    freq            : pandas frequency string e.g. 'D'
    """

    def __init__(self, local_dir, group_keys, group_to_chunk,
                 time_col, group_col, target_col,
                 static_cov_cols=None, freq='D'):
        self.local_dir       = local_dir
        self.group_keys      = group_keys
        self.group_to_chunk  = group_to_chunk
        self.time_col        = time_col
        self.group_col       = group_col
        self.target_col      = target_col
        self.static_cov_cols = static_cov_cols or []
        self.freq            = freq

        # Chunk-level cache: avoid re-reading the same file for consecutive groups
        self._cached_chunk_path = None
        self._cached_chunk_df   = None

    def __len__(self):
        return len(self.group_keys)

    def _get_group_df(self, group_key):
        chunk_path = self.group_to_chunk.get(str(group_key))
        if chunk_path is None:
            raise KeyError(f"Group key not found in index: {group_key}")

        # Load chunk into cache only if it changed
        if chunk_path != self._cached_chunk_path:
            df = pd.read_parquet(chunk_path)
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            self._cached_chunk_df   = df
            self._cached_chunk_path = chunk_path

        group_df = self._cached_chunk_df[
            self._cached_chunk_df[self.group_col] == group_key
        ].sort_values(self.time_col).reset_index(drop=True)

        return group_df

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range")

        group_key = self.group_keys[idx]
        df = self._get_group_df(group_key)

        static_df = None
        if self.static_cov_cols:
            static_df = df[self.static_cov_cols].iloc[[0]].reset_index(drop=True)

        return TimeSeries.from_dataframe(
            df,
            time_col=self.time_col,
            value_cols=self.target_col,
            static_covariates=static_df,
            freq=self.freq,
            fill_missing_dates=False
        )


class DiskLazyFutureCovSequence(collections.abc.Sequence):
    """
    Parallel sequence for future covariates.
    Must use the same group_keys order as DiskLazyTimeSeriesSequence.
    """

    def __init__(self, local_dir, group_keys, group_to_chunk,
                 time_col, group_col, future_cov_cols, freq='D'):
        self.local_dir       = local_dir
        self.group_keys      = group_keys
        self.group_to_chunk  = group_to_chunk
        self.time_col        = time_col
        self.group_col       = group_col
        self.future_cov_cols = future_cov_cols
        self.freq            = freq

        self._cached_chunk_path = None
        self._cached_chunk_df   = None

    def __len__(self):
        return len(self.group_keys)

    def _get_group_df(self, group_key):
        chunk_path = self.group_to_chunk.get(str(group_key))
        if chunk_path is None:
            raise KeyError(f"Group key not found in index: {group_key}")

        if chunk_path != self._cached_chunk_path:
            df = pd.read_parquet(chunk_path)
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            self._cached_chunk_df   = df
            self._cached_chunk_path = chunk_path

        return self._cached_chunk_df[
            self._cached_chunk_df[self.group_col] == group_key
        ].sort_values(self.time_col).reset_index(drop=True)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range")

        df = self._get_group_df(self.group_keys[idx])

        return TimeSeries.from_dataframe(
            df,
            time_col=self.time_col,
            value_cols=self.future_cov_cols,
            freq=self.freq,
            fill_missing_dates=False
        )


# =============================================================================
# SECTION 8: INSTANTIATE SEQUENCES
# =============================================================================

print("\n" + "="*60)
print("SECTION 8: INSTANTIATE SEQUENCES")
print("="*60)

with open(group_keys_path, "r") as f:
    group_keys = json.load(f)

print(f"Loaded {len(group_keys)} group keys")

train_target_seq = DiskLazyTimeSeriesSequence(
    local_dir       = local_train_dir,
    group_keys      = group_keys,
    group_to_chunk  = group_to_chunk_train,
    time_col        = time_col,
    group_col       = group_col,
    target_col      = target_col,
    static_cov_cols = static_covariates,
    freq            = FREQ
)

train_future_seq = DiskLazyFutureCovSequence(
    local_dir       = local_train_dir,
    group_keys      = group_keys,
    group_to_chunk  = group_to_chunk_train,
    time_col        = time_col,
    group_col       = group_col,
    future_cov_cols = future_covariates,
    freq            = FREQ
)

# =============================================================================
# SECTION 9: SANITY CHECK
# =============================================================================

print("\n" + "="*60)
print("SECTION 9: SANITY CHECK")
print("="*60)

print("Checking series[0]...")
ts0 = train_target_seq[0]
fc0 = train_future_seq[0]
print(f"  Target : {ts0.start_time()} → {ts0.end_time()} | len={len(ts0)}")
print(f"  Future : {fc0.start_time()} → {fc0.end_time()} | cols={fc0.components.tolist()}")
print(f"  Static : {ts0.static_covariates}")

print("\nChecking series[-1]...")
ts_last = train_target_seq[-1]
print(f"  Target : {ts_last.start_time()} → {ts_last.end_time()} | len={len(ts_last)}")

print(f"\nTotal series : {len(train_target_seq)}")
print("\nPipeline complete. Ready for TFT training.")

# =============================================================================
# USAGE IN TFT FIT
# =============================================================================
# model.fit(
#     series                = train_target_seq,
#     future_covariates     = train_future_seq,
#     val_series            = val_target_seq,
#     val_future_covariates = val_future_seq,
# )
