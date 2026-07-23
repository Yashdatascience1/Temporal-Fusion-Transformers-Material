SHARED_COV.to_pickle(os.path.join(CACHE_DIR, "shared_cov.pkl"))
print("Saved:", SHARED_COV.start_time().date(), "→", SHARED_COV.end_time().date())

# =============================================================================
# PREDICT 2026-07-01 → 2026-12-31  (184 days)
#
# Runs after training was stopped. Uses the best checkpoint saved by
# save_checkpoints=True.
#
# Assumes local_train_data / local_test_data are DELETED. Everything needed
# now lives in CACHE_DIR:
#     *.npz                      per-series sales + festive flag
#     manifest.json              series keys, has_val, scaler min/max
#     static_covariates.parquet  raw static attributes
#     shared_cov.pkl             covariate calendar  <-- must exist
#
# Prediction runs in chunks and appends to disk after each one, so an
# interruption does not lose completed work.
# =============================================================================

import os, json, gc, time
import numpy as np
import pandas as pd
import torch
import collections.abc

from darts import TimeSeries
from darts.models import TFTModel
from sklearn.preprocessing import OrdinalEncoder

# =============================================================================
# SECTION 1: CONFIG  — must match training
# =============================================================================

DATA_ROOT  = r"C:\Users\G0004878\Desktop\TFT_Data"      # <-- adjust if needed
CACHE_DIR  = os.path.join(DATA_ROOT, "series_cache")
OUT_DIR    = os.path.join(DATA_ROOT, "predictions_2026")

MODEL_NAME = "PASTE_YOUR_MODEL_NAME_HERE"   # e.g. daily_tft_festive_lazy_2026-07-21_11_30_12
WORK_DIR   = os.getcwd()                    # where darts_logs lives

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
FREQ       = 'D'

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 184
HORIZON             = 184                    # 2026-07-01 → 2026-12-31

FORECAST_START = pd.Timestamp("2026-07-01")
FORECAST_END   = pd.Timestamp("2026-12-31")

PREDICT_CHUNK = 5000        # series per predict() call; lower if RAM is tight

static_covariates = [
    'PARENT_DEALER_CODE', 'MODEL_FAMILY', 'MODEL_NAME', 'BRAKE_TYPE',
    'IGNITION_TYPE', 'WHEEL_TYPE', 'COLOUR', 'DEALER_CITY',
    'X_CITY_CATEGORY', 'ZONAL_OFFICE_NAME'
]

os.makedirs(OUT_DIR, exist_ok=True)

def safe_name(key):
    return str(key).replace("<>", "_").replace("/", "_").replace("\\", "_")

# =============================================================================
# SECTION 2: LOAD CACHE ARTEFACTS
# =============================================================================

print("="*60)
print("SECTION 2: LOADING CACHE ARTEFACTS")
print("="*60)

with open(os.path.join(CACHE_DIR, "manifest.json"), "r") as f:
    manifest = json.load(f)

series_keys  = manifest["series_keys"]
has_val      = manifest["has_val"]
scaler_stats = manifest["scaler_stats"]

print(f"Series in manifest    : {len(series_keys):,}")
print(f"Series with val strip : {sum(has_val):,}")

# --- covariate calendar ---
cov_path = os.path.join(CACHE_DIR, "shared_cov.pkl")
if not os.path.exists(cov_path):
    raise FileNotFoundError(
        f"{cov_path} not found.\n"
        "SHARED_COV was never saved and local_train_data/local_test_data are "
        "deleted. It must be rebuilt from Snowflake before predicting."
    )

SHARED_COV = TimeSeries.from_pickle(cov_path)
print(f"Covariate calendar    : {SHARED_COV.start_time().date()} → "
      f"{SHARED_COV.end_time().date()}  ({len(SHARED_COV)} days)")

# Covariates must span the whole forecast horizon, or predict() will fail.
if SHARED_COV.end_time() < FORECAST_END:
    raise ValueError(
        f"Covariates end {SHARED_COV.end_time().date()} but the forecast runs to "
        f"{FORECAST_END.date()}. Cannot predict past the covariate calendar."
    )
print("Covariate coverage OK.")

# =============================================================================
# SECTION 3: STATIC COVARIATES  (identical encoding to training)
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: STATIC COVARIATES")
print("="*60)

static_df_all = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoded_arr = encoder.fit_transform(
    static_df_all[static_covariates].astype(str)
).astype(np.float32)

encoded_df = pd.DataFrame(encoded_arr, columns=static_covariates)
STATIC_ENCODED = [
    encoded_df.iloc[[i]].reset_index(drop=True) for i in range(len(encoded_df))
]

assert len(STATIC_ENCODED) == len(series_keys), \
    f"Mismatch: {len(STATIC_ENCODED)} statics vs {len(series_keys)} series"
print(f"Encoded static covariates for {len(STATIC_ENCODED):,} series.")

# =============================================================================
# SECTION 4: SEQUENCES  (same classes as training)
# =============================================================================

class DiskLazyTargetSequence(collections.abc.Sequence):
    """Target series from per-series .npz. Scaling applied at read time."""

    def __init__(self, cache_dir, series_keys, scaler_stats, static_encoded,
                 split="val", freq='D', cache_in_ram=True):
        self.cache_dir      = cache_dir
        self.series_keys    = series_keys
        self.scaler_stats   = scaler_stats
        self.static_encoded = static_encoded
        self.split          = split
        self.freq           = freq
        self.cache_in_ram   = cache_in_ram
        self._ram           = {} if cache_in_ram else None

    def __len__(self):
        return len(self.series_keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ram"] = {} if self.cache_in_ram else None
        return state

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        if self._ram is not None and idx in self._ram:
            sales, flag, start = self._ram[idx]
        else:
            key  = self.series_keys[idx]
            path = os.path.join(self.cache_dir, f"{safe_name(key)}.npz")
            with np.load(path, allow_pickle=False) as z:
                sales = z[f"{self.split}_sales"]
                flag  = z[f"{self.split}_flag"]
                start = str(z[f"{self.split}_start"])
            if self._ram is not None:
                self._ram[idx] = (sales, flag, start)

        lo, hi = self.scaler_stats[self.series_keys[idx]]
        scaled = ((sales - lo) / (hi - lo)).astype(np.float32)

        values = np.stack([scaled, flag], axis=1)
        times  = pd.date_range(start=start, periods=len(values), freq=self.freq)

        return TimeSeries.from_times_and_values(
            times, values,
            columns=[target_col, "FESTIVE_FLAG"],
            static_covariates=self.static_encoded[idx],
        )


class SharedCovSequence(collections.abc.Sequence):
    """Same in-RAM covariate series for every index."""

    def __init__(self, shared_series, n):
        self.shared = shared_series
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.shared for _ in range(*idx.indices(self.n))]
        if idx < 0:
            idx += self.n
        if not 0 <= idx < self.n:
            raise IndexError(idx)
        return self.shared

# Only series with a val strip end on 2026-06-30, which is what a 184-day
# horizon needs to land exactly on 2026-12-31.
predict_keys    = [k for k, h in zip(series_keys, has_val) if h]
predict_statics = [s for s, h in zip(STATIC_ENCODED, has_val) if h]

print(f"\nSeries to forecast : {len(predict_keys):,}")
skipped = len(series_keys) - len(predict_keys)
if skipped:
    print(f"Skipped (no val strip, series ends 2025-12-31): {skipped:,}")
    print("  -> these need n=365 from the train split; see Section 7.")

# =============================================================================
# SECTION 5: LOAD BEST CHECKPOINT
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: LOADING BEST CHECKPOINT")
print("="*60)

if MODEL_NAME == "PASTE_YOUR_MODEL_NAME_HERE":
    import glob as _glob
    found = sorted(_glob.glob(os.path.join(WORK_DIR, "darts_logs", "daily_tft_festive_lazy_*")))
    raise ValueError(
        "Set MODEL_NAME first. Candidates found in darts_logs:\n  " +
        "\n  ".join(os.path.basename(p) for p in found) if found
        else "Set MODEL_NAME first. No matching folders found in darts_logs."
    )

best_model = TFTModel.load_from_checkpoint(MODEL_NAME, work_dir=WORK_DIR, best=True)
print(f"Loaded best checkpoint: {MODEL_NAME}")
print(f"  input_chunk_length  : {best_model.input_chunk_length}")
print(f"  output_chunk_length : {best_model.output_chunk_length}")

# Prediction is forward-only; make sure nothing tries to track gradients.
best_model.model.eval()

# =============================================================================
# SECTION 6: PREDICT IN CHUNKS, SAVE AS WE GO
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: PREDICTING")
print("="*60)
print(f"Horizon: {FORECAST_START.date()} → {FORECAST_END.date()} ({HORIZON} days)")
print(f"Chunk size: {PREDICT_CHUNK:,} series\n")

n_chunks = int(np.ceil(len(predict_keys) / PREDICT_CHUNK))
t_start  = time.time()

for ci in range(n_chunks):
    out_path = os.path.join(OUT_DIR, f"pred_chunk_{ci:04d}.parquet")
    if os.path.exists(out_path):
        print(f"Chunk {ci+1}/{n_chunks} already exists — skipping.")
        continue

    lo_i = ci * PREDICT_CHUNK
    hi_i = min(lo_i + PREDICT_CHUNK, len(predict_keys))
    keys_c    = predict_keys[lo_i:hi_i]
    statics_c = predict_statics[lo_i:hi_i]

    seq_c = DiskLazyTargetSequence(
        CACHE_DIR, keys_c, scaler_stats, statics_c,
        split="val", freq=FREQ, cache_in_ram=False    # one pass only
    )
    cov_c = SharedCovSequence(SHARED_COV, len(keys_c))

    t0 = time.time()
    with torch.no_grad():
        preds = best_model.predict(
            n=HORIZON,
            series=seq_c,
            future_covariates=cov_c,
            verbose=False,
        )

    # Undo read-time scaling, keep component 0 (NET_SALES), clip negatives
    rows = []
    for key, p in zip(keys_c, preds):
        lo, hi = scaler_stats[key]
        vals = p.values()[:, 0] * (hi - lo) + lo
        rows.append(pd.DataFrame({
            time_col: p.time_index,
            group_col: key,
            "PREDICTED_NET_SALES": np.clip(vals, 0, None).astype(np.float32),
        }))

    chunk_df = pd.concat(rows, ignore_index=True)
    chunk_df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    done    = ci + 1
    eta     = (time.time() - t_start) / done * (n_chunks - done) / 60
    print(f"Chunk {done}/{n_chunks} | {len(keys_c):,} series | "
          f"{elapsed/60:.1f} min | ETA {eta:.0f} min")

    del preds, rows, chunk_df, seq_c
    gc.collect()

print(f"\nAll chunks done in {(time.time()-t_start)/60:.1f} min.")

# =============================================================================
# SECTION 7: COMBINE + SANITY CHECK
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: COMBINING & SANITY CHECK")
print("="*60)

import glob as _glob
parts = sorted(_glob.glob(os.path.join(OUT_DIR, "pred_chunk_*.parquet")))
pred_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

final_path = os.path.join(DATA_ROOT, "predictions_jul_dec_2026.parquet")
pred_df.to_parquet(final_path, index=False)

print(f"Rows            : {len(pred_df):,}")
print(f"Series          : {pred_df[group_col].nunique():,}")
print(f"Date range      : {pred_df[time_col].min().date()} → {pred_df[time_col].max().date()}")
print(f"Saved           : {final_path}")

# Aggregate daily total — the fastest way to spot a broken forecast
daily = pred_df.groupby(time_col)["PREDICTED_NET_SALES"].sum()
print(f"\nDaily total    : min {daily.min():,.0f} | mean {daily.mean():,.0f} | max {daily.max():,.0f}")
print(f"Peak day       : {daily.idxmax().date()}  ({daily.max():,.0f})")
print(f"Total forecast : {pred_df['PREDICTED_NET_SALES'].sum()/1e5:,.2f} lacs")

zero_series = (pred_df.groupby(group_col)["PREDICTED_NET_SALES"].sum() == 0).sum()
print(f"All-zero series: {zero_series:,}")

# Monthly view — is there an Oct/Nov festive lift?
monthly = pred_df.groupby(pred_df[time_col].dt.to_period("M"))["PREDICTED_NET_SALES"].sum()
print("\nMonthly totals (lacs):")
for m, v in monthly.items():
    print(f"  {m}: {v/1e5:>10,.2f}")
