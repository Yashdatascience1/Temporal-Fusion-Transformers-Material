# =============================================================================
# DISK-LAZY TFT TRAINING
# Replaces Sections 3-6 of train_tft_custom_loss.py
#
# Why this design:
#   Darts keeps a *reference* to the sequence and calls sequence[i] once per
#   training sample (verified from source). So:
#     - RAM stays tiny; workers pickle only paths -> num_workers>0 is safe
#     - BUT every sample = 1 disk read, in shuffled order
#   Therefore each series gets its own small .npz holding pre-computed arrays.
#   Reading a 50-dealer chunk parquet per sample would be unusably slow.
#
#   Future covariates are date-only (identical across all series), so ONE
#   shared in-RAM TimeSeries serves all 117K -> never touches disk.
# =============================================================================

import os, json, glob, gc, time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import collections.abc

from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping

# =============================================================================
# SECTION 1: CONFIG
# =============================================================================

local_train_dir = "./local_train_data"
local_test_dir  = "./local_test_data"
CACHE_DIR       = "./series_cache"          # per-series .npz files live here

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
FREQ       = 'D'

TRAIN_END = pd.Timestamp("2025-12-31")
VAL_START = pd.Timestamp("2026-01-01")
VAL_END   = pd.Timestamp("2026-06-30")

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 184
TEST_HORIZON        = 184

static_covariates = [
    'PARENT_DEALER_CODE', 'MODEL_FAMILY', 'MODEL_NAME', 'BRAKE_TYPE',
    'IGNITION_TYPE', 'WHEEL_TYPE', 'COLOUR', 'DEALER_CITY',
    'X_CITY_CATEGORY', 'ZONAL_OFFICE_NAME'
]

future_covariates = [
    'NEW_YEAR','LOHRI','MAKAR_SANKRANTI','REPUBLIC_DAY','VASANT_PANCHAMI',
    'MAHA_SHIVRATRI','EID_UL_FITR','HOLIKA_DAHAN','HOLI','HANUMAN_JAYANTI',
    'AKSHAYA_TRITYA','BUDDHA_PURNIMA','GANGA_DUSSEHRA','JAGANNATH_RATHYATRA',
    'GURU_PURNIMA','NAG_PANCHAMI','RAKSHA_BANDHAN','HARTALIK_TEEJ',
    'GANESH_CHATURTHI','JANMASHTAMI','VISHWAKARMA_PUJA','KARWA_CHAUTH',
    'ONAM','MARRIAGE_DAY',
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

penalty_cols = [
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

val_window_days = (VAL_END - VAL_START).days + 1                       # 181
warmup_days     = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH - val_window_days  # 368
warmup_start    = VAL_START - pd.Timedelta(days=warmup_days)
MIN_LEN         = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH             # 549

# =============================================================================
# SECTION 2: BUILD PER-SERIES CACHE (one-time)
# =============================================================================
# For each series writes ./series_cache/<key>.npz containing:
#   train_sales, train_flag, val_sales, val_flag  (float32 arrays)
#   train_start, val_start                        (ISO date strings)
# Scaling params (min/max of the TRAIN window) go into scaler_stats.json so
# scaling happens at read time -- no Darts Scaler, no 117K-series alignment.
#
# Static covariates for all series are collected into one small DataFrame.

print("="*60)
print("SECTION 2: BUILDING PER-SERIES CACHE")
print("="*60)

os.makedirs(CACHE_DIR, exist_ok=True)
manifest_path = os.path.join(CACHE_DIR, "manifest.json")

def safe_name(key):
    return str(key).replace("<>", "_").replace("/", "_").replace("\\", "_")

if os.path.exists(manifest_path):
    print("Cache already exists — loading manifest.")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
else:
    needed_cols = [time_col, group_col, target_col] + static_covariates + penalty_cols
    chunk_files = sorted(glob.glob(os.path.join(local_train_dir, "chunk_*.parquet")))
    print(f"Scanning {len(chunk_files)} chunk files...")

    series_keys  = []
    has_val      = []
    scaler_stats = {}
    static_rows  = []

    for ci, chunk_path in enumerate(chunk_files):
        df = pd.read_parquet(chunk_path, columns=needed_cols)
        df[time_col] = pd.to_datetime(df[time_col])

        for key, g in df.groupby(group_col, sort=False):
            g = g.sort_values(time_col).reset_index(drop=True)
            flag = (g[penalty_cols] != 0).any(axis=1).to_numpy(dtype=np.float32)

            t = g[time_col]
            tr = (t <= TRAIN_END).to_numpy()
            va = ((t >= warmup_start) & (t <= VAL_END)).to_numpy()

            if tr.sum() < MIN_LEN:
                continue   # too short to yield even one training sample

            sales = g[target_col].to_numpy(dtype=np.float32)
            tr_sales, tr_flag = sales[tr], flag[tr]

            # scaling params from TRAIN window only (no leakage from val)
            lo = float(tr_sales.min())
            hi = float(tr_sales.max())
            if hi - lo < 1e-8:
                hi = lo + 1.0            # constant series guard

            keep_val = va.sum() >= MIN_LEN
            payload = {
                "train_sales": tr_sales,
                "train_flag":  tr_flag,
                "train_start": np.array(str(t[tr].iloc[0].date())),
            }
            if keep_val:
                payload["val_sales"] = sales[va]
                payload["val_flag"]  = flag[va]
                payload["val_start"] = np.array(str(t[va].iloc[0].date()))

            np.savez(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz"), **payload)

            series_keys.append(str(key))
            has_val.append(bool(keep_val))
            scaler_stats[str(key)] = [lo, hi]
            static_rows.append(g[static_covariates].iloc[0].to_dict())

        del df
        gc.collect()
        print(f"  chunk {ci+1}/{len(chunk_files)} done — series so far: {len(series_keys)}")

    pd.DataFrame(static_rows).to_parquet(
        os.path.join(CACHE_DIR, "static_covariates.parquet"), index=False
    )
    manifest = {"series_keys": series_keys, "has_val": has_val,
                "scaler_stats": scaler_stats}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    del static_rows
    gc.collect()

series_keys  = manifest["series_keys"]
has_val      = manifest["has_val"]
scaler_stats = manifest["scaler_stats"]
print(f"\nSeries cached          : {len(series_keys)}")
print(f"Series with val strip  : {sum(has_val)}")

# =============================================================================
# SECTION 3: STATIC COVARIATES (encoded once, held in RAM — small)
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: STATIC COVARIATES")
print("="*60)

static_df_all = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))

# Fit the transformer on a few dummy series, then reuse its encoding on the
# whole DataFrame at once (far cheaper than transforming 117K TimeSeries).
_dummy_idx = pd.date_range("2023-01-01", periods=2, freq=FREQ)
_probe = [
    TimeSeries.from_times_and_values(
        _dummy_idx, np.zeros((2, 1), np.float32),
        static_covariates=static_df_all.iloc[[i]].reset_index(drop=True)
    )
    for i in range(min(len(static_df_all), 5000))   # sample is enough to see categories
]
static_transformer = StaticCovariatesTransformer()
static_transformer.fit(_probe)
del _probe
gc.collect()

# Encode every row up-front into a list of 1-row DataFrames
_encoded = static_transformer.transform([
    TimeSeries.from_times_and_values(
        _dummy_idx, np.zeros((2, 1), np.float32),
        static_covariates=static_df_all.iloc[[i]].reset_index(drop=True)
    )
    for i in range(len(static_df_all))
])
STATIC_ENCODED = [ts.static_covariates for ts in _encoded]
del _encoded
gc.collect()
print(f"Encoded static covariates for {len(STATIC_ENCODED)} series.")

# =============================================================================
# SECTION 4: SHARED FUTURE COVARIATES (in RAM — never hits disk)
# =============================================================================
# All 67 covariates are date-derived and identical across series, so one
# TimeSeries covering the full date range serves every series.

print("\n" + "="*60)
print("SECTION 4: SHARED FUTURE COVARIATES")
print("="*60)

cov_cols = [time_col] + future_covariates
train_chunk0 = sorted(glob.glob(os.path.join(local_train_dir, "chunk_*.parquet")))[0]
test_chunk0  = sorted(glob.glob(os.path.join(local_test_dir,  "chunk_*.parquet")))[0]

cal = pd.concat([
    pd.read_parquet(train_chunk0, columns=cov_cols),
    pd.read_parquet(test_chunk0,  columns=cov_cols),
])
cal[time_col] = pd.to_datetime(cal[time_col])
cal = cal.drop_duplicates(subset=time_col).sort_values(time_col).reset_index(drop=True)

SHARED_COV = TimeSeries.from_dataframe(
    cal, time_col=time_col, value_cols=future_covariates,
    freq=FREQ, fill_missing_dates=False
).astype(np.float32)

print(f"Covariate calendar: {cal[time_col].min().date()} → {cal[time_col].max().date()} "
      f"({len(cal)} days, {len(future_covariates)} cols)")
del cal
gc.collect()

# =============================================================================
# SECTION 5: LAZY SEQUENCES
# =============================================================================

class DiskLazyTargetSequence(collections.abc.Sequence):
    """
    Target series backed by per-series .npz files.

    Returns a 2-component TimeSeries: [scaled NET_SALES, raw FESTIVE_FLAG].
    Scaling is applied at read time using train-window min/max, so no Darts
    Scaler and no 117K-series alignment problem.

    Pickles to just paths + small dicts -> safe with num_workers > 0.
    """

    def __init__(self, cache_dir, series_keys, scaler_stats, static_encoded,
                 split="train", freq='D'):
        self.cache_dir      = cache_dir
        self.series_keys    = series_keys
        self.scaler_stats   = scaler_stats
        self.static_encoded = static_encoded
        self.split          = split          # "train" or "val"
        self.freq           = freq

    def __len__(self):
        return len(self.series_keys)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        key  = self.series_keys[idx]
        path = os.path.join(self.cache_dir, f"{safe_name(key)}.npz")

        with np.load(path, allow_pickle=False) as z:
            sales = z[f"{self.split}_sales"]
            flag  = z[f"{self.split}_flag"]
            start = str(z[f"{self.split}_start"])

        lo, hi = self.scaler_stats[key]
        scaled = ((sales - lo) / (hi - lo)).astype(np.float32)

        values = np.stack([scaled, flag], axis=1)          # (T, 2)
        times  = pd.date_range(start=start, periods=len(values), freq=self.freq)

        return TimeSeries.from_times_and_values(
            times, values,
            columns=[target_col, "FESTIVE_FLAG"],
            static_covariates=self.static_encoded[idx],
        )


class SharedCovSequence(collections.abc.Sequence):
    """Returns the same in-RAM covariate TimeSeries for every index."""

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


# --- train sequences (all series) ---
train_seq = DiskLazyTargetSequence(
    CACHE_DIR, series_keys, scaler_stats, STATIC_ENCODED, split="train", freq=FREQ
)
train_cov_seq = SharedCovSequence(SHARED_COV, len(series_keys))

# --- val sequences (only series that have a val strip) ---
val_keys    = [k for k, h in zip(series_keys, has_val) if h]
val_statics = [s for s, h in zip(STATIC_ENCODED, has_val) if h]
val_seq = DiskLazyTargetSequence(
    CACHE_DIR, val_keys, scaler_stats, val_statics, split="val", freq=FREQ
)
val_cov_seq = SharedCovSequence(SHARED_COV, len(val_keys))

print("\n" + "="*60)
print("SECTION 5: SANITY CHECK")
print("="*60)
_t0 = time.time()
s0 = train_seq[0]
print(f"train_seq[0]: {s0.start_time().date()} → {s0.end_time().date()} | "
      f"len={len(s0)} | comps={s0.components.tolist()}")
print(f"single lazy read took {(time.time()-_t0)*1000:.1f} ms")
print(f"train series: {len(train_seq)} | val series: {len(val_seq)}")

# Read-speed probe: this number decides whether disk-lazy is viable at all
_t0 = time.time()
for i in np.random.randint(0, len(train_seq), 200):
    _ = train_seq[int(i)]
_per_read = (time.time() - _t0) / 200
print(f"\nAvg random read: {_per_read*1000:.2f} ms")
print(f"→ est. time for 1 epoch at 50 samples/series: "
      f"{_per_read * len(train_seq) * 50 / 3600:.1f} h of pure disk I/O")

# =============================================================================
# SECTION 6: LOSS
# =============================================================================

class HuberMaeFeatureLoss(nn.HuberLoss):
    """Huber(y_hat, y) + flag * |y - y_hat|, on component 0 only.
    Component 1 of the target carries the festive flag."""

    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__(reduction='none', delta=delta)
        self.user_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_hat = input[..., 0]
        y     = target[..., 0]
        flag  = target[..., 1]

        total = super().forward(y_hat, y) + flag * torch.abs(y - y_hat)

        if self.user_reduction == 'mean':
            return total.mean()
        if self.user_reduction == 'sum':
            return total.sum()
        return total

# =============================================================================
# SECTION 7: MODEL + TRAIN
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: MODEL")
print("="*60)

torch.set_float32_matmul_precision('high')

now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
MODEL_NAME = f"daily_tft_festive_lazy_{now}"
print("Model name:", MODEL_NAME)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, min_delta=1e-4, mode="min"
)

model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=4,
    num_attention_heads=16,
    dropout=0.05,

    batch_size=256,
    n_epochs=100,

    likelihood=None,
    loss_fn=HuberMaeFeatureLoss(delta=1.0, reduction='mean'),

    random_state=42,
    add_relative_index=True,

    save_checkpoints=True,
    force_reset=True,
    model_name=MODEL_NAME,
    skip_interpolation=True,

    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
        "precision": "bf16-mixed",
    },
)

print("\n" + "="*60)
print("SECTION 8: TRAINING")
print("="*60)

# num_workers > 0 is SAFE here: workers pickle paths, not 117K TimeSeries.
# persistent_workers avoids re-spawning (and re-pickling) every epoch.
model.fit(
    series=train_seq,
    future_covariates=train_cov_seq,

    val_series=val_seq,
    val_future_covariates=val_cov_seq,

    max_samples_per_ts=50,     # remove to use all ~458 windows per series

    dataloader_kwargs={
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
    },
    verbose=True,
)

print("\nTraining complete.")

# =============================================================================
# SECTION 9: PREDICT
# =============================================================================

best_model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)
print("Best checkpoint loaded.")

# Predict from the val series (they end 2026-06-30, so n=184 lands on 2026-12-31)
preds_scaled = best_model.predict(
    n=TEST_HORIZON,
    series=val_seq,
    future_covariates=val_cov_seq,
    verbose=True,
)

# Undo the read-time scaling, keep component 0, clip negatives
rows = []
for key, p in zip(val_keys, preds_scaled):
    lo, hi = scaler_stats[key]
    vals = p.univariate_component(0).values().flatten() * (hi - lo) + lo
    rows.append(pd.DataFrame({
        time_col: p.time_index,
        "PREDICTED_NET_SALES": np.clip(vals, 0, None),
        group_col: key,
    }))

pred_df = pd.concat(rows, ignore_index=True)
pred_df.to_parquet("./predictions_184d.parquet", index=False)
print(f"Saved {len(pred_df):,} rows → ./predictions_184d.parquet")
print(f"Horizon: {pred_df[time_col].min().date()} → {pred_df[time_col].max().date()}")
