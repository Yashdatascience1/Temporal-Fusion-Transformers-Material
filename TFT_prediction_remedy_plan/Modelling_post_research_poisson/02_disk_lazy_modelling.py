"""
=============================================================================
STEP 2 OF 2 -- DISK-LAZY TFT MODELLING (count likelihood, no target scaling)
=============================================================================

Changes vs the previous version, and what each one forces:

  1. COUNT LIKELIHOOD, NO SCALING
     likelihood=NegativeBinomialLikelihood() replaces the min-max scaler.
     Consequences you cannot opt out of:
       a) loss_fn is IGNORED once a likelihood is set. HuberMaeFeatureLoss
          is therefore gone. Festive weighting moves to sample_weight.
       b) The target must be ONE component. The old [sales, FESTIVE_FLAG]
          2-component target would make the model fit a Negative Binomial to
          the 0/1 flag as well, which is meaningless.
       c) use_reversible_instance_norm MUST be False. RIN re-centres and
          re-scales the target per window; a count likelihood needs raw
          counts on non-negative integer support. The two conflict directly.
       d) NET_SALES must be >= 0. It is built as INVOICED + CANCELLED +
          RETURNED, so it can go negative. Clamped here, with a printed count.

  2. StaticCovariatesTransformer REPLACES sklearn OrdinalEncoder.
     Fitted once on a compact frame containing every category, then applied
     lazily per series inside __getitem__.

     NOTE: this changes WHICH encoder runs, not how the network reads the
     result. StaticCovariatesTransformer also ordinal-encodes categoricals.
     The thing that actually stops PARENT_DEALER_CODE being read as a
     continuous number is categorical_embedding_sizes on the TFTModel
     constructor -- set in Section 7.

  3. Calendar features come from shared_calendar.parquet, built in step 1.
     Not add_encoders -- see the Section 6 comment in 01_data_chunking.py.

Prerequisite: run 01_data_chunking.py first.
"""

import os
import json
import glob
import gc
import collections.abc
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.models import TFTModel
from darts.utils.likelihood_models import (
    NegativeBinomialLikelihood,
    PoissonLikelihood,
)
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import OrdinalEncoder


# =============================================================================
# SECTION 0: CONFIG
# =============================================================================

BASE = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"

local_train_dir = os.path.join(BASE, "local_train_scooter_data")
local_test_dir  = os.path.join(BASE, "local_test_scooter_data")
calendar_path   = os.path.join(BASE, "shared_calendar.parquet")
roles_path      = os.path.join(BASE, "column_roles.json")
CACHE_DIR       = os.path.join(os.getcwd(), "series_cache_scooter")

with open(roles_path) as f:
    ROLES = json.load(f)

time_col          = ROLES["time_col"]
group_col         = ROLES["group_col"]
target_col        = ROLES["target_col"]
FREQ              = ROLES["freq"]
static_covariates = ROLES["static_covariates"]
future_covariates = ROLES["future_covariates"]

TRAIN_START = pd.Timestamp(ROLES.get("train_start", "2023-04-01"))
TRAIN_END   = pd.Timestamp(ROLES.get("train_end",   "2026-04-30"))
VAL_START   = pd.Timestamp(ROLES.get("val_start",   "2026-05-01"))
VAL_END     = pd.Timestamp(ROLES.get("val_end",     "2026-08-31"))

FORECAST_START = pd.Timestamp(ROLES.get("forecast_start", "2026-09-01"))
FORECAST_END   = pd.Timestamp(ROLES.get("forecast_end",   "2026-12-07"))

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 98
TEST_HORIZON        = (FORECAST_END - FORECAST_START).days + 1   # 98

assert TEST_HORIZON == OUTPUT_CHUNK_LENGTH, (
    f"Forecast window is {TEST_HORIZON} days but OUTPUT_CHUNK_LENGTH is "
    f"{OUTPUT_CHUNK_LENGTH}. Predicting n > output_chunk_length forces Darts "
    f"into auto-regressive rollout, which compounds error across the festive "
    f"peak. Keep them equal."
)

# festive sample weighting -- replaces the old custom-loss festive term.
# 1.0 means "no weighting"; raise it to make festive days matter more.
FESTIVE_WEIGHT = 3.0

# which likelihood. NegBin allows variance > mean (overdispersion), which
# daily retail counts essentially always have. Poisson forces variance == mean
# and will under-disperse; keep it only as a simpler first check.
LIKELIHOOD = NegativeBinomialLikelihood()
# LIKELIHOOD = PoissonLikelihood()

penalty_cols = [
    'N-16', 'N-15', 'N-14', 'N-13', 'N-12', 'N-11', 'N-10', 'N-9', 'N-8', 'N-7',
    'N-6', 'N-5', 'N-4', 'N-3', 'N-2', 'N-1', 'N', 'N+1', 'N+2', 'N+3', 'N+4',
    'N+5', 'N+6', 'N+7', 'N+8', 'N+9', 'N+10',
    'D-3', 'D-2', 'D-1', 'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6',
    'C', 'C+1', 'C+2', 'C+3', 'C+4', 'C+5', 'C+6'
]
penalty_cols = [c for c in penalty_cols if c in future_covariates]

val_window_days = (VAL_END - VAL_START).days + 1
warmup_days     = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH - val_window_days
warmup_start    = VAL_START - pd.Timedelta(days=warmup_days)
MIN_LEN         = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH

_val_in_end   = warmup_start + pd.Timedelta(days=INPUT_CHUNK_LENGTH - 1)
_val_out_start = _val_in_end + pd.Timedelta(days=1)

print("\n" + "=" * 60)
print("DATE CONFIGURATION")
print("=" * 60)
print(f"TRAIN     : {TRAIN_START.date()} -> {TRAIN_END.date()}")
print(f"VAL cfg   : {VAL_START.date()} -> {VAL_END.date()}  ({val_window_days} days)")
print(f"FORECAST  : {FORECAST_START.date()} -> {FORECAST_END.date()}  ({TEST_HORIZON} days)")
print(f"ICL / OCL : {INPUT_CHUNK_LENGTH} / {OUTPUT_CHUNK_LENGTH}  (MIN_LEN = {MIN_LEN})")
print()
print(f"Validation sample -> input : {warmup_start.date()} to {_val_in_end.date()}")
print(f"Validation sample -> output: {_val_out_start.date()} to {VAL_END.date()}")
print("NOTE: the scored output window is NOT VAL_START..VAL_END. VAL_START only")
print("      sizes the warmup; it cancels out of the final window.")

# --- diagnostic 1: overlap between the scored window and training ---------
if _val_out_start <= TRAIN_END:
    _ov = (min(TRAIN_END, VAL_END) - _val_out_start).days + 1
    print(f"\nWARNING: {_ov} of {OUTPUT_CHUNK_LENGTH} scored days fall on or before "
          f"TRAIN_END ({TRAIN_END.date()}).")
    print("         That fraction of val_loss measures memorisation, not generalisation.")
else:
    print(f"\nOK: scored window starts {_val_out_start.date()}, after TRAIN_END "
          f"({TRAIN_END.date()}). No overlap with training.")

# --- diagnostic 2: does the scored window contain any festive days? -------
if not (_val_out_start <= FORECAST_END and VAL_END >= FORECAST_START):
    print("\nWARNING: the scored validation window contains NO days from the "
          "festive period")
    print(f"         ({FORECAST_START.date()} - {FORECAST_END.date()}). Early stopping and")
    print("         checkpoint selection are therefore optimising for non-festive")
    print("         trading. A model that flattens peaks scores well here.")
    print("         See the remediation plan, Priority 1.")


def safe_name(key):
    return str(key).replace("<>", "_").replace("/", "_").replace("\\", "_")


# =============================================================================
# SECTION 1: PER-SERIES CACHE
# =============================================================================
# Stores RAW counts. No scaler_stats -- there is no scaling any more.
# Stores a festive WEIGHT array instead of a festive FLAG component, because
# the flag can no longer ride along as a second target component.

print("\n" + "=" * 60)
print("SECTION 1: BUILDING PER-SERIES CACHE")
print("=" * 60)

os.makedirs(CACHE_DIR, exist_ok=True)
manifest_path = os.path.join(CACHE_DIR, "manifest.json")

if os.path.exists(manifest_path):
    print("Cache exists -- loading manifest.")
    with open(manifest_path) as f:
        manifest = json.load(f)
else:
    needed_cols = [time_col, group_col, target_col] + static_covariates + penalty_cols
    chunk_files = sorted(glob.glob(os.path.join(local_train_dir, "chunk_*.parquet")))
    print(f"Scanning {len(chunk_files)} chunk files...")

    series_keys, has_val, static_rows = [], [], []
    n_neg_clamped = 0
    n_neg_series  = 0

    for ci, chunk_path in enumerate(chunk_files):
        df = pd.read_parquet(chunk_path, columns=needed_cols)
        df[time_col] = pd.to_datetime(df[time_col])

        for key, g in df.groupby(group_col, sort=False):
            g = g.sort_values(time_col).reset_index(drop=True)

            t  = g[time_col]
            tr = (t <= TRAIN_END).to_numpy()
            va = ((t >= warmup_start) & (t <= VAL_END)).to_numpy()

            if tr.sum() < MIN_LEN:
                continue

            sales = g[target_col].to_numpy(dtype=np.float32)

            # --- count-likelihood guard: negative support is undefined -------
            neg_mask = sales < 0
            if neg_mask.any():
                n_neg_clamped += int(neg_mask.sum())
                n_neg_series  += 1
                sales = np.clip(sales, 0.0, None)

            # counts must be integral for a discrete likelihood
            sales = np.rint(sales).astype(np.float32)

            flag   = (g[penalty_cols] != 0).any(axis=1).to_numpy(dtype=np.float32)
            weight = (1.0 + FESTIVE_WEIGHT * flag).astype(np.float32)

            keep_val = va.sum() >= MIN_LEN
            payload = {
                "train_sales":  sales[tr],
                "train_weight": weight[tr],
                "train_start":  np.array(str(t[tr].iloc[0].date())),
            }
            if keep_val:
                payload["val_sales"]  = sales[va]
                payload["val_weight"] = weight[va]
                payload["val_start"]  = np.array(str(t[va].iloc[0].date()))

            np.savez(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz"), **payload)

            series_keys.append(str(key))
            has_val.append(bool(keep_val))
            static_rows.append(g[static_covariates].iloc[0].to_dict())

        del df
        gc.collect()
        print(f"  chunk {ci+1}/{len(chunk_files)} -- series so far: {len(series_keys)}")

    pd.DataFrame(static_rows).to_parquet(
        os.path.join(CACHE_DIR, "static_covariates.parquet"), index=False
    )
    manifest = {"series_keys": series_keys, "has_val": has_val,
                "n_neg_clamped": n_neg_clamped, "n_neg_series": n_neg_series}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    del static_rows
    gc.collect()

series_keys = manifest["series_keys"]
has_val     = manifest["has_val"]

print(f"\nSeries cached         : {len(series_keys)}")
print(f"Series with val strip : {sum(has_val)}")
if manifest.get("n_neg_clamped", 0):
    print(f"WARNING: clamped {manifest['n_neg_clamped']:,} negative NET_SALES values "
          f"to 0 across {manifest['n_neg_series']:,} series.")
    print("         Confirm CANCELLED/RETURNED sign convention upstream before")
    print("         treating these forecasts as final.")


# =============================================================================
# SECTION 2: DROP DEAD SERIES
# =============================================================================

DROP_NEVER_SOLD = True
DORMANT_DAYS    = None

keep = []
for k in series_keys:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        s = z["train_sales"]
    if DROP_NEVER_SOLD and s.sum() == 0:
        keep.append(False); continue
    if DORMANT_DAYS and s[-DORMANT_DAYS:].sum() == 0:
        keep.append(False); continue
    keep.append(True)

static_df_all = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))

n_before = len(series_keys)
series_keys   = [k for k, m in zip(series_keys, keep) if m]
has_val       = [h for h, m in zip(has_val, keep) if m]
static_df_all = static_df_all.loc[keep].reset_index(drop=True)

assert len(series_keys) == len(has_val) == len(static_df_all)
print(f"series: {n_before:,} -> {len(series_keys):,} (dropped {n_before - len(series_keys):,})")


# =============================================================================
# SECTION 3: STATIC COVARIATES via StaticCovariatesTransformer
# =============================================================================
# The transformer is fitted ONCE on a compact frame that contains every
# category in every column, then applied lazily to each series inside
# __getitem__. Fitting on a compact frame rather than on all ~116K series
# keeps the fit cheap: ordinal encoding is per-column independent, so the
# transformer only needs to SEE every distinct value, not every combination.

print("\n" + "=" * 60)
print("SECTION 3: STATIC COVARIATES (StaticCovariatesTransformer)")
print("=" * 60)

static_df_all = static_df_all[static_covariates].astype(str)

for c in static_covariates:
    print(f"  {c}: {static_df_all[c].nunique()} categories")

# --- compact fit frame: every distinct value of every column, forward-filled
n_fit_rows = int(static_df_all.nunique().max())
fit_frame = pd.DataFrame({
    c: pd.Series(sorted(static_df_all[c].unique()))
         .reindex(range(n_fit_rows)).ffill()
    for c in static_covariates
}).astype(str)
print(f"\nFit frame: {n_fit_rows} rows covering all categories in all columns.")

_dummy_times = pd.date_range("2000-01-01", periods=2, freq="D")
fit_series = [
    TimeSeries.from_times_and_values(
        _dummy_times,
        np.zeros((2, 1), dtype=np.float32),
        columns=[target_col],
        static_covariates=fit_frame.iloc[[i]].reset_index(drop=True),
    )
    for i in range(n_fit_rows)
]

sc_transformer = StaticCovariatesTransformer(
    transformer_cat=OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    cols_cat=static_covariates,
)
sc_transformer.fit(fit_series)
del fit_series
gc.collect()
print("StaticCovariatesTransformer fitted.")

# per-series raw static frames, transformed lazily at read time
STATIC_RAW = [static_df_all.iloc[[i]].reset_index(drop=True) for i in range(len(static_df_all))]

# cardinalities feed categorical_embedding_sizes in Section 7
CATEGORY_COUNTS = {c: int(static_df_all[c].nunique()) for c in static_covariates}


# =============================================================================
# SECTION 4: SHARED FUTURE COVARIATES (festive + calendar, built in step 1)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: SHARED FUTURE COVARIATES")
print("=" * 60)

cal = pd.read_parquet(calendar_path)
cal[time_col] = pd.to_datetime(cal[time_col])
cal = cal.sort_values(time_col).reset_index(drop=True)

SHARED_COV = TimeSeries.from_dataframe(
    cal, time_col=time_col, value_cols=future_covariates,
    freq=FREQ, fill_missing_dates=False
).astype(np.float32)

print(f"Covariate calendar: {cal[time_col].min().date()} -> {cal[time_col].max().date()} "
      f"({len(cal)} days, {len(future_covariates)} cols)")

# the calendar must reach the end of the forecast horizon or predict() fails
required_end = FORECAST_END
if cal[time_col].max() < required_end:
    raise ValueError(
        f"Calendar ends {cal[time_col].max().date()} but the forecast needs "
        f"{required_end.date()}. Extend the source table before training."
    )

del cal
gc.collect()


# =============================================================================
# SECTION 5: LAZY SEQUENCES
# =============================================================================

class DiskLazyTargetSequence(collections.abc.Sequence):
    """
    Target series backed by per-series .npz files.

    Returns a SINGLE-component TimeSeries of RAW counts. No scaling: the
    Negative Binomial likelihood models counts on their native support, so
    there is nothing to scale and nothing to invert afterwards.

    Static covariates are attached raw and then passed through the fitted
    StaticCovariatesTransformer, so the encoding is identical for every
    series and is applied by the same Darts object at train and predict time.
    """

    def __init__(self, cache_dir, series_keys, static_raw, sc_transformer,
                 split="train", freq='D', cache_in_ram=True):
        self.cache_dir      = cache_dir
        self.series_keys    = series_keys
        self.static_raw     = static_raw
        self.sc_transformer = sc_transformer
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
            return self._ram[idx]

        key = self.series_keys[idx]
        with np.load(os.path.join(self.cache_dir, f"{safe_name(key)}.npz"),
                     allow_pickle=False) as z:
            sales = z[f"{self.split}_sales"]
            start = str(z[f"{self.split}_start"])

        times = pd.date_range(start=start, periods=len(sales), freq=self.freq)
        ts = TimeSeries.from_times_and_values(
            times,
            sales.reshape(-1, 1).astype(np.float32),   # RAW counts, 1 component
            columns=[target_col],
            static_covariates=self.static_raw[idx],
        )
        ts = self.sc_transformer.transform(ts)

        if self._ram is not None:
            self._ram[idx] = ts
        return ts


class DiskLazyWeightSequence(collections.abc.Sequence):
    """
    Per-timestep sample weights, aligned 1:1 with the target series.

    This is where the festive emphasis now lives. The old approach put a
    FESTIVE_FLAG in a second target component and read it inside a custom
    loss_fn -- impossible once a likelihood is set, because Darts computes
    negative log-likelihood internally and never calls loss_fn.
    """

    def __init__(self, cache_dir, series_keys, split="train", freq='D', cache_in_ram=True):
        self.cache_dir    = cache_dir
        self.series_keys  = series_keys
        self.split        = split
        self.freq         = freq
        self.cache_in_ram = cache_in_ram
        self._ram         = {} if cache_in_ram else None

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
            return self._ram[idx]

        key = self.series_keys[idx]
        with np.load(os.path.join(self.cache_dir, f"{safe_name(key)}.npz"),
                     allow_pickle=False) as z:
            weight = z[f"{self.split}_weight"]
            start  = str(z[f"{self.split}_start"])

        times = pd.date_range(start=start, periods=len(weight), freq=self.freq)
        ts = TimeSeries.from_times_and_values(
            times, weight.reshape(-1, 1).astype(np.float32), columns=["WEIGHT"]
        )
        if self._ram is not None:
            self._ram[idx] = ts
        return ts


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


train_seq        = DiskLazyTargetSequence(CACHE_DIR, series_keys, STATIC_RAW,
                                          sc_transformer, split="train", freq=FREQ)
train_cov_seq    = SharedCovSequence(SHARED_COV, len(series_keys))
train_weight_seq = DiskLazyWeightSequence(CACHE_DIR, series_keys, split="train", freq=FREQ)

val_mask    = list(has_val)
val_keys    = [k for k, h in zip(series_keys, val_mask) if h]
val_statics = [s for s, h in zip(STATIC_RAW,  val_mask) if h]

val_seq        = DiskLazyTargetSequence(CACHE_DIR, val_keys, val_statics,
                                        sc_transformer, split="val", freq=FREQ)
val_cov_seq    = SharedCovSequence(SHARED_COV, len(val_keys))
val_weight_seq = DiskLazyWeightSequence(CACHE_DIR, val_keys, split="val", freq=FREQ)

# --- smoke test: confirm one series is raw counts, 1 component, encoded statics
_probe = train_seq[0]
print(f"\nProbe series: {_probe.n_components} component(s), "
      f"{len(_probe)} steps, "
      f"values min={float(_probe.values().min()):.1f} max={float(_probe.values().max()):.1f}")
print("Encoded static covariates:")
print(_probe.static_covariates)
assert _probe.n_components == 1, "target must be single-component under a likelihood"
assert float(_probe.values().min()) >= 0, "count likelihood requires non-negative target"


# =============================================================================
# SECTION 6: SAVE ARTIFACTS NEEDED AT PREDICT TIME
# =============================================================================

SHARED_COV.to_pickle(os.path.join(CACHE_DIR, "shared_cov.pkl"))

import pickle
with open(os.path.join(CACHE_DIR, "static_cov_transformer.pkl"), "wb") as f:
    pickle.dump(sc_transformer, f)

print(f"\nSaved shared_cov.pkl and static_cov_transformer.pkl -> {CACHE_DIR}")


# =============================================================================
# SECTION 7: MODEL
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: MODEL")
print("=" * 60)

torch.set_float32_matmul_precision('high')

now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
MODEL_NAME = f"daily_tft_negbin_scooters_{now}"
print("Model name:", MODEL_NAME)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")

# --- categorical embeddings -------------------------------------------------
# THIS is what stops PARENT_DEALER_CODE being read as a continuous number.
# The transformer in Section 3 only turns categories into integers; without
# this argument the network would treat dealer 4102 as sitting numerically
# between 4101 and 4103. +1 on each count leaves room for the
# unknown_value=-1 slot from the OrdinalEncoder.
categorical_embedding_sizes = {
    c: (CATEGORY_COUNTS[c] + 1, min(50, (CATEGORY_COUNTS[c] + 2) // 2))
    for c in static_covariates
}
print("\ncategorical_embedding_sizes:")
for c, v in categorical_embedding_sizes.items():
    print(f"  {c}: {v}")

model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=4,
    num_attention_heads=4,
    dropout=0.05,

    batch_size=64,
    n_epochs=100,

    # --- the two changes that must move together ---------------------------
    likelihood=LIKELIHOOD,
    loss_fn=None,                        # ignored when likelihood is set
    use_reversible_instance_norm=False,  # MUST be False: RIN rescales the
                                         # target, which breaks a count
                                         # likelihood's non-negative integer
                                         # support

    categorical_embedding_sizes=categorical_embedding_sizes,

    random_state=42,
    add_relative_index=True,

    save_checkpoints=True,
    force_reset=True,
    model_name=MODEL_NAME,

    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 4,
        "limit_train_batches": 5000,
        "limit_val_batches": 500,
    },
)


# =============================================================================
# SECTION 8: TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: TRAINING")
print("=" * 60)

fit_kwargs = dict(
    series=train_seq,
    future_covariates=train_cov_seq,
    val_series=val_seq,
    val_future_covariates=val_cov_seq,
    max_samples_per_ts=400,
    dataloader_kwargs={"num_workers": 0, "pin_memory": True},
    verbose=True,
)

# sample_weight arrived in Darts 0.30. If this build predates it, the call
# raises TypeError and we fall back to unweighted training rather than
# silently dropping the festive emphasis without saying so.
try:
    model.fit(sample_weight=train_weight_seq,
              val_sample_weight=val_weight_seq,
              **fit_kwargs)
except TypeError as e:
    print("\n" + "!" * 60)
    print("sample_weight not supported by this Darts version:")
    print(f"  {e}")
    print("Falling back to UNWEIGHTED training. Festive days carry no extra")
    print("emphasis in this run -- upgrade Darts to restore it.")
    print("!" * 60 + "\n")
    model.fit(**fit_kwargs)


# =============================================================================
# SECTION 9: PREDICTION
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: PREDICTION")
print("=" * 60)

# With a likelihood, predict() samples from the predicted distribution.
# num_samples>1 gives the full predictive distribution; the mean is the point
# forecast. Raising the quantile taken here is the transparent lever for
# systematic under-forecasting -- q0.60 instead of the mean biases upward by a
# stated amount rather than by tuning a hidden loss term.
NUM_SAMPLES = 200

# History for the forecast must END on FORECAST_START - 1 (2026-08-31), so the
# 365-day encoder window reads 2025-09-01 -> 2026-08-31. val_seq ends exactly
# there, so it is the correct series to forecast from -- train_seq stops at
# TRAIN_END and would leave a four-month gap before FORECAST_START.
_hist_end = FORECAST_START - pd.Timedelta(days=1)
print(f"Forecasting from history ending {_hist_end.date()}")
print(f"Encoder window: {(FORECAST_START - pd.Timedelta(days=INPUT_CHUNK_LENGTH)).date()} "
      f"-> {_hist_end.date()}")
print(f"Series forecast: {len(val_keys):,} (those with a complete val strip)")

if len(val_keys) < len(series_keys):
    print(f"NOTE: {len(series_keys) - len(val_keys):,} series lack a full "
          f"{MIN_LEN}-day val strip and are not forecast here.")
    print("      They need a separate shorter-history treatment.")

preds = model.predict(
    n=TEST_HORIZON,
    series=val_seq,
    future_covariates=val_cov_seq,
    num_samples=NUM_SAMPLES,
    verbose=True,
)

OUT_DIR = os.path.join(os.getcwd(), "predictions_2026")
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for key, p in zip(val_keys, preds):
    mean_vals = p.values(copy=False).mean(axis=-1).ravel()
    q60_vals  = p.quantile(0.60).values(copy=False).ravel()
    rows.append(pd.DataFrame({
        group_col:   key,
        time_col:    p.time_index,
        "PRED_MEAN": mean_vals,
        "PRED_Q60":  q60_vals,
    }))

pred_df = pd.concat(rows, ignore_index=True)
out_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet")
pred_df.to_parquet(out_path, index=False)

print(f"Predictions written -> {out_path}")
print(f"  rows: {len(pred_df):,}")
print(f"  total PRED_MEAN: {pred_df['PRED_MEAN'].sum():,.0f} "
      f"({pred_df['PRED_MEAN'].sum()/1e5:.2f} lacs)")
print(f"  total PRED_Q60 : {pred_df['PRED_Q60'].sum():,.0f} "
      f"({pred_df['PRED_Q60'].sum()/1e5:.2f} lacs)")
