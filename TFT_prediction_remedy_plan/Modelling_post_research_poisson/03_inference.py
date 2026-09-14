"""
=============================================================================
STEP 3 -- STANDALONE INFERENCE
=============================================================================

Loads the BEST checkpoint and forecasts FORECAST_START..FORECAST_END.
Does not retrain, does not rebuild the cache, does not touch Snowflake.

Prerequisites (all produced by steps 1 and 2):
    series_cache_scooter/*.npz            per-series history
    series_cache_scooter/manifest.json
    series_cache_scooter/static_covariates.parquet
    series_cache_scooter/shared_cov.pkl   OR shared_calendar.parquet
    series_cache_scooter/static_cov_transformer.pkl
    darts_logs/<MODEL_NAME>/checkpoints/  trained model

Usage:
    python 03_inference.py --model-name daily_tft_negbin_scooters_2026-09-14_11_02_33
    python 03_inference.py --model-name <name> --limit 2000     # smoke test

WHY THIS IS FASTER THAN THE OLD SECTION 9
-----------------------------------------
  1. No sampling. predict_likelihood_parameters=True returns the Negative
     Binomial (mu, alpha) per timestep in ONE forward pass. The old code ran
     num_samples=200, i.e. 200 passes per series, then took an empirical mean.
     The mean of a NegBin IS mu exactly, and quantiles are closed-form from
     (mu, alpha) via scipy. Nothing is lost.
  2. No RAM cache. Each series is touched exactly once during inference, so
     caching Darts TimeSeries objects only consumes memory. They are
     xarray-backed and heavy; ~116K of them is many GB, which on a box with
     an HDD means swapping -- turning a millisecond read into a disk seek.
  3. Static covariates encoded ONCE up front, not per series. The old code
     called sc_transformer.transform() inside __getitem__, which is a full
     sklearn call plus DataFrame construction repeated ~116K times in pure
     single-threaded Python.
  4. Larger predict batch size, so the GPU is not left waiting on Python.

Blocks are written incrementally and skipped on re-run, so an interrupted run
resumes instead of starting over.
"""

import os
import gc
import json
import glob
import pickle
import argparse
import collections.abc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats

from darts import TimeSeries
from darts.models import TFTModel


# =============================================================================
# SECTION 0: CONFIG
# =============================================================================

BASE          = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
calendar_path = os.path.join(BASE, "shared_calendar.parquet")
roles_path    = os.path.join(BASE, "column_roles.json")
CACHE_DIR     = os.path.join(os.getcwd(), "series_cache_scooter")
OUT_DIR       = os.path.join(os.getcwd(), "predictions_2026")

parser = argparse.ArgumentParser(description="Standalone TFT inference from the best checkpoint.")
parser.add_argument("--model-name", required=True,
                    help="Model name as passed to TFTModel(model_name=...) during training.")
parser.add_argument("--batch-size", type=int, default=512,
                    help="Predict batch size. Raise until GPU memory complains.")
parser.add_argument("--block", type=int, default=5000,
                    help="Series per block. Each block is written to disk before the next starts.")
parser.add_argument("--limit", type=int, default=None,
                    help="Only forecast the first N series. Use for a smoke test.")
parser.add_argument("--work-dir", default=None,
                    help="Directory containing darts_logs/. Defaults to cwd.")
args = parser.parse_args()

MODEL_NAME         = args.model_name
PREDICT_BATCH_SIZE = args.batch_size
PREDICT_BLOCK      = args.block
QUANTILES          = [0.50, 0.60, 0.70, 0.80]

if args.work_dir:
    os.chdir(args.work_dir)

with open(roles_path) as f:
    ROLES = json.load(f)

time_col          = ROLES["time_col"]
group_col         = ROLES["group_col"]
target_col        = ROLES["target_col"]
FREQ              = ROLES["freq"]
static_covariates = ROLES["static_covariates"]
future_covariates = ROLES["future_covariates"]

TRAIN_END      = pd.Timestamp(ROLES.get("train_end",      "2026-04-30"))
VAL_START      = pd.Timestamp(ROLES.get("val_start",      "2026-05-01"))
VAL_END        = pd.Timestamp(ROLES.get("val_end",        "2026-08-31"))
FORECAST_START = pd.Timestamp(ROLES.get("forecast_start", "2026-09-01"))
FORECAST_END   = pd.Timestamp(ROLES.get("forecast_end",   "2026-12-07"))

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 98
TEST_HORIZON        = (FORECAST_END - FORECAST_START).days + 1

assert TEST_HORIZON == OUTPUT_CHUNK_LENGTH, (
    f"Forecast window is {TEST_HORIZON} days but OUTPUT_CHUNK_LENGTH is "
    f"{OUTPUT_CHUNK_LENGTH}. Predicting n > output_chunk_length forces Darts "
    f"into auto-regressive rollout, which compounds error across the festive peak."
)

val_window_days = (VAL_END - VAL_START).days + 1
warmup_days     = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH - val_window_days
warmup_start    = VAL_START - pd.Timedelta(days=warmup_days)
MIN_LEN         = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH


def safe_name(key):
    return str(key).replace("<>", "_").replace("/", "_").replace("\\", "_")


print("=" * 68)
print("STANDALONE INFERENCE")
print("=" * 68)
print(f"Model        : {MODEL_NAME}")
print(f"Forecast     : {FORECAST_START.date()} -> {FORECAST_END.date()} ({TEST_HORIZON} days)")
print(f"History ends : {(FORECAST_START - pd.Timedelta(days=1)).date()}")
print(f"Encoder reads: {(FORECAST_START - pd.Timedelta(days=INPUT_CHUNK_LENGTH)).date()} "
      f"-> {(FORECAST_START - pd.Timedelta(days=1)).date()}")
print(f"Batch size   : {PREDICT_BATCH_SIZE} | block: {PREDICT_BLOCK:,}")


# =============================================================================
# SECTION 1: LAZY SEQUENCES (inference variants)
# =============================================================================

class ForecastHistorySequence(collections.abc.Sequence):
    """
    Series history to forecast FROM. Reads the cached "val" split, which by
    construction ends on VAL_END == FORECAST_START - 1, so predict() continues
    from exactly the right date and the 365-day encoder window covers the
    twelve months up to the forecast.

    Differences from the training-time sequence, both deliberate:
      - cache_in_ram=False. Each series is read once during inference; caching
        would only consume memory.
      - static covariates are already ENCODED, so no sklearn transform runs
        per series.
    """

    def __init__(self, cache_dir, series_keys, encoded_statics, split="val", freq='D'):
        self.cache_dir       = cache_dir
        self.series_keys     = series_keys
        self.encoded_statics = encoded_statics
        self.split           = split
        self.freq            = freq

    def __len__(self):
        return len(self.series_keys)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        key = self.series_keys[idx]
        with np.load(os.path.join(self.cache_dir, f"{safe_name(key)}.npz"),
                     allow_pickle=False) as z:
            sales = z[f"{self.split}_sales"]
            start = str(z[f"{self.split}_start"])

        times = pd.date_range(start=start, periods=len(sales), freq=self.freq)
        return TimeSeries.from_times_and_values(
            times,
            sales.reshape(-1, 1).astype(np.float32),
            columns=[target_col],
            static_covariates=self.encoded_statics[idx],
        )


class SharedCovSequence(collections.abc.Sequence):
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


# =============================================================================
# SECTION 2: LOAD ARTIFACTS
# =============================================================================

print("\n" + "=" * 68)
print("SECTION 2: LOADING ARTIFACTS")
print("=" * 68)

with open(os.path.join(CACHE_DIR, "manifest.json")) as f:
    manifest = json.load(f)

series_keys_all = manifest["series_keys"]
has_val_all     = manifest["has_val"]
static_df_all   = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))

# Re-apply the SAME dead-series filter used at training time, so indices stay
# aligned with static_covariates.parquet. If this diverges, static covariates
# get attached to the wrong series -- silently.
DROP_NEVER_SOLD = True
DORMANT_DAYS    = None

keep = []
for k in series_keys_all:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        s = z["train_sales"]
    if DROP_NEVER_SOLD and s.sum() == 0:
        keep.append(False); continue
    if DORMANT_DAYS and s[-DORMANT_DAYS:].sum() == 0:
        keep.append(False); continue
    keep.append(True)

series_keys   = [k for k, m in zip(series_keys_all, keep) if m]
has_val       = [h for h, m in zip(has_val_all,    keep) if m]
static_df_all = static_df_all.loc[keep].reset_index(drop=True)
assert len(series_keys) == len(has_val) == len(static_df_all), "index misalignment"

print(f"Series in manifest : {len(series_keys_all):,}")
print(f"After dead-filter  : {len(series_keys):,}")

# --- series that can actually be forecast -----------------------------------
forecast_keys = [k for k, h in zip(series_keys, has_val) if h]
forecast_idx  = [i for i, h in enumerate(has_val) if h]
print(f"Forecastable       : {len(forecast_keys):,}")

if len(forecast_keys) < len(series_keys):
    print(f"NOTE: {len(series_keys) - len(forecast_keys):,} series lack a full "
          f"{MIN_LEN}-day history ending {VAL_END.date()} and are NOT forecast.")
    print("      They are absent from the output entirely -- handle separately")
    print("      if they carry meaningful volume.")

if args.limit:
    forecast_keys = forecast_keys[:args.limit]
    forecast_idx  = forecast_idx[:args.limit]
    print(f"\n--limit set: forecasting only {len(forecast_keys):,} series (smoke test).")

# --- static covariate transformer -------------------------------------------
with open(os.path.join(CACHE_DIR, "static_cov_transformer.pkl"), "rb") as f:
    sc_transformer = pickle.load(f)
print("Loaded static_cov_transformer.pkl")

# --- shared future covariates -----------------------------------------------
shared_cov_pkl = os.path.join(CACHE_DIR, "shared_cov.pkl")
if os.path.exists(shared_cov_pkl):
    SHARED_COV = TimeSeries.from_pickle(shared_cov_pkl)
    print("Loaded shared_cov.pkl")
else:
    cal = pd.read_parquet(calendar_path)
    cal[time_col] = pd.to_datetime(cal[time_col])
    cal = cal.sort_values(time_col).reset_index(drop=True)
    SHARED_COV = TimeSeries.from_dataframe(
        cal, time_col=time_col, value_cols=future_covariates,
        freq=FREQ, fill_missing_dates=False
    ).astype(np.float32)
    del cal; gc.collect()
    print("Rebuilt covariates from shared_calendar.parquet")

if SHARED_COV.end_time() < FORECAST_END:
    raise ValueError(
        f"Future covariates end {SHARED_COV.end_time().date()} but the forecast "
        f"needs {FORECAST_END.date()}. Re-run 01_data_chunking.py against a "
        f"source table that extends far enough."
    )
print(f"Covariates cover   : {SHARED_COV.start_time().date()} -> {SHARED_COV.end_time().date()}")


# =============================================================================
# SECTION 3: PRE-ENCODE STATIC COVARIATES (once, not per series)
# =============================================================================

print("\n" + "=" * 68)
print("SECTION 3: PRE-ENCODING STATIC COVARIATES")
print("=" * 68)

static_df_all = static_df_all[static_covariates].astype(str)
_dummy_t = pd.date_range("2000-01-01", periods=2, freq="D")

encoded_statics = []
for i in forecast_idx:
    _t = TimeSeries.from_times_and_values(
        _dummy_t, np.zeros((2, 1), dtype=np.float32),
        columns=[target_col],
        static_covariates=static_df_all.iloc[[i]].reset_index(drop=True),
    )
    encoded_statics.append(sc_transformer.transform(_t).static_covariates)

print(f"Encoded {len(encoded_statics):,} static frames "
      f"(done once here, instead of once per series inside the data loader).")


# =============================================================================
# SECTION 4: LOAD THE BEST CHECKPOINT
# =============================================================================

print("\n" + "=" * 68)
print("SECTION 4: LOADING BEST CHECKPOINT")
print("=" * 68)
print("NOTE: model.predict() on an in-memory model uses the LAST epoch's")
print("      weights. Lightning's EarlyStopping does not restore the best")
print("      weights the way Keras does -- with patience=5 the final weights")
print("      are 5 epochs past the selected checkpoint. Loading best=True.")

model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)
print(f"Loaded {MODEL_NAME} (best=True)")

try:
    model.trainer_params["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
    model.trainer_params["devices"] = 1
    model.trainer_params.pop("limit_val_batches", None)
    model.trainer_params.pop("limit_train_batches", None)
except Exception as e:
    print(f"(could not adjust trainer params: {e})")

print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")


# =============================================================================
# SECTION 5: FORECAST IN BLOCKS
# =============================================================================

print("\n" + "=" * 68)
print("SECTION 5: FORECASTING")
print("=" * 68)

os.makedirs(OUT_DIR, exist_ok=True)


def negbin_summary(param_ts, quantiles):
    """
    Convert Negative Binomial distribution parameters into a mean and
    quantiles, closed-form -- no sampling.

    Darts' NegativeBinomialLikelihood emits two components: (mu, alpha),
    mean and dispersion. scipy parameterises nbinom as (n, p), so:
        n = 1 / alpha
        p = n / (n + mu)
    The distribution mean is mu exactly, which is why 200-sample Monte Carlo
    was pure waste -- it was estimating a number available in closed form.
    """
    vals  = param_ts.values(copy=False)
    mu    = vals[:, 0].astype(np.float64)
    alpha = np.clip(vals[:, 1].astype(np.float64), 1e-6, None)

    n = 1.0 / alpha
    p = n / (n + mu)

    out = {"PRED_MEAN": mu}
    for q in quantiles:
        out[f"PRED_Q{int(q * 100)}"] = stats.nbinom.ppf(q, n, p).astype(np.float64)
    return out


n_total   = len(forecast_keys)
n_blocks  = (n_total + PREDICT_BLOCK - 1) // PREDICT_BLOCK
block_tag = f"{MODEL_NAME}_block"
block_paths, t0, first_block_checked = [], datetime.now(), False

for bi in range(n_blocks):
    lo, hi = bi * PREDICT_BLOCK, min((bi + 1) * PREDICT_BLOCK, n_total)
    block_path = os.path.join(OUT_DIR, f"{block_tag}_{bi:04d}.parquet")
    block_paths.append(block_path)

    if os.path.exists(block_path):
        print(f"[{bi+1}/{n_blocks}] exists, skipping.")
        continue

    blk_seq = ForecastHistorySequence(
        CACHE_DIR, forecast_keys[lo:hi], encoded_statics[lo:hi], split="val", freq=FREQ
    )
    blk_cov = SharedCovSequence(SHARED_COV, hi - lo)

    preds = model.predict(
        n=TEST_HORIZON,
        series=blk_seq,
        future_covariates=blk_cov,
        predict_likelihood_parameters=True,   # one pass, not num_samples passes
        num_samples=1,                        # required with the flag above
        batch_size=PREDICT_BATCH_SIZE,
        verbose=False,
    )

    # sanity-check the parameter layout ONCE, before writing 116K rows of it
    if not first_block_checked:
        comps = list(preds[0].components)
        print(f"\n  Likelihood parameter components: {comps}")
        if len(comps) != 2:
            raise ValueError(
                f"Expected 2 NegBin parameters, got {len(comps)}: {comps}. "
                "Check the Darts version's parameter layout before trusting output."
            )
        print(f"  Forecast window: {preds[0].start_time().date()} -> "
              f"{preds[0].end_time().date()}")
        assert preds[0].start_time() == FORECAST_START, (
            f"Forecast starts {preds[0].start_time().date()}, expected "
            f"{FORECAST_START.date()}. The history series does not end where "
            f"it should."
        )
        first_block_checked = True

    rows = [
        pd.DataFrame({group_col: key, time_col: p.time_index,
                      **negbin_summary(p, QUANTILES)})
        for key, p in zip(forecast_keys[lo:hi], preds)
    ]
    pd.concat(rows, ignore_index=True).to_parquet(block_path, index=False)

    el   = (datetime.now() - t0).total_seconds()
    rate = hi / el if el else 0
    eta  = (n_total - hi) / rate / 60 if rate else 0
    print(f"[{bi+1}/{n_blocks}] {hi:,}/{n_total:,} series | "
          f"{rate:,.0f} series/s | elapsed {el/60:.1f} min | ETA {eta:.0f} min")

    del preds, rows, blk_seq, blk_cov
    gc.collect()


# =============================================================================
# SECTION 6: STITCH AND REPORT
# =============================================================================

print("\n" + "=" * 68)
print("SECTION 6: WRITING FINAL OUTPUT")
print("=" * 68)

pred_df  = pd.concat([pd.read_parquet(bp) for bp in block_paths], ignore_index=True)
out_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet")
pred_df.to_parquet(out_path, index=False)

elapsed = (datetime.now() - t0).total_seconds()
print(f"Written -> {out_path}")
print(f"  series  : {pred_df[group_col].nunique():,}")
print(f"  rows    : {len(pred_df):,}  (expected {len(forecast_keys) * TEST_HORIZON:,})")
print(f"  dates   : {pred_df[time_col].min().date()} -> {pred_df[time_col].max().date()}")
print(f"  elapsed : {elapsed/60:.1f} min")

print("\n  Forecast totals:")
for c in ["PRED_MEAN"] + [f"PRED_Q{int(q*100)}" for q in QUANTILES]:
    tot = pred_df[c].sum()
    print(f"    {c:>11}: {tot:>12,.0f}  ({tot/1e5:.2f} lacs)")

print("\n  PRED_MEAN is the distribution mean -- the unbiased point forecast.")
print("  The higher quantiles are the transparent lever against systematic")
print("  under-forecasting: reporting Q60 or Q70 states the upward bias as a")
print("  number rather than burying it in a loss function.")

print("\nTo remove the intermediate block files once satisfied:")
print(f"  del {os.path.join(OUT_DIR, block_tag)}_*.parquet")
