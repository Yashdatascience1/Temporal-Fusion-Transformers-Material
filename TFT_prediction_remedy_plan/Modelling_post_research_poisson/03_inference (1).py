"""
Standalone inference: load the best checkpoint, forecast Sep 1 - Dec 7 2026.

Run:  python 03_inference.py
"""

import os, json, pickle, gc, collections.abc
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from darts import TimeSeries
from darts.models import TFTModel


# ---------------- CONFIG ----------------
MODEL_NAME = "PUT_YOUR_MODEL_NAME_HERE"     # e.g. daily_tft_negbin_scooters_2026-09-14_11_02_33

BASE       = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
CACHE_DIR  = os.path.join(os.getcwd(), "series_cache_scooter")
OUT_DIR    = os.path.join(os.getcwd(), "predictions_2026")

BATCH_SIZE = 512        # raise until GPU memory complains
BLOCK      = 5000       # series per block, written to disk as it goes
LIMIT      = None       # set to e.g. 2000 for a quick smoke test

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)

time_col, group_col, target_col = ROLES["time_col"], ROLES["group_col"], ROLES["target_col"]
FREQ = ROLES["freq"]
static_covariates = ROLES["static_covariates"]

FORECAST_START = pd.Timestamp(ROLES.get("forecast_start", "2026-09-01"))
FORECAST_END   = pd.Timestamp(ROLES.get("forecast_end",   "2026-12-07"))
HORIZON        = (FORECAST_END - FORECAST_START).days + 1     # 98

safe_name = lambda k: str(k).replace("<>", "_").replace("/", "_").replace("\\", "_")


# ---------------- SEQUENCES ----------------
class History(collections.abc.Sequence):
    """Series history to forecast from. Reads the cached 'val' split,
    which ends on the day before FORECAST_START."""

    def __init__(self, keys, statics):
        self.keys, self.statics = keys, statics

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        with np.load(os.path.join(CACHE_DIR, f"{safe_name(self.keys[i])}.npz")) as z:
            sales, start = z["val_sales"], str(z["val_start"])
        return TimeSeries.from_times_and_values(
            pd.date_range(start, periods=len(sales), freq=FREQ),
            sales.reshape(-1, 1).astype(np.float32),
            columns=[target_col],
            static_covariates=self.statics[i],
        )


class SharedCov(collections.abc.Sequence):
    def __init__(self, cov, n):
        self.cov, self.n = cov, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self.cov for _ in range(*i.indices(self.n))]
        return self.cov


# ---------------- LOAD ----------------
print(f"Forecast: {FORECAST_START.date()} -> {FORECAST_END.date()} ({HORIZON} days)")

with open(os.path.join(CACHE_DIR, "manifest.json")) as f:
    manifest = json.load(f)

static_df = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))

# same dead-series filter as training, so indices stay aligned with static_df
keep = []
for k in manifest["series_keys"]:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        keep.append(z["train_sales"].sum() > 0)

series_keys = [k for k, m in zip(manifest["series_keys"], keep) if m]
has_val     = [h for h, m in zip(manifest["has_val"],     keep) if m]
static_df   = static_df.loc[keep].reset_index(drop=True)[static_covariates].astype(str)

keys = [k for k, h in zip(series_keys, has_val) if h]
idxs = [i for i, h in enumerate(has_val) if h]
if LIMIT:
    keys, idxs = keys[:LIMIT], idxs[:LIMIT]

print(f"Series to forecast: {len(keys):,}")
if len(keys) < len(series_keys):
    print(f"  ({len(series_keys) - len(keys):,} lack enough history and are skipped)")

with open(os.path.join(CACHE_DIR, "static_cov_transformer.pkl"), "rb") as f:
    sc_transformer = pickle.load(f)

SHARED_COV = TimeSeries.from_pickle(os.path.join(CACHE_DIR, "shared_cov.pkl"))
if SHARED_COV.end_time() < FORECAST_END:
    raise ValueError(f"Covariates end {SHARED_COV.end_time().date()}, need {FORECAST_END.date()}")

# encode static covariates once, not once per series inside the loader
dummy_t = pd.date_range("2000-01-01", periods=2, freq="D")
statics = []
for i in idxs:
    t = TimeSeries.from_times_and_values(
        dummy_t, np.zeros((2, 1), dtype=np.float32), columns=[target_col],
        static_covariates=static_df.iloc[[i]].reset_index(drop=True),
    )
    statics.append(sc_transformer.transform(t).static_covariates)

# load BEST weights -- model.predict() on an in-memory model uses the LAST
# epoch, which with patience=5 is 5 epochs past what early stopping picked
model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)
print(f"Loaded best checkpoint: {MODEL_NAME}")


# ---------------- FORECAST ----------------
def summarise(p):
    """NegBin params (mu, alpha) -> mean and quantiles, closed form.
    The mean IS mu, so sampling 200 times to estimate it was wasted work."""
    v = p.values(copy=False)
    mu = v[:, 0].astype(np.float64)
    alpha = np.clip(v[:, 1].astype(np.float64), 1e-6, None)
    n = 1.0 / alpha
    pr = n / (n + mu)
    out = {"PRED_MEAN": mu}
    for q in (0.60, 0.70):
        out[f"PRED_Q{int(q*100)}"] = stats.nbinom.ppf(q, n, pr)
    return out


os.makedirs(OUT_DIR, exist_ok=True)
n_blocks = (len(keys) + BLOCK - 1) // BLOCK
paths, t0, checked = [], datetime.now(), False

for b in range(n_blocks):
    lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(keys))
    path = os.path.join(OUT_DIR, f"{MODEL_NAME}_block_{b:04d}.parquet")
    paths.append(path)

    if os.path.exists(path):
        print(f"[{b+1}/{n_blocks}] exists, skipping")
        continue

    preds = model.predict(
        n=HORIZON,
        series=History(keys[lo:hi], statics[lo:hi]),
        future_covariates=SharedCov(SHARED_COV, hi - lo),
        predict_likelihood_parameters=True,   # 1 pass instead of num_samples passes
        num_samples=1,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    if not checked:   # fail fast if dates or params are wrong
        assert preds[0].start_time() == FORECAST_START, \
            f"forecast starts {preds[0].start_time().date()}, expected {FORECAST_START.date()}"
        print(f"  params: {list(preds[0].components)}")
        checked = True

    pd.concat(
        [pd.DataFrame({group_col: k, time_col: p.time_index, **summarise(p)})
         for k, p in zip(keys[lo:hi], preds)],
        ignore_index=True,
    ).to_parquet(path, index=False)

    el = (datetime.now() - t0).total_seconds()
    print(f"[{b+1}/{n_blocks}] {hi:,}/{len(keys):,} | {hi/el:,.0f} series/s | "
          f"ETA {(len(keys)-hi)/(hi/el)/60:.0f} min")

    del preds
    gc.collect()


# ---------------- OUTPUT ----------------
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
out = os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet")
df.to_parquet(out, index=False)

print(f"\nWritten -> {out}")
print(f"  rows: {len(df):,} | elapsed: {(datetime.now()-t0).total_seconds()/60:.1f} min")
for c in ["PRED_MEAN", "PRED_Q60", "PRED_Q70"]:
    print(f"  {c}: {df[c].sum():,.0f} ({df[c].sum()/1e5:.2f} lacs)")
