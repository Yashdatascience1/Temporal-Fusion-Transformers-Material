"""
Validation backtest: predict 2026-05-26 to 2026-08-31 and compare to actuals.

The cached "val" strip is 463 days (2025-05-26 -> 2026-08-31). This truncates
each series to its FIRST 365 days (the encoder window, ending 2026-05-25) and
forecasts 98 days forward -- landing exactly on the scored window. Those dates
were never trained on, so this is a clean out-of-sample check and the only
verifiable accuracy number available before the festive season happens.

Run:  python 09_validation_backtest.py
"""

import os, json, glob, pickle, gc, collections.abc
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from darts import TimeSeries
from darts.models import TFTModel

# ---------------- CONFIG ----------------
MODEL_NAME = "PUT_YOUR_MODEL_NAME_HERE"

BASE       = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
CACHE_DIR  = os.path.join(os.getcwd(), "series_cache_scooter")
OUT_DIR    = os.path.join(os.getcwd(), "predictions_2026")

ICL, HORIZON = 365, 98
BATCH_SIZE, BLOCK = 512, 5000
LIMIT = None                      # set e.g. 5000 for a quick check

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)
time_col, group_col, target_col = ROLES["time_col"], ROLES["group_col"], ROLES["target_col"]
FREQ, static_covariates = ROLES["freq"], ROLES["static_covariates"]

safe_name = lambda k: str(k).replace("<>", "_").replace("/", "_").replace("\\", "_")


class TruncatedHistory(collections.abc.Sequence):
    """Val strip cut to its first ICL days -- the encoder window only.
    Everything after that is what we are trying to predict, so it must not
    be handed to the model."""

    def __init__(self, keys, statics, icl):
        self.keys, self.statics, self.icl = keys, statics, icl

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        with np.load(os.path.join(CACHE_DIR, f"{safe_name(self.keys[i])}.npz")) as z:
            sales, start = z["val_sales"], str(z["val_start"])
        sales = sales[:self.icl]                      # <-- the truncation
        return TimeSeries.from_times_and_values(
            pd.date_range(start, periods=len(sales), freq=FREQ),
            sales.reshape(-1, 1).astype(np.float32),
            columns=[target_col], static_covariates=self.statics[i],
        )


class SharedCov(collections.abc.Sequence):
    def __init__(self, cov, n): self.cov, self.n = cov, n
    def __len__(self): return self.n
    def __getitem__(self, i):
        if isinstance(i, slice): return [self.cov for _ in range(*i.indices(self.n))]
        return self.cov


# ---------------- LOAD ----------------
with open(os.path.join(CACHE_DIR, "manifest.json")) as f:
    manifest = json.load(f)

static_df = pd.read_parquet(os.path.join(CACHE_DIR, "static_covariates.parquet"))
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
print(f"Series: {len(keys):,}")

with open(os.path.join(CACHE_DIR, "static_cov_transformer.pkl"), "rb") as f:
    sc_transformer = pickle.load(f)

dummy_t = pd.date_range("2000-01-01", periods=2, freq="D")
statics = []
for i in idxs:
    t = TimeSeries.from_times_and_values(
        dummy_t, np.zeros((2, 1), dtype=np.float32), columns=[target_col],
        static_covariates=static_df.iloc[[i]].reset_index(drop=True))
    statics.append(sc_transformer.transform(t).static_covariates)

SHARED_COV = TimeSeries.from_pickle(os.path.join(CACHE_DIR, "shared_cov.pkl"))
model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)
print(f"Loaded best checkpoint: {MODEL_NAME}")


# ---------------- PREDICT ----------------
def summarise(p):
    v = p.values(copy=False)
    mu = v[:, 0].astype(np.float64)
    alpha = np.clip(v[:, 1].astype(np.float64), 1e-6, None)
    n = 1.0 / alpha
    pr = n / (n + mu)
    return {"PRED_MEAN": mu,
            "PRED_Q60": stats.nbinom.ppf(0.60, n, pr),
            "PRED_Q70": stats.nbinom.ppf(0.70, n, pr)}


os.makedirs(OUT_DIR, exist_ok=True)
n_blocks = (len(keys) + BLOCK - 1) // BLOCK
paths, t0, checked = [], datetime.now(), False

for b in range(n_blocks):
    lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(keys))
    path = os.path.join(OUT_DIR, f"{MODEL_NAME}_valbt_{b:04d}.parquet")
    paths.append(path)
    if os.path.exists(path):
        print(f"[{b+1}/{n_blocks}] exists, skipping"); continue

    preds = model.predict(
        n=HORIZON,
        series=TruncatedHistory(keys[lo:hi], statics[lo:hi], ICL),
        future_covariates=SharedCov(SHARED_COV, hi - lo),
        predict_likelihood_parameters=True, num_samples=1,
        batch_size=BATCH_SIZE, verbose=False,
    )
    if not checked:
        print(f"  predicting {preds[0].start_time().date()} -> {preds[0].end_time().date()}")
        assert preds[0].start_time() == pd.Timestamp("2026-05-26"), \
            f"expected 2026-05-26, got {preds[0].start_time().date()}"
        checked = True

    pd.concat([pd.DataFrame({group_col: k, time_col: p.time_index, **summarise(p)})
               for k, p in zip(keys[lo:hi], preds)], ignore_index=True
              ).to_parquet(path, index=False)
    el = (datetime.now() - t0).total_seconds()
    print(f"[{b+1}/{n_blocks}] {hi:,}/{len(keys):,} | {hi/el:,.0f} series/s")
    del preds; gc.collect()

pred = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
pred[time_col] = pd.to_datetime(pred[time_col])
pred_daily = pred.groupby(time_col)["PRED_MEAN"].sum().sort_index()


# ---------------- ACTUALS ----------------
files = sorted(glob.glob(os.path.join(BASE, "local_train_scooter_data", "chunk_*.parquet")))
hist = pd.concat([pd.read_parquet(f, columns=[time_col, target_col]) for f in files],
                 ignore_index=True)
hist[time_col] = pd.to_datetime(hist[time_col])
act_daily = hist.groupby(time_col)[target_col].sum().sort_index()

PLOT_START = pd.Timestamp("2026-05-01")
act_plot = act_daily[(act_daily.index >= PLOT_START) & (act_daily.index <= pd.Timestamp("2026-08-31"))]
common = pred_daily.index.intersection(act_daily.index)
a, p = act_daily.loc[common], pred_daily.loc[common]


# ---------------- ACCURACY ----------------
wmape = np.abs(p - a).sum() / a.sum()
bias  = (p.sum() - a.sum()) / a.sum()
print("\n" + "=" * 60)
print("ACCURACY on the aggregate daily series (98 days)")
print("=" * 60)
print(f"  actual    : {a.sum():>12,.0f} ({a.sum()/1e5:.2f} lacs)")
print(f"  predicted : {p.sum():>12,.0f} ({p.sum()/1e5:.2f} lacs)")
print(f"  bias      : {bias*100:>+11.1f}%")
print(f"  WMAPE     : {wmape*100:>11.1f}%   -> accuracy {100-wmape*100:.1f}%")
print("\n  NOTE: this is accuracy on the AGGREGATE daily total, which is far")
print("  easier than per-series-per-day. Treat it as an upper bound, not as")
print("  the 1-MAPE number reported on Runner SKUs.")

print("\n  Monthly:")
for m in sorted(set(common.strftime("%Y-%m"))):
    am, pm = a[a.index.strftime("%Y-%m") == m].sum(), p[p.index.strftime("%Y-%m") == m].sum()
    print(f"    {m}: actual {am/1e5:>6.2f} | pred {pm/1e5:>6.2f} lacs | {(pm/am-1)*100:>+7.1f}%")


# ---------------- PLOT ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=act_plot.index, y=act_plot.values, mode="lines",
                         name="Actual", line=dict(color="#1f77b4", width=2),
                         hovertemplate="%{x|%d %b} (%{x|%a})<br>actual %{y:,.0f}<extra></extra>"))
fig.add_trace(go.Scatter(x=pred_daily.index, y=pred_daily.values, mode="lines",
                         name="Predicted", line=dict(color="#d62728", width=2, dash="dash"),
                         hovertemplate="%{x|%d %b}<br>pred %{y:,.0f}<extra></extra>"))

fig.add_vrect(x0=PLOT_START, x1=pd.Timestamp("2026-05-25"),
              fillcolor="grey", opacity=0.12, line_width=0,
              annotation_text="encoder window (no prediction)",
              annotation_position="top left", annotation_font_size=10)

fig.update_layout(
    title=f"Validation backtest — predicted vs actual, 26 May to 31 Aug 2026"
          f"  (bias {bias*100:+.1f}%, WMAPE {wmape*100:.1f}%)",
    xaxis_title="Date", yaxis_title="Units",
    height=560, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
html = os.path.join(OUT_DIR, "validation_backtest.html")
fig.write_html(html)
print(f"\nChart -> {html}")
