import numpy as np, os

H = 154
tot_last_H = 0.0
tot_365    = 0.0
for k in series_keys:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        s = z["val_sales"]          # ends 2026-07-30, day before FORECAST_START
    tot_last_H += s[-H:].sum()
    tot_365    += s[-365:].sum()

print(f"series counted          : {len(series_keys):,}")
print(f"ACTUAL sales, last {H}d : {tot_last_H:,.0f}  ({tot_last_H/1e5:.2f} lacs)")
print(f"ACTUAL sales, last 365d : {tot_365:,.0f}  ({tot_365/1e5:.2f} lacs)")
print()
print(f"model predicted         : 0.10 lacs")
print(f"you expected            : 1.90 lacs")


series counted          : 32,568
ACTUAL sales, last 154d : 173,041  (1.73 lacs)
ACTUAL sales, last 365d : 411,025  (4.11 lacs)

model predicted         : 0.10 lacs
you expected            : 1.90 lacs


print("MODEL_NAME  :", MODEL_NAME)
print("loss delta  :", best_model.model.criterion.delta)
print("festive_wt  :", best_model.model.criterion.festive_weight)

import numpy as np
seq  = DiskLazyTargetSequence(CACHE_DIR, series_keys[:500], scaler_stats,
                              STATIC_ENCODED[:500], split="val", freq=FREQ,
                              cache_in_ram=False)
covs = SharedCovSequence(SHARED_COV, 500)
p    = best_model.predict(n=HORIZON, series=seq, future_covariates=covs,
                          verbose=False, num_loader_workers=0)

raw = np.concatenate([x.values()[:, 0] for x in p])
print(f"SCALED preds: mean {raw.mean():.4f} median {np.median(raw):.4f} max {raw.max():.2f}")
print("training targets had mean 0.59")


import shutil
shutil.rmtree(os.path.join(DATA_ROOT, "scooter_predictions_2026"), ignore_errors=True)
os.makedirs(os.path.join(DATA_ROOT, "scooter_predictions_2026"), exist_ok=True)
print("stale prediction chunks cleared")

# actual for the same calendar window last year
import numpy as np, pandas as pd, os, torch

N_BT   = 91                        # May 1 -> Jul 30 2026
SAMPLE = predict_keys[:5000]
statics = predict_statics[:5000]

seq  = DiskLazyTargetSequence(CACHE_DIR, SAMPLE, scaler_stats, statics,
                              split="train", freq=FREQ, cache_in_ram=False)
covs = SharedCovSequence(SHARED_COV, len(SAMPLE))

with torch.no_grad():
    preds = best_model.predict(n=N_BT, series=seq, future_covariates=covs, verbose=False)

pred_tot = 0.0
for key, p in zip(SAMPLE, preds):
    lo, hi = scaler_stats[key]
    pred_tot += np.clip(p.values()[:, 0] * (hi - lo) + lo, 0, None).sum()

a_start, a_end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-07-30")
act_tot = 0.0
for key in SAMPLE:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:
        s  = z["val_sales"]; st = pd.Timestamp(str(z["val_start"]))
    idx = pd.date_range(st, periods=len(s), freq="D")
    act_tot += s[(idx >= a_start) & (idx <= a_end)].sum()

print(f"series in test         : {len(SAMPLE):,}")
print(f"PREDICTED May-Jul 2026 : {pred_tot:,.0f}")
print(f"ACTUAL    May-Jul 2026 : {act_tot:,.0f}")
print(f"ratio pred/actual      : {pred_tot/act_tot:.3f}   ({(pred_tot/act_tot-1)*100:+.1f}%)")