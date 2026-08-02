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