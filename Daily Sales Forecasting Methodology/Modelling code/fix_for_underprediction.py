import numpy as np, pandas as pd, os, torch

SAMPLE, statics = predict_keys[:5000], predict_statics[:5000]
N_BT = 91
a_start, a_end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-07-30")

seq  = DiskLazyTargetSequence(CACHE_DIR, SAMPLE, scaler_stats, statics,
                              split="train", freq=FREQ, cache_in_ram=False)
covs = SharedCovSequence(SHARED_COV, len(SAMPLE))
with torch.no_grad():
    preds = best_model.predict(n=N_BT, series=seq, future_covariates=covs, verbose=False)

ctx365 = ctx90 = pred_d = act_d = 0.0
for key, p in zip(SAMPLE, preds):
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:
        tr = z["train_sales"]; va = z["val_sales"]; st = pd.Timestamp(str(z["val_start"]))
    ctx365 += tr[-365:].sum() / 365
    ctx90  += tr[-90:].sum()  / 90
    lo, hi = scaler_stats[key]
    pred_d += np.clip(p.values()[:, 0] * (hi - lo) + lo, 0, None).sum() / N_BT
    idx = pd.date_range(st, periods=len(va), freq="D")
    act_d += va[(idx >= a_start) & (idx <= a_end)].sum() / N_BT

print("daily totals across the sample")
print(f"  trailing 365d mean (RIN anchor) : {ctx365:9,.0f}")
print(f"  trailing  90d mean (recent)     : {ctx90:9,.0f}")
print(f"  PREDICTED May-Jul               : {pred_d:9,.0f}")
print(f"  ACTUAL    May-Jul               : {act_d:9,.0f}")
print()
print(f"  pred / 365d-anchor : {pred_d/ctx365:.3f}")
print(f"  pred / 90d-recent  : {pred_d/ctx90:.3f}")
print(f"  actual / 365d      : {act_d/ctx365:.3f}")

####################################################

import numpy as np, pandas as pd, os

monthly = {}
for key in predict_keys:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:
        va = z["val_sales"]; st = pd.Timestamp(str(z["val_start"]))
    idx = pd.date_range(st, periods=len(va), freq="D")
    s = pd.Series(va, index=idx)
    for per, v in s.groupby(s.index.to_period("M")).sum().items():
        monthly[per] = monthly.get(per, 0.0) + v

prev = None
print("actual monthly totals:")
for per in sorted(monthly):
    v = monthly[per]
    py = per - 12
    yoy = f"   YoY {(v/monthly[py]-1)*100:+6.1f}%" if py in monthly and monthly[py] > 0 else ""
    mom = f"  MoM {(v/prev-1)*100:+6.1f}%" if prev else ""
    print(f"  {per}: {v:>10,.0f}{mom}{yoy}")
    prev = v