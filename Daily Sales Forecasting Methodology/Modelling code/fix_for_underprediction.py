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

#######Test 1 output######
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
daily totals across the sample
  trailing 365d mean (RIN anchor) :       155
  trailing  90d mean (recent)     :       174
  PREDICTED May-Jul               :       126
  ACTUAL    May-Jul               :       164

  pred / 365d-anchor : 0.813
  pred / 90d-recent  : 0.724
  actual / 365d      : 1.062


############Test 2 output###########
actual monthly totals:
  2025-02:      2,693
  2025-03:     25,554  MoM +848.9%
  2025-04:     23,392  MoM   -8.5%
  2025-05:     21,025  MoM  -10.1%
  2025-06:     23,056  MoM   +9.7%
  2025-07:     27,547  MoM  +19.5%
  2025-08:     25,588  MoM   -7.1%
  2025-09:     29,054  MoM  +13.5%
  2025-10:     67,877  MoM +133.6%
  2025-11:     30,650  MoM  -54.8%
  2025-12:     24,137  MoM  -21.2%
  2026-01:     31,526  MoM  +30.6%
  2026-02:     29,603  MoM   -6.1%   YoY +999.3%
  2026-03:     40,124  MoM  +35.5%   YoY  +57.0%
  2026-04:     32,152  MoM  -19.9%   YoY  +37.4%
  2026-05:     31,262  MoM   -2.8%   YoY  +48.7%
  2026-06:     33,363  MoM   +6.7%   YoY  +44.7%
  2026-07:     34,527  MoM   +3.5%   YoY  +25.3%


full_mean = 0.0
for key in predict_keys[:5000]:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:
        full_mean += z["train_sales"].mean()
print(f"full-training daily mean : {full_mean:,.0f}   (predicted was 126)")

m = best_model.model
print("RIN attr:", [a for a in dir(m) if "norm" in a.lower() or "rin" in a.lower()])
print("hparams :", {k: v for k, v in best_model.model_params.items() if "norm" in k.lower()})

RIN attr: ['layer_norm', 'print', 'rin', 'use_reversible_instance_norm']
hparams : {'norm_type': 'LayerNorm', 'use_reversible_instance_norm': True}