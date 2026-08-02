import numpy as np, os, json

new_stats = {}
for key in manifest["series_keys"]:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:
        s = z["train_sales"]
    m = float(s.mean())
    new_stats[key] = [0.0, max(m, 0.05)]      # was: m if m > 1e-3 else 1.0

manifest["scaler_stats"] = new_stats
with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f)
scaler_stats = new_stats
print("scaler_stats recomputed (guard 0.05)")


# ---- drop series that never sold during training ----
DROP_NEVER_SOLD  = True
DORMANT_DAYS     = None      # set to 180/270/365 once the backtest tells you which

keep = []
for k in series_keys:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        s = z["train_sales"]
    if DROP_NEVER_SOLD and s.sum() == 0:
        keep.append(False); continue
    if DORMANT_DAYS and s[-DORMANT_DAYS:].sum() == 0:
        keep.append(False); continue
    keep.append(True)

n_before = len(series_keys)
series_keys    = [k for k, m in zip(series_keys, keep) if m]
has_val        = [h for h, m in zip(has_val, keep) if m]
STATIC_ENCODED = [s for s, m in zip(STATIC_ENCODED, keep) if m]

assert len(series_keys) == len(has_val) == len(STATIC_ENCODED)
print(f"series: {n_before:,} -> {len(series_keys):,}  (dropped {n_before-len(series_keys):,})")


loss_fn=HuberMaeFeatureLoss(delta=100.0, festive_weight=3.0, under_weight=3.0),

