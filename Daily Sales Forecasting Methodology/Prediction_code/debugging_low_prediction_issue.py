import numpy as np, glob, os
fs = glob.glob(os.path.join(CACHE_DIR, "*.npz"))[:3000]

zero_frac, peak_ratio, means = [], [], []
for f in fs:
    with np.load(f) as z:
        s = z["train_sales"]
    zero_frac.append((s == 0).mean())
    means.append(s.mean())
    if s.mean() > 0:
        peak_ratio.append(s.max() / s.mean())

print(f"median zero-day fraction : {np.median(zero_frac):.1%}")
print(f"90th pct zero fraction   : {np.percentile(zero_frac, 90):.1%}")
print(f"median daily mean        : {np.median(means):.2f}")
print(f"median peak/mean         : {np.median(peak_ratio):.1f}")