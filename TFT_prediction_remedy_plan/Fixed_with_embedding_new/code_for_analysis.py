import os, glob
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.getcwd(), "series_cache")

# pick one file -- the first one here; or set path to a specific series' file
path = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))[0]
print("File:", os.path.basename(path))

with np.load(path) as z:
    print("Arrays inside:", z.files)
    for name in z.files:
        arr = z[name]
        print(f"  {name:12s} shape={arr.shape} dtype={arr.dtype}"
              + (f" value={arr}" if arr.ndim == 0 else ""))

    # rebuild the two pieces as readable tables with dates
    train = pd.DataFrame({
        "date": pd.date_range(str(z["train_start"]), periods=len(z["train_sales"]), freq="D"),
        "sales": z["train_sales"],
    })
    val = pd.DataFrame({
        "date": pd.date_range(str(z["val_start"]), periods=len(z["val_sales"]), freq="D"),
        "sales": z["val_sales"],
    })

print(f"\nTRAIN piece: {train.date.min().date()} -> {train.date.max().date()} "
      f"({len(train)} days, total sales {train.sales.sum():.0f})")
print(train.head())
print(f"\nVAL piece:   {val.date.min().date()} -> {val.date.max().date()} "
      f"({len(val)} days, total sales {val.sales.sum():.0f})")
print(val.tail())