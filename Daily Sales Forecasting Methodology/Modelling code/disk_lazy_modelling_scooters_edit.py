def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        # cache the FINISHED TimeSeries — after first touch this is a dict lookup
        if self._ram is not None and idx in self._ram:
            return self._ram[idx]

        key  = self.series_keys[idx]
        path = os.path.join(self.cache_dir, f"{safe_name(key)}.npz")
        with np.load(path, allow_pickle=False) as z:
            sales = z[f"{self.split}_sales"]
            flag  = z[f"{self.split}_flag"]
            start = str(z[f"{self.split}_start"])

        lo, hi = self.scaler_stats[key]
        scaled = ((sales - lo) / (hi - lo)).astype(np.float32)
        values = np.stack([scaled, flag], axis=1)
        times  = pd.date_range(start=start, periods=len(values), freq=self.freq)

        ts = TimeSeries.from_times_and_values(
            times, values,
            columns=[target_col, "FESTIVE_FLAG"],
            static_covariates=self.static_encoded[idx],
        )
        if self._ram is not None:
            self._ram[idx] = ts
        return ts

pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 4,
        "limit_train_batches": 5000,
        "limit_val_batches": 500,
    },

import time, sys
t0 = time.time()
for i in range(len(train_seq)):
    _ = train_seq[i]
    if i % 5000 == 0:
        print(f"  train {i}/{len(train_seq)}  {time.time()-t0:.0f}s", flush=True)
for i in range(len(val_seq)):
    _ = val_seq[i]
print(f"pre-warmed {len(train_seq)+len(val_seq)} series in {time.time()-t0:.0f}s")

import psutil, os
print(f"RSS now: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")

import os, re
# your venv path from the earlier traceback
p = r"c:\Users\G0004878\Desktop\Virtual_environments\darts_gpu\lib\site-packages\darts\models\forecasting\torch_forecasting_model.py"

with open(p, encoding="utf8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "shuffle" in line:
        print(f"{i+1}: {line.rstrip()}")


900:             By default, Darts configures parameters ("batch_size", "shuffle", "drop_last", "collate_fn", "pin_memory")
1137:             By default, Darts configures parameters ("batch_size", "shuffle", "drop_last", "collate_fn", "pin_memory")
1281:                 "shuffle": True,
1295:         dataloader_kwargs["shuffle"] = False
1487:             By default, Darts configures parameters ("batch_size", "shuffle", "drop_last", "collate_fn", "pin_memory")
1617:             By default, Darts configures parameters ("batch_size", "shuffle", "drop_last", "collate_fn", "pin_memory")
1772:             By default, Darts configures parameters ("batch_size", "shuffle", "drop_last", "collate_fn", "pin_memory")
1845:             **{"shuffle": False},