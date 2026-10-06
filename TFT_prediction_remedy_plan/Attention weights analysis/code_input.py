tm = ctx.model.model
print("static vars     :", len(tm.static_variables), tm.static_variables)
print("categorical     :", getattr(tm, "categorical_static_variables", None))
print("numeric         :", getattr(tm, "numeric_static_variables", None))

labels = ctx.labels()
idx = stratified_indices(labels, n_per_group=100)
out = extract_tft_interpretation(
    ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
    indices=idx, group_labels=labels, target_names=[ctx.target_col],
    future_names=ctx.future_covariates, static_names=ctx.static_covariates,
)
print(out["ALL"]["n_series"])

import pandas as pd

sc = ctx.shared_cov
cov = pd.DataFrame(sc.values(), index=sc.time_index, columns=sc.components).loc["2026-09-01":"2026-12-07"]
assert len(cov) == 98, len(cov)          # must equal the 98 decoder steps

dt = out["ALL"]["decoder_time"]
for v in ["D-2", "N+3", "D", "IS_MONTH_START"]:
    a = (cov[v] != 0).to_numpy()
    print(f"{v:15s} active: {dt[v].to_numpy()[a].mean():.3f}   zero: {dt[v].to_numpy()[~a].mean():.3f}")