import numpy as np
v = np.concatenate([ts.values().ravel()
                    for ts in scaled_target_series_with_static_covariates_training[:2000]])
print(f"scaled: mean {v.mean():.3f}  median {np.median(v):.3f}")
print(f"zero months: {(v == 0).mean()*100:.1f}%")
print(f"share of values above delta=0.1: {(v > 0.1).mean()*100:.1f}%")\


import numpy as np

def force_float32_target(ts):
    """Targets: keep static covariates (already numeric via
    StaticCovariatesTransformer), cast both values and statics to float32."""
    ts = ts.astype(np.float32)
    if ts.has_static_covariates:
        ts = ts.with_static_covariates(ts.static_covariates.astype(np.float32))
    return ts

def force_float32_cov(ts):
    """Future covariates: from_group_dataframe attaches the group key as a
    STRING static covariate. Darts reads statics from the target series only,
    so drop them here, then cast values."""
    return ts.with_static_covariates(None).astype(np.float32)


scaled_target_series_with_static_covariates_training = [
    force_float32_target(ts) for ts in scaled_target_series_with_static_covariates_training
]
scaled_target_series_with_static_covariates_validation = [
    force_float32_target(ts) for ts in scaled_target_series_with_static_covariates_validation
]
scaled_future_covariates_training = [
    force_float32_cov(ts) for ts in scaled_future_covariates_training
]
scaled_future_covariates_validation = [
    force_float32_cov(ts) for ts in scaled_future_covariates_validation
]

for name, lst in [("tgt train", scaled_target_series_with_static_covariates_training),
                  ("tgt val",   scaled_target_series_with_static_covariates_validation),
                  ("cov train", scaled_future_covariates_training),
                  ("cov val",   scaled_future_covariates_validation)]:
    ts = lst[0]
    sc = ts.static_covariates.dtypes.unique().tolist() if ts.has_static_covariates else "none"
    print(f"{name:10s} n={len(lst):6,}  values={ts.dtype}  statics={sc}")

