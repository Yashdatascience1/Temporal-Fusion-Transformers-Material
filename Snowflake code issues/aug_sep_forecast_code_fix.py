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

scaled_temporal = future_covariates_scaler.transform(lookahead_data_darts_df)

final_scaled_lookahead_data = transformer.transform(scaled_temporal)

target_scaled_data = target_scaler.transform(lookback_data_darts_df)

final_scaled_lookback_data = transformer.transform(target_scaled_data)

forecast_series = loaded_model.predict(
    n=2, 
    series=final_scaled_lookback_data, 
    future_covariates=final_scaled_lookahead_data
)

print("Forecast generated successfully!")

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[75], line 1
----> 1 forecast_series = loaded_model.predict(
      2     n=2, 
      3     series=final_scaled_lookback_data, 
      4     future_covariates=final_scaled_lookahead_data
      5 )
      7 print("Forecast generated successfully!")

File c:\Users\G0004878\Desktop\Virtual_environments\darts_gpu\lib\site-packages\darts\utils\torch.py:94, in random_method.<locals>.decorator(self, *args, **kwargs)
     92 with fork_rng():
     93     manual_seed(random_instance.randint(0, high=MAX_TORCH_SEED_VALUE))
---> 94     return decorated(self, *args, **kwargs)

File c:\Users\G0004878\Desktop\Virtual_environments\darts_gpu\lib\site-packages\darts\models\forecasting\torch_forecasting_model.py:1698, in TorchForecastingModel.predict(self, n, series, past_covariates, future_covariates, trainer, batch_size, verbose, n_jobs, roll_size, num_samples, dataloader_kwargs, mc_dropout, predict_likelihood_parameters, show_warnings, random_state)
   1678 super().predict(
   1679     n,
   1680     series,
   (...)
   1686     show_warnings=show_warnings,
   1687 )
   1689 dataset = self._build_inference_dataset(
   1690     n=n,
   1691     series=series,
...
File c:\Users\G0004878\Desktop\Virtual_environments\darts_gpu\lib\site-packages\torch\nn\modules\linear.py:116, in Linear.forward(self, input)
    115 def forward(self, input: Tensor) -> Tensor:
--> 116     return F.linear(input, self.weight, self.bias)

RuntimeError: mat1 and mat2 must have the same dtype, but got Double and BFloat16


# same casts as training
final_scaled_lookback_data  = [force_float32_target(ts) for ts in final_scaled_lookback_data]
final_scaled_lookahead_data = [force_float32_cov(ts)    for ts in final_scaled_lookahead_data]

for name, lst in [("lookback", final_scaled_lookback_data),
                  ("lookahead", final_scaled_lookahead_data)]:
    ts = lst[0]
    sc = ts.static_covariates.dtypes.unique().tolist() if ts.has_static_covariates else "none"
    print(f"{name:10s} n={len(lst):6,}  values={ts.dtype}  statics={sc}")

