import numpy as np
import pandas as pd

group_col = "PARENT_DEALER_CODE_MODEL_FAMILY"

# ----- 1. Build a modified lookahead df: Jun/Jul 2026 covariates <- Jun/Jul 2025 -----
swapped_df = lookahead_data_pandas_df.copy()
swapped_df.index = pd.to_datetime(swapped_df.index)

swap_pairs = {                                   # target 2026 month : source 2025 month
    pd.Timestamp("2026-06-01"): pd.Timestamp("2025-06-01"),
    pd.Timestamp("2026-07-01"): pd.Timestamp("2025-07-01"),
}

for tgt_date, src_date in swap_pairs.items():
    src = swapped_df[swapped_df.index == src_date].set_index(group_col)[future_covariates]
    tgt_mask = swapped_df.index == tgt_date
    grp_at_tgt = swapped_df.loc[tgt_mask, group_col].values
    new_vals  = src.reindex(grp_at_tgt)[future_covariates].values.astype(float)
    orig_vals = swapped_df.loc[tgt_mask, future_covariates].values.astype(float)
    nan_mask  = np.isnan(new_vals)               # series with no 2025 row -> keep original
    new_vals[nan_mask] = orig_vals[nan_mask]
    swapped_df.loc[tgt_mask, future_covariates] = new_vals

# sanity check: the swapped sums should now match 2025, not 2026
print("SWAPPED Jun+Jul 2026 covariate sums:")
print(swapped_df.loc['2026-06-01':'2026-07-01', future_covariates].sum())
print("\nORIGINAL Jun+Jul 2026 covariate sums (for contrast):")
print(lookahead_data_pandas_df.loc['2026-06-01':'2026-07-01', future_covariates].sum())

# ----- 2. Rebuild darts series + rescale (same pipeline as your original) -----
swapped_lookahead_darts = TimeSeries.from_group_dataframe(
    df=swapped_df,
    group_cols=group_col,
    static_cols=static_covariates,
    value_cols=future_covariates,
    freq='MS'
)
swapped_scaled_temporal = future_covariates_scaler.transform(swapped_lookahead_darts)
swapped_final_lookahead = transformer.transform(swapped_scaled_temporal)

# ----- 3. Predict both runs, lookback held identical -----
orig_forecast    = loaded_model.predict(n=3, series=final_scaled_lookback_data,
                                         future_covariates=final_scaled_lookahead_data)
swapped_forecast = loaded_model.predict(n=3, series=final_scaled_lookback_data,
                                         future_covariates=swapped_final_lookahead)

orig_unscaled    = target_scaler.inverse_transform(orig_forecast)
swapped_unscaled = target_scaler.inverse_transform(swapped_forecast)

# ----- 4. Compare monthly totals across all series -----
def monthly_totals(forecast_list):
    months = forecast_list[0].time_index
    totals = np.zeros(len(months))
    for ts in forecast_list:
        totals += ts.values().flatten()
    return pd.Series(totals, index=months)

print("\nORIGINAL forecast (real 2026 covariates):")
print(monthly_totals(orig_unscaled))
print("\nSWAPPED forecast (2025 festival/marriage values injected into Jun/Jul 2026):")
print(monthly_totals(swapped_unscaled))