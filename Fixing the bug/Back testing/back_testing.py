full_target_list = [
    ts.slice(pd.Timestamp('2023-04-01'), pd.Timestamp('2025-11-01'))
    for ts in darts_df_with_static_covariates
]

# Transform using ALREADY FITTED scalers — never refit
scaled_full_target = target_scaler.transform(full_target_list)
scaled_full_target = transformer.transform(scaled_full_target)


full_futcov_list = [
    ts.slice(pd.Timestamp('2023-04-01'), pd.Timestamp('2025-11-01'))
    for ts in darts_df_with_future_covariates
]

scaled_full_futcov = future_covariates_scaler.transform(full_futcov_list)