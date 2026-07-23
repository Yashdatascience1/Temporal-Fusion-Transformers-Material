# =============================================================================
# PREDICT THE 2026 WINDOW  (SERIES_INDEX 366..487)
#
# Run in the SAME kernel as the training notebook, so target_series,
# target_scaler, static_transformer, future_covariate_scaler and
# full_future_covariates are all still in scope.
#
# Geometry:
#   historical  = 366 indices  ->  2023: 0-121 | 2024: 122-243 | 2025: 244-365
#   full        = 488 indices  ->  2026: 366-487
#   i = o = 122
#
#   Feeding the full historical series (ends 365) with n=122 makes the model
#   read indices 244-365 (the 2025 window) and emit 366-487 (the 2026 window).
#   Single shot, no autoregression.
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
from darts.models import TFTModel

# =============================================================================
# SECTION 1: LOAD BEST CHECKPOINT
# =============================================================================

CKPT_MODEL_NAME = "daily_festive_tft_boss_approach"   # from TFTModel(model_name=...)
WORK_DIR = os.getcwd()

print("Loading best checkpoint...")
best_model = TFTModel.load_from_checkpoint(
    CKPT_MODEL_NAME, work_dir=WORK_DIR, best=True
)
best_model.model.eval()
print(f"  loaded: {CKPT_MODEL_NAME}")
print(f"  input_chunk_length  : {best_model.input_chunk_length}")
print(f"  output_chunk_length : {best_model.output_chunk_length}")

# =============================================================================
# SECTION 2: PREPARE INPUTS
# =============================================================================
# target_series      -> full historical, indices 0..365 (unscaled, raw statics)
# full_future_covariates -> indices 0..487, includes the 2026 block
#
# Transformers must be REUSED, not refitted:
#   target_scaler          global_fit=False -> per series, same count + order
#   static_transformer     encodes the string statics
#   future_covariate_scaler global_fit=True -> one scaler, any count

print("\nPreparing inputs...")
print(f"  target_series          : {len(target_series)} series, "
      f"len={len(target_series[0])}")
print(f"  full_future_covariates : {len(full_future_covariates)} series, "
      f"len={len(full_future_covariates[0])}")

assert len(target_series) == len(full_future_covariates), \
    "series count mismatch between targets and covariates"

scaled_full_series = target_scaler.transform(target_series)
scaled_full_series = static_transformer.transform(scaled_full_series)

scaled_full_covariates = future_covariate_scaler.transform(full_future_covariates)

# Match the dtype used during training
scaled_full_series     = [s.astype(np.float32) for s in scaled_full_series]
scaled_full_covariates = [c.astype(np.float32) for c in scaled_full_covariates]

print(f"  scaled series len      : {len(scaled_full_series[0])}  "
      f"(index {scaled_full_series[0].start_time()}..{scaled_full_series[0].end_time()})")
print(f"  scaled covariates len  : {len(scaled_full_covariates[0])}  "
      f"(index {scaled_full_covariates[0].start_time()}..{scaled_full_covariates[0].end_time()})")

# =============================================================================
# SECTION 3: PREDICT
# =============================================================================

N_STEPS = 122      # 2026 window length

print(f"\nPredicting {N_STEPS} steps ahead for {len(scaled_full_series)} series...")

with torch.no_grad():
    scaled_2026 = best_model.predict(
        n=N_STEPS,
        series=scaled_full_series,
        future_covariates=scaled_full_covariates,
        verbose=True,
    )

print(f"Forecast index range: {scaled_2026[0].start_time()} → {scaled_2026[0].end_time()}")

# =============================================================================
# SECTION 4: INVERSE SCALE + CLIP
# =============================================================================

print("\nInverse transforming...")
forecasts_2026 = target_scaler.inverse_transform(scaled_2026)
forecasts_2026 = [f.map(lambda x: np.maximum(x, 0)) for f in forecasts_2026]

# =============================================================================
# SECTION 5: BUILD OUTPUT TABLE
# =============================================================================
# Series names come from the ORIGINAL target_series — after
# static_transformer the statics are numeric-encoded and unreadable.

print("Building output table...")

records = []
for forecast, orig in zip(forecasts_2026, target_series):
    name = orig.static_covariates['PARENT_DEALER_CODE_MODEL_FAMILY'].values[0]
    for idx, val in zip(forecast.time_index, forecast.values().flatten()):
        records.append({
            'SERIES_INDEX': int(idx),
            'PARENT_DEALER_CODE_MODEL_FAMILY': name,
            'PREDICTED_SALES': round(float(val), 2),
        })

df_2026 = pd.DataFrame(records)

# Attach real dates: SERIES_INDEX -> CAL_DATE, taken from the source data
idx_to_date = (
    data[['SERIES_INDEX', 'CAL_DATE']]
    .drop_duplicates(subset='SERIES_INDEX')
    .set_index('SERIES_INDEX')['CAL_DATE']
)
df_2026['CAL_DATE'] = df_2026['SERIES_INDEX'].map(idx_to_date)

df_2026 = df_2026[
    ['PARENT_DEALER_CODE_MODEL_FAMILY', 'SERIES_INDEX', 'CAL_DATE', 'PREDICTED_SALES']
].sort_values(['PARENT_DEALER_CODE_MODEL_FAMILY', 'SERIES_INDEX']).reset_index(drop=True)

print(f"\nRows   : {len(df_2026):,}")
print(f"Series : {df_2026['PARENT_DEALER_CODE_MODEL_FAMILY'].nunique():,}")
print(f"Index  : {df_2026['SERIES_INDEX'].min()} → {df_2026['SERIES_INDEX'].max()}")
print(f"Dates  : {df_2026['CAL_DATE'].min()} → {df_2026['CAL_DATE'].max()}")
print(f"Total  : {df_2026['PREDICTED_SALES'].sum()/1e5:,.2f} lacs")

df_2026.to_csv("2026_daily_forecast_output.csv", index=False)
print("Saved: 2026_daily_forecast_output.csv")

# =============================================================================
# SECTION 6: SEASONALITY CHECK
# =============================================================================
# The question that matters: are the festive peaks there, and are they the
# right size? Compare the 2026 forecast against the 2025 actuals it was
# conditioned on — same 122-day window, one year apart.

print("\n" + "="*60)
print("SEASONALITY CHECK")
print("="*60)

# Daily totals across all series
daily_pred = df_2026.groupby('SERIES_INDEX')['PREDICTED_SALES'].sum()
daily_pred.index = daily_pred.index - 366           # 0..121 within the window

actual_2025 = []
for s in target_series:
    v = s.values().flatten()
    actual_2025.append(v[244:366])                  # 2025 window
actual_2025 = np.array(actual_2025).sum(axis=0)     # daily totals

print(f"2025 actual  : peak {actual_2025.max():,.0f} | mean {actual_2025.mean():,.0f} "
      f"| peak/mean {actual_2025.max()/actual_2025.mean():.2f}")
print(f"2026 forecast: peak {daily_pred.max():,.0f} | mean {daily_pred.mean():,.0f} "
      f"| peak/mean {daily_pred.max()/daily_pred.mean():.2f}")
print(f"\nTotal 2025 actual   : {actual_2025.sum()/1e5:,.2f} lacs")
print(f"Total 2026 forecast : {daily_pred.sum()/1e5:,.2f} lacs")
print(f"YoY                 : {(daily_pred.sum()/actual_2025.sum()-1)*100:+.1f}%")

print(f"\nPeak day within window — actual 2025: day {actual_2025.argmax()} "
      f"| forecast 2026: day {daily_pred.values.argmax()}")

# Per-series flatness
per_series = df_2026.groupby('PARENT_DEALER_CODE_MODEL_FAMILY')['PREDICTED_SALES'].agg(
    ['mean', 'max', 'std'])
flat = (per_series['std'] < 1e-6).mean() * 100
ratio = (per_series['max'] / per_series['mean'].replace(0, np.nan)).median()
print(f"\nPerfectly flat series : {flat:.1f}%")
print(f"Median peak/mean      : {ratio:.2f}")
