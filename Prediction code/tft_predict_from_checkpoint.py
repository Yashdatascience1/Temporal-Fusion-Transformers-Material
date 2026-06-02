"""
TFT prediction from a saved Darts checkpoint — clean version.

Replaces the previous fit(epochs=1) + _TFTModule.load_from_checkpoint hack with
the proper Darts load, which restores both the neural weights AND the fitted
encoder transformers.

WARNING on this checkpoint:
    The "best" checkpoint filename is `tft-best-...-train_loss=0.0184.ckpt`
    i.e. selected by TRAINING loss, not validation loss. That's typically the
    most-overfit epoch. After this run, change the ModelCheckpoint callback to
    monitor="val_loss" before treating "best" as reliable.

Fill in the four PATHS below and the LOOKBACK / LOOKAHEAD dates for your run.
"""

import pandas as pd
import joblib
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler

# ---------------------------------------------------------------------------
# Encoder functions — must be in scope BEFORE load_from_checkpoint, because
# the saved model references them by name when it unpickles add_encoders.
# Keep these identical to what you used during training.
# ---------------------------------------------------------------------------
def encode_year(idx):
    return (idx.year - 2000) / 50

def encode_days_in_month(index):
    return index.days_in_month.to_numpy().reshape(-1, 1)

# ---------------------------------------------------------------------------
# 1. PATHS — set these for your environment
# ---------------------------------------------------------------------------
WORK_DIR   = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Re-training the model"
MODEL_NAME = "tft_net_sales_2026-06-02_11_52_56"

SCALER_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model\scaled_objects_pickled_version"
# If you re-fit the scalers during retraining, point these at the new pickles.

target_scaler            = joblib.load(rf"{SCALER_DIR}\target_scaler.pkl")
future_covariates_scaler = joblib.load(rf"{SCALER_DIR}\future_covariates_scaler.pkl")
static_transformer       = joblib.load(rf"{SCALER_DIR}\static_transformer.pkl")

# ---------------------------------------------------------------------------
# 2. LOAD THE MODEL — the proper way (no fit/swap hack)
# ---------------------------------------------------------------------------
loaded_model = TFTModel.load_from_checkpoint(
    model_name = MODEL_NAME,
    work_dir   = WORK_DIR,
    best       = True,   # picks the file with "best" in its name
)
print(f"Loaded model: {MODEL_NAME}")
print(f"input_chunk_length  = {loaded_model.input_chunk_length}")
print(f"output_chunk_length = {loaded_model.output_chunk_length}")

# ---------------------------------------------------------------------------
# 3. PREPARE PREDICTION DATA
#    Assumes `pandas_df` is already loaded in your session and indexed by date.
# ---------------------------------------------------------------------------
# Date windows — adjust for which months you want to forecast.
# Forecast horizon = LOOKAHEAD_END - LOOKBACK_END months.
LOOKBACK_START = '2025-02-01'
LOOKBACK_END   = '2026-05-01'   # last fully-closed month going into the model
LOOKAHEAD_END  = '2026-08-01'   # last forecast month (inclusive)
N_HORIZON      = 3              # must match (LOOKAHEAD_END - LOOKBACK_END) in months

# Column groups (matching your original pipeline)
target_col = ["NET_SALES"]

num_cols        = pandas_df.select_dtypes(include=['number']).columns.tolist()
static_cols_all = pandas_df.select_dtypes(exclude=['number', 'datetime', 'datetime64']).columns.tolist()

# Future covariates = everything numeric except the target itself
future_covariates = [c for c in num_cols if c != 'NET_SALES']
# Static covariates seen by the model = all string ID cols except the group key
static_covariates = [c for c in static_cols_all if c != 'PARENT_DEALER_CODE_MODEL_FAMILY']

target_plus_static_cols = target_col      + static_cols_all
static_plus_future_cov  = static_cols_all + future_covariates

# Slice the dataframe
prediction_df = pandas_df.loc[LOOKBACK_START:LOOKAHEAD_END]

lookback_data_pandas_df  = prediction_df.loc[LOOKBACK_START:LOOKBACK_END,  target_plus_static_cols]
lookahead_data_pandas_df = prediction_df.loc[LOOKBACK_START:LOOKAHEAD_END, static_plus_future_cov]

# ---------------------------------------------------------------------------
# 4. BUILD DARTS TIMESERIES
# ---------------------------------------------------------------------------
lookback_data_darts = TimeSeries.from_group_dataframe(
    df          = lookback_data_pandas_df,
    group_cols  = "PARENT_DEALER_CODE_MODEL_FAMILY",
    static_cols = static_covariates,
    value_cols  = ["NET_SALES"],
    freq        = 'MS',
)

lookahead_data_darts = TimeSeries.from_group_dataframe(
    df          = lookahead_data_pandas_df,
    group_cols  = "PARENT_DEALER_CODE_MODEL_FAMILY",
    static_cols = static_covariates,
    value_cols  = future_covariates,
    freq        = 'MS',
)

# ---------------------------------------------------------------------------
# 5. SCALE  (same order as training: value scaler first, then static transformer)
# ---------------------------------------------------------------------------
target_scaled    = target_scaler.transform(lookback_data_darts)
lookback_scaled  = static_transformer.transform(target_scaled)

future_scaled    = future_covariates_scaler.transform(lookahead_data_darts)
lookahead_scaled = static_transformer.transform(future_scaled)

# ---------------------------------------------------------------------------
# 6. PREDICT
# ---------------------------------------------------------------------------
forecast_scaled = loaded_model.predict(
    n                 = N_HORIZON,
    series            = lookback_scaled,
    future_covariates = lookahead_scaled,
)

# ---------------------------------------------------------------------------
# 7. INVERSE-TRANSFORM BACK TO ORIGINAL SCALE
# ---------------------------------------------------------------------------
forecast = target_scaler.inverse_transform(forecast_scaled)

# ---------------------------------------------------------------------------
# 8. BUILD OUTPUT DATAFRAME
# ---------------------------------------------------------------------------
records = []
for fc in forecast:
    series_name = fc.static_covariates['PARENT_DEALER_CODE_MODEL_FAMILY'].values[0]
    months      = fc.time_index
    values      = fc.values().flatten()
    for month, pred in zip(months, values):
        records.append({
            'MONTH_OF_SALE'                  : month,
            'PARENT_DEALER_CODE_MODEL_FAMILY': series_name,
            'PREDICTED_SALES'                : round(float(pred), 2),
        })

df_final_output = pd.DataFrame(records)
df_final_output['MONTH_OF_SALE'] = pd.to_datetime(df_final_output['MONTH_OF_SALE']).dt.strftime('%Y-%m-%d')
df_final_output = df_final_output.sort_values(
    ['PARENT_DEALER_CODE_MODEL_FAMILY', 'MONTH_OF_SALE']
).reset_index(drop=True)

print(f"\nOutput shape : {df_final_output.shape}")
print(f"Months       : {sorted(df_final_output['MONTH_OF_SALE'].unique())}")
print(f"Series count : {df_final_output['PARENT_DEALER_CODE_MODEL_FAMILY'].nunique()}")
print(df_final_output.head(10))

# Optional: save to disk
# df_final_output.to_csv('forecast_jun_jul_aug_2026.csv', index=False)
