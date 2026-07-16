# ============================================================
# TFT QUICK TEST:
# Train      : 2023-04-01 to 2025-12-31
# Validate   : 2026-01-01 to 2026-06-30
# Forecast   : 2026-08-01 to 2026-09-30
#
# The feature-aware penalty is implemented with Darts
# sample_weight. Darts multiplies the element-wise Huber loss
# by the timestamp weight.
# ============================================================

import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping

from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session

# Existing project imports
import sys
sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import snowflake_utils
import Snowflake_configuration


# ============================================================
# 1. CONFIGURATION
# ============================================================

TARGET_COL = "NET_SALES"
GROUP_COL = "PARENT_DEALER_CODE_MODEL_FAMILY"
TIME_COL = "CAL_DATE"

TRAIN_START = pd.Timestamp("2023-04-01")
TRAIN_END = pd.Timestamp("2025-12-31")

VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-06-30")

# Target observations are assumed available through this date.
OBSERVED_END_FOR_FORECAST = pd.Timestamp("2026-06-30")

FORECAST_START = pd.Timestamp("2026-08-01")
FORECAST_END = pd.Timestamp("2026-09-30")

# Keeping the original encoder/decoder lengths for a controlled test.
INPUT_CHUNK_LENGTH = 122
OUTPUT_CHUNK_LENGTH = 122

# Increase the Huber contribution on a date if ANY listed feature is non-zero.
NORMAL_DAY_WEIGHT = 1.0
FESTIVE_DAY_WEIGHT = 5.0
HUBER_DELTA = 1.0

STATIC_COVARIATES = [
    "PARENT_DEALER_CODE",
    "MODEL_FAMILY",
    "MODEL_FAMILY_CODE",
    "MODEL_NAME",
    "BRAKE_TYPE",
    "IGNITION_TYPE",
    "WHEEL_TYPE",
    "COLOUR",
    "DEALER_CITY",
    "X_CITY_CATEGORY",
    "ZONAL_OFFICE_NAME",
]

CALENDAR_COVARIATES = [
    "YEAR",
    "MONTH",
    "DAY_OF_THE_WEEK",
    "DAY_OF_THE_MONTH",
]

BINARY_FEATURES = [
    "N-16", "N-15", "N-14", "N-13",
    "N-12", "N-11", "N-10", "N-9",
    "N-8", "N-7", "N-6", "N-5",
    "N-4", "N-3", "N-2", "N-1",
    "N", "N+1", "N+2", "N+3",
    "N+4", "N+5", "N+6", "N+7",
    "N+8", "N+9", "N+10",
    "D-3", "D-2", "D-1", "D",
    "D+1", "D+2", "D+3", "D+4",
    "D+5", "D+6",
    "C", "C+1", "C+2", "C+3",
    "C+4", "C+5", "C+6",
]

FUTURE_COVARIATES = CALENDAR_COVARIATES + BINARY_FEATURES


# ============================================================
# 2. SNOWFLAKE READ
# ============================================================

session = (
    Session.builder
    .configs(Snowflake_configuration.ds1_role_json)
    .create()
)

session.use_database("MOP_DATABASE")
session.use_schema("SOQ")

snowpark_df = session.sql(
    """
    SELECT *
    FROM MOP_DATABASE.SOQ.DAILY_FORECASTING_DATA_FOR_MODELLING_TFT
    WHERE PARENT_DEALER_CODE_MODEL_FAMILY IN (
        SELECT PARENT_DEALER_CODE_MODEL_FAMILY
        FROM MOP_DATABASE.SOQ.VALID_SERIES_FOR_DAILY_MODELLING
    )
    """
)

if "DATASET_TYPE" in snowpark_df.columns:
    snowpark_df = snowpark_df.drop("DATASET_TYPE")

data = snowpark_df.to_pandas()


# ============================================================
# 3. DATA PREPARATION AND VALIDATION
# ============================================================

required_columns = (
    [TIME_COL, GROUP_COL, TARGET_COL]
    + STATIC_COVARIATES
    + BINARY_FEATURES
)

missing_columns = sorted(set(required_columns) - set(data.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

data[TIME_COL] = pd.to_datetime(data[TIME_COL], errors="raise").dt.normalize()
data[GROUP_COL] = data[GROUP_COL].astype(str)

# Recreate calendar columns from CAL_DATE so they are consistent.
data["YEAR"] = data[TIME_COL].dt.year.astype(np.float32)
data["MONTH"] = data[TIME_COL].dt.month.astype(np.float32)
data["DAY_OF_THE_WEEK"] = data[TIME_COL].dt.dayofweek.astype(np.float32)
data["DAY_OF_THE_MONTH"] = data[TIME_COL].dt.day.astype(np.float32)

# Treat the requested features as binary flags.
data[BINARY_FEATURES] = (
    data[BINARY_FEATURES]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .astype(np.float32)
)

# A non-zero value in ANY binary feature makes the date important.
data["FESTIVE_FLAG"] = (
    data[BINARY_FEATURES]
    .ne(0)
    .any(axis=1)
    .astype(np.float32)
)

data["LOSS_WEIGHT"] = np.where(
    data["FESTIVE_FLAG"].eq(1.0),
    FESTIVE_DAY_WEIGHT,
    NORMAL_DAY_WEIGHT,
).astype(np.float32)

# Keep only the complete modelling/prediction date range.
data = data[
    data[TIME_COL].between(TRAIN_START, FORECAST_END)
].copy()

data = data.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)

duplicate_mask = data.duplicated([GROUP_COL, TIME_COL], keep=False)
if duplicate_mask.any():
    duplicate_sample = data.loc[
        duplicate_mask,
        [GROUP_COL, TIME_COL]
    ].head(20)

    raise ValueError(
        "Duplicate group/date rows found. Aggregate NET_SALES before creating "
        f"Darts series.\nSample:\n{duplicate_sample}"
    )

# Target values are required only through June 2026.
observed_target_mask = data[TIME_COL].between(TRAIN_START, OBSERVED_END_FOR_FORECAST)

if data.loc[observed_target_mask, TARGET_COL].isna().any():
    raise ValueError(
        "NET_SALES contains missing values between "
        f"{TRAIN_START.date()} and {OBSERVED_END_FOR_FORECAST.date()}."
    )

if data[FUTURE_COVARIATES].isna().any().any():
    null_counts = data[FUTURE_COVARIATES].isna().sum()
    raise ValueError(
        "Missing future-covariate values:\n"
        f"{null_counts[null_counts.gt(0)].sort_values(ascending=False)}"
    )


# ============================================================
# 4. EXACT DATE SPLITS
# ============================================================

# Validation series must contain encoder history before 2026-01-01.
# 2026-01-01 minus 122 days = 2025-09-01.
VALIDATION_CONTEXT_START = (
    VALIDATION_START - pd.Timedelta(days=INPUT_CHUNK_LENGTH)
)

train_df = data[
    data[TIME_COL].between(TRAIN_START, TRAIN_END)
].copy()

validation_with_context_df = data[
    data[TIME_COL].between(VALIDATION_CONTEXT_START, VALIDATION_END)
].copy()

# Used as the target context when producing the August-September forecast.
observed_through_june_df = data[
    data[TIME_COL].between(TRAIN_START, OBSERVED_END_FOR_FORECAST)
].copy()

# Future-known covariates are available through the end of September.
full_covariate_df = data[
    data[TIME_COL].between(TRAIN_START, FORECAST_END)
].copy()

train_covariate_df = train_df[
    [TIME_COL, GROUP_COL] + FUTURE_COVARIATES
].copy()

validation_covariate_df = validation_with_context_df[
    [TIME_COL, GROUP_COL] + FUTURE_COVARIATES
].copy()

all_covariate_df = full_covariate_df[
    [TIME_COL, GROUP_COL] + FUTURE_COVARIATES
].copy()


# ============================================================
# 5. DARTS SERIES HELPERS
# ============================================================

def _series_key(ts: TimeSeries) -> str:
    """Extract the group id automatically attached as a static covariate."""
    return str(ts.static_covariates[GROUP_COL].iloc[0])


def make_series_map(
    frame: pd.DataFrame,
    value_cols,
    static_cols=None,
) -> dict[str, TimeSeries]:
    """Create a stable group-id -> TimeSeries mapping."""
    ts_list = TimeSeries.from_group_dataframe(
        df=frame,
        group_cols=GROUP_COL,
        time_col=TIME_COL,
        value_cols=value_cols,
        static_cols=static_cols,
        freq="D",
        fill_missing_dates=False,
        n_jobs=-1,
        verbose=True,
    )

    output = {}
    for ts in ts_list:
        key = _series_key(ts)
        if key in output:
            raise ValueError(f"Duplicate TimeSeries key created: {key}")
        output[key] = ts

    return output


train_target_map = make_series_map(
    train_df[[TIME_COL, GROUP_COL, TARGET_COL] + STATIC_COVARIATES],
    value_cols=TARGET_COL,
    static_cols=STATIC_COVARIATES,
)

validation_target_map = make_series_map(
    validation_with_context_df[
        [TIME_COL, GROUP_COL, TARGET_COL] + STATIC_COVARIATES
    ],
    value_cols=TARGET_COL,
    static_cols=STATIC_COVARIATES,
)

observed_target_map = make_series_map(
    observed_through_june_df[
        [TIME_COL, GROUP_COL, TARGET_COL] + STATIC_COVARIATES
    ],
    value_cols=TARGET_COL,
    static_cols=STATIC_COVARIATES,
)

train_covariate_map = make_series_map(
    train_covariate_df,
    value_cols=FUTURE_COVARIATES,
)

validation_covariate_map = make_series_map(
    validation_covariate_df,
    value_cols=FUTURE_COVARIATES,
)

all_covariate_map = make_series_map(
    all_covariate_df,
    value_cols=FUTURE_COVARIATES,
)

train_weight_map = make_series_map(
    train_df[[TIME_COL, GROUP_COL, "LOSS_WEIGHT"]],
    value_cols="LOSS_WEIGHT",
)

validation_weight_map = make_series_map(
    validation_with_context_df[[TIME_COL, GROUP_COL, "LOSS_WEIGHT"]],
    value_cols="LOSS_WEIGHT",
)

# Keep only groups available in every required object and force identical order.
common_series_ids = sorted(
    set(train_target_map)
    & set(validation_target_map)
    & set(observed_target_map)
    & set(train_covariate_map)
    & set(validation_covariate_map)
    & set(all_covariate_map)
    & set(train_weight_map)
    & set(validation_weight_map)
)

if not common_series_ids:
    raise ValueError("No common series found across train/validation/forecast datasets.")

print("Number of common series:", len(common_series_ids))

train_series = [train_target_map[key] for key in common_series_ids]
validation_series = [validation_target_map[key] for key in common_series_ids]
observed_through_june_series = [
    observed_target_map[key] for key in common_series_ids
]

train_covariates = [train_covariate_map[key] for key in common_series_ids]
validation_covariates = [
    validation_covariate_map[key] for key in common_series_ids
]
all_covariates = [all_covariate_map[key] for key in common_series_ids]

train_weights = [train_weight_map[key] for key in common_series_ids]
validation_weights = [
    validation_weight_map[key] for key in common_series_ids
]

minimum_train_length = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH
too_short_train = [
    common_series_ids[i]
    for i, ts in enumerate(train_series)
    if len(ts) < minimum_train_length
]

too_short_validation = [
    common_series_ids[i]
    for i, ts in enumerate(validation_series)
    if len(ts) < minimum_train_length
]

if too_short_train or too_short_validation:
    raise ValueError(
        f"Series shorter than input+output={minimum_train_length}. "
        f"Train examples: {too_short_train[:5]}; "
        f"Validation examples: {too_short_validation[:5]}"
    )


# ============================================================
# 6. SCALING
# ============================================================

target_scaler = Scaler(global_fit=False, n_jobs=-1)
future_covariate_scaler = Scaler(global_fit=True, n_jobs=-1)
static_transformer = StaticCovariatesTransformer(n_jobs=-1)

scaled_train_series = target_scaler.fit_transform(train_series)
scaled_train_series = static_transformer.fit_transform(scaled_train_series)

scaled_validation_series = target_scaler.transform(validation_series)
scaled_validation_series = static_transformer.transform(
    scaled_validation_series
)

scaled_observed_through_june_series = target_scaler.transform(
    observed_through_june_series
)
scaled_observed_through_june_series = static_transformer.transform(
    scaled_observed_through_june_series
)

scaled_train_covariates = future_covariate_scaler.fit_transform(
    train_covariates
)
scaled_validation_covariates = future_covariate_scaler.transform(
    validation_covariates
)
scaled_all_covariates = future_covariate_scaler.transform(
    all_covariates
)

# Keep all tensors on the same dtype.
scaled_train_series = [ts.astype(np.float32) for ts in scaled_train_series]
scaled_validation_series = [
    ts.astype(np.float32) for ts in scaled_validation_series
]
scaled_observed_through_june_series = [
    ts.astype(np.float32)
    for ts in scaled_observed_through_june_series
]

scaled_train_covariates = [
    ts.astype(np.float32) for ts in scaled_train_covariates
]
scaled_validation_covariates = [
    ts.astype(np.float32) for ts in scaled_validation_covariates
]
scaled_all_covariates = [
    ts.astype(np.float32) for ts in scaled_all_covariates
]

# Do NOT scale loss weights.
train_weights = [ts.astype(np.float32) for ts in train_weights]
validation_weights = [
    ts.astype(np.float32) for ts in validation_weights
]


# ============================================================
# 7. FEATURE-AWARE WEIGHTED LOSS
# ============================================================

# Important:
# Darts loss_fn receives predictions and targets, not future covariates.
# The feature-aware part must therefore be passed through sample_weight.
# Darts temporarily changes HuberLoss.reduction to "none", computes the
# element-wise loss, multiplies by LOSS_WEIGHT, and then averages it.

if "sample_weight" not in inspect.signature(TFTModel.fit).parameters:
    raise RuntimeError(
        "This installed Darts version does not expose sample_weight in "
        "TFTModel.fit(). Upgrade Darts before running this test."
    )

base_loss = torch.nn.HuberLoss(
    delta=HUBER_DELTA,
    reduction="mean",
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="min",
)

trainer_kwargs = {
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "devices": 1,
    "callbacks": [early_stopping],
    "gradient_clip_val": 0.1,
}

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    torch.set_float32_matmul_precision("high")
    trainer_kwargs["precision"] = "bf16-mixed"

model_name = (
    "daily_tft_festive_weighted_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=4,
    num_attention_heads=16,
    dropout=0.05,

    batch_size=256,
    n_epochs=100,

    likelihood=None,
    loss_fn=base_loss,

    random_state=42,
    add_relative_index=True,
    skip_interpolation=True,

    save_checkpoints=True,
    force_reset=True,
    model_name=model_name,

    pl_trainer_kwargs=trainer_kwargs,
)

model.fit(
    series=scaled_train_series,
    future_covariates=scaled_train_covariates,

    val_series=scaled_validation_series,
    val_future_covariates=scaled_validation_covariates,

    # This is the custom festive penalty.
    sample_weight=train_weights,
    val_sample_weight=validation_weights,

    load_best=True,

    dataloader_kwargs={
        "num_workers": 4,
        "pin_memory": torch.cuda.is_available(),
    },
    verbose=True,
)


# ============================================================
# 8. VALIDATION: JANUARY-JUNE 2026
# ============================================================

validation_horizon = (
    VALIDATION_END - TRAIN_END
).days

print("Validation horizon:", validation_horizon)  # 181 days

scaled_validation_forecasts = model.predict(
    n=validation_horizon,
    series=scaled_train_series,
    # A longer covariate series is allowed; Darts aligns by CAL_DATE.
    future_covariates=scaled_all_covariates,
)

validation_forecasts = target_scaler.inverse_transform(
    scaled_validation_forecasts
)

validation_forecasts = [
    forecast.map(lambda value: np.maximum(value, 0.0))
    for forecast in validation_forecasts
]


def forecasts_to_dataframe(
    forecasts: list[TimeSeries],
    series_ids: list[str],
) -> pd.DataFrame:
    records = []

    for series_id, forecast in zip(series_ids, forecasts):
        values = forecast.values(copy=False).reshape(-1)

        for forecast_date, prediction in zip(
            forecast.time_index,
            values,
        ):
            records.append(
                {
                    GROUP_COL: series_id,
                    TIME_COL: pd.Timestamp(forecast_date),
                    "PREDICTED_SALES": round(float(prediction), 4),
                }
            )

    return pd.DataFrame(records)


validation_output = forecasts_to_dataframe(
    validation_forecasts,
    common_series_ids,
)

validation_actual = data[
    data[TIME_COL].between(VALIDATION_START, VALIDATION_END)
][[GROUP_COL, TIME_COL, TARGET_COL]].copy()

validation_output = validation_output.merge(
    validation_actual,
    on=[GROUP_COL, TIME_COL],
    how="left",
)

validation_output = validation_output.rename(
    columns={TARGET_COL: "ACTUAL_SALES"}
)

validation_output.to_csv(
    "jan_to_jun_2026_validation_weighted_loss.csv",
    index=False,
)

print(validation_output.head())
print(validation_output[TIME_COL].min(), validation_output[TIME_COL].max())


# ============================================================
# 9. FORECAST AUGUST-SEPTEMBER 2026
# ============================================================

# The target context ends on 2026-06-30, so Darts must first forecast July.
# Forecast 2026-07-01 through 2026-09-30 (92 days), then retain Aug-Sep.
forecast_horizon_from_june = (
    FORECAST_END - OBSERVED_END_FOR_FORECAST
).days

print("Forecast horizon from June 30:", forecast_horizon_from_june)  # 92

scaled_july_to_september_forecasts = model.predict(
    n=forecast_horizon_from_june,
    series=scaled_observed_through_june_series,
    future_covariates=scaled_all_covariates,
)

july_to_september_forecasts = target_scaler.inverse_transform(
    scaled_july_to_september_forecasts
)

july_to_september_forecasts = [
    forecast.map(lambda value: np.maximum(value, 0.0))
    for forecast in july_to_september_forecasts
]

august_september_output = forecasts_to_dataframe(
    july_to_september_forecasts,
    common_series_ids,
)

august_september_output = august_september_output[
    august_september_output[TIME_COL].between(
        FORECAST_START,
        FORECAST_END,
    )
].reset_index(drop=True)

august_september_output.to_csv(
    "aug_sep_2026_forecast_weighted_loss.csv",
    index=False,
)

print(august_september_output.head())
print(
    august_september_output[TIME_COL].min(),
    august_september_output[TIME_COL].max(),
)
