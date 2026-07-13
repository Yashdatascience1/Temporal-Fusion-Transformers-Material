target_col = "NET_SALES"
group_col = "PARENT_DEALER_CODE_MODEL_FAMILY"
time_col = "SERIES_INDEX"


#Static covariates
static_covariates = [
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


#Known future covariates
calendar_covariates = [
    "YEAR",
    "MONTH",
    "DAY_OF_THE_WEEK",
    "DAY_OF_THE_MONTH",
]

festival_covariates = [
    "HARTALIK_TEEJ",
    "GANESH_CHATURTHI",
    "JANMASHTAMI",
    "VISHWAKARMA_PUJA",
    "KARWA_CHAUTH",
    "ONAM",
    "MARRIAGE_DAY",

    "N-16", "N-15", "N-14", "N-13",
    "N-12", "N-11", "N-10", "N-9",
    "N-8", "N-7", "N-6", "N-5",
    "N-4", "N-3", "N-2", "N-1",
    "N",
    "N+1", "N+2", "N+3", "N+4",
    "N+5", "N+6", "N+7", "N+8", "N+9",

    "D-3", "D-2", "D-1", "D",
    "D+1", "D+2", "D+3",
    "D+4", "D+5", "D+6",
]

future_covariates = calendar_covariates + festival_covariates

import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    Scaler,
    StaticCovariatesTransformer,
)
from darts.models import TFTModel


df = pd.read_csv(
    "daily_festive_model_data.csv",
    parse_dates=["CAL_DATE"],
)

df = df.sort_values(
    [
        "PARENT_DEALER_CODE_MODEL_FAMILY",
        "SERIES_INDEX",
    ]
).reset_index(drop=True)

df.columns = (
    df.columns
      .str.strip()
      .str.replace('"', "", regex=False)
)

required_columns = (
    [group_col, time_col, target_col, "CAL_DATE", "YEAR"]
    + static_covariates
    + future_covariates
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Columns missing from dataframe: {missing_columns}"
    )

df[festival_covariates] = (
    df[festival_covariates]
    .fillna(0)
    .astype("float32")
)

weekday_mapping = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}

if df["DAY_OF_THE_WEEK"].dtype == "object":
    df["DAY_OF_THE_WEEK"] = (
        df["DAY_OF_THE_WEEK"]
        .str.upper()
        .map(weekday_mapping)
    )

if df[future_covariates].isna().any().any():
    null_counts = (
        df[future_covariates]
        .isna()
        .sum()
    )

    print(
        null_counts[null_counts > 0]
        .sort_values(ascending=False)
    )

    raise ValueError(
        "Null values found in known future covariates."
    )

static_nunique = (
    df.groupby(group_col)[static_covariates]
      .nunique(dropna=False)
)

changing_static_columns = (
    static_nunique > 1
).stack()

changing_static_columns = changing_static_columns[
    changing_static_columns
]

if not changing_static_columns.empty:
    print(
        "These supposed static covariates change "
        "within a series:"
    )
    print(changing_static_columns.head(30))

    raise ValueError(
        "Static covariates are not constant within series."
    )

historical_df = df[
    df["YEAR"].isin([2023, 2024, 2025])
].copy()

future_2026_df = df[
    df["YEAR"].eq(2026)
].copy()


if historical_df[target_col].isna().any():
    raise ValueError(
        "Historical NET_SALES contains missing values."
    )

if future_2026_df[target_col].notna().any():
    print(
        "Warning: Some 2026 NET_SALES values are populated. "
        "Ensure these are not placeholder zeroes."
    )


target_input_df = historical_df[
    [time_col, group_col, target_col]
    + static_covariates
].copy()


target_series = TimeSeries.from_group_dataframe(
    df=target_input_df,
    group_cols=group_col,
    time_col=time_col,
    value_cols=target_col,
    static_cols=static_covariates,
    freq=1,
    fill_missing_dates=False,
)

print("Number of series:", len(target_series))
print("Length of first series:", len(target_series[0]))
print("First series index:", target_series[0].time_index[:5])
print("Last series index:", target_series[0].time_index[-5:])
print("Static covariates:")
print(target_series[0].static_covariates)

training_covariate_df = historical_df[
    [time_col, group_col]
    + future_covariates
].copy()

all_covariate_df = df[
    [time_col, group_col]
    + future_covariates
].copy()

training_future_covariates = (
    TimeSeries.from_group_dataframe(
        df=training_covariate_df,
        group_cols=group_col,
        time_col=time_col,
        value_cols=future_covariates,
        freq=1,
        fill_missing_dates=False,
    )
)

full_future_covariates = (
    TimeSeries.from_group_dataframe(
        df=all_covariate_df,
        group_cols=group_col,
        time_col=time_col,
        value_cols=future_covariates,
        freq=1,
        fill_missing_dates=False,
    )
)

assert len(target_series) == len(training_future_covariates)
assert len(target_series) == len(full_future_covariates)

print(len(training_future_covariates[0]))  # 366
print(len(full_future_covariates[0]))      # 488

validation_start_index = 245

train_series = []
validation_series = []

for series in target_series:
    train_part = series.slice(
        series.start_time(),
        validation_start_index - 1,
    )

    validation_part = series.slice(
        validation_start_index,
        series.end_time(),
    )

    train_series.append(train_part)
    validation_series.append(validation_part)


train_covariates = []
validation_covariates = []

for covariate_series in training_future_covariates:
    train_covariates.append(
        covariate_series.slice(
            covariate_series.start_time(),
            validation_start_index - 1,
        )
    )

    validation_covariates.append(
        covariate_series
    )

target_scaler = Scaler(
    global_fit=False,
    n_jobs=-1,
)

future_covariate_scaler = Scaler(
    global_fit=True,
    n_jobs=-1,
)

static_transformer = StaticCovariatesTransformer(
    n_jobs=-1,
)


scaled_train_series = (
    target_scaler.fit_transform(train_series)
)

scaled_train_series = (
    static_transformer.fit_transform(
        scaled_train_series
    )
)

scaled_validation_series = (
    target_scaler.transform(validation_series)
)

scaled_validation_series = (
    static_transformer.transform(
        scaled_validation_series
    )
)

scaled_train_covariates = (
    future_covariate_scaler.fit_transform(
        train_covariates
    )
)

scaled_validation_covariates = (
    future_covariate_scaler.transform(
        validation_covariates
    )
)

INPUT_CHUNK_LENGTH = 100
OUTPUT_CHUNK_LENGTH = 32

import torch

from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="min",
)

model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,

    batch_size=256,
    n_epochs=100,

    likelihood=None,
    loss_fn=torch.nn.HuberLoss(),

    random_state=42,

    add_relative_index=True,

    save_checkpoints=True,
    force_reset=True,
    model_name="daily_festive_tft_boss_approach",

    pl_trainer_kwargs={
        "accelerator": "gpu"
            if torch.cuda.is_available()
            else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
    },
)

model.fit(
    series=scaled_train_series,
    future_covariates=scaled_train_covariates,

    val_series=scaled_validation_series,
    val_future_covariates=scaled_validation_covariates,

    verbose=True,
)

scaled_validation_forecasts = model.predict(
    n=122,
    series=scaled_train_series,
    future_covariates=scaled_validation_covariates,
)

validation_forecasts = target_scaler.inverse_transform(
    scaled_validation_forecasts
)

validation_forecasts = [
    forecast.map(
        lambda x: np.maximum(x, 0)
    )
    for forecast in validation_forecasts
]

records = []

for forecast in validation_forecasts:

    series_name = (
        forecast.static_covariates[
            "PARENT_DEALER_CODE_MODEL_FAMILY"
        ].iloc[0]
    )

    indices = forecast.time_index
    predictions = forecast.values().flatten()

    for idx, pred in zip(indices, predictions):

        records.append(
            {
                "PARENT_DEALER_CODE_MODEL_FAMILY": series_name,
                "SERIES_INDEX": int(idx),
                "PREDICTED_SALES": round(float(pred), 2),
            }
        )

prediction_df = pd.DataFrame(records)