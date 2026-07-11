"""
Daily festive sales forecasting with one global TFT model in Darts.

Approach implemented
--------------------
- Train only on Sep 1-Dec 31 for 2023, 2024, and 2025.
- Forecast Sep 1-Dec 31, 2026.
- One global model is trained across all
  PARENT_DEALER_CODE_MODEL_FAMILY series.
- SERIES_INDEX is the Darts time axis:
      2023 ->   1 to 122
      2024 -> 123 to 244
      2025 -> 245 to 366
      2026 -> 367 to 488
- The actual calendar date is retained as metadata but is not used as the
  TimeSeries index in this restricted-season approach.

Important assumptions
---------------------
1. The input data has one row per series per retained date.
2. Missing historical rows are NOT automatically converted into zero sales.
   Validate the source-system meaning before doing that.
3. NET_SALES is populated for 2023-2025 and null for 2026.
4. The 2026 festive calendar is already corrected and merged into the data.
5. Static covariates are constant inside each series.

Typical installation
--------------------
pip install "u8darts[torch]" pandas numpy scikit-learn openpyxl

The exact Darts/Lightning import paths can differ slightly by installed
version. This script uses the modern Darts interface and supports both
lightning.pytorch and pytorch_lightning callback imports.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.metrics import mae, rmse
from darts.models import TFTModel

try:
    from darts.utils.likelihood_models import QuantileRegression
except ImportError:
    QuantileRegression = None

try:
    from lightning.pytorch.callbacks import EarlyStopping
except ImportError:
    from pytorch_lightning.callbacks import EarlyStopping


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForecastConfig:
    """Central configuration for the daily festive TFT pipeline."""

    # Input/output
    input_path: str
    output_dir: str = "daily_tft_outputs"
    sheet_name: str | int = 0

    # Core columns
    group_col: str = "PARENT_DEALER_CODE_MODEL_FAMILY"
    time_col: str = "SERIES_INDEX"
    date_col: str = "CAL_DATE"
    target_col: str = "NET_SALES"
    year_col: str = "YEAR"

    # Retained periods
    historical_years: Tuple[int, ...] = (2023, 2024, 2025)
    forecast_year: int = 2026
    validation_year: int = 2025

    # Expected seasonal structure
    days_per_season: int = 122
    expected_index_ranges: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: {
            2023: (1, 122),
            2024: (123, 244),
            2025: (245, 366),
            2026: (367, 488),
        }
    )

    # Model settings
    input_chunk_length: int = 60
    output_chunk_length: int = 14
    hidden_size: int = 32
    lstm_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.10
    batch_size: int = 256
    n_epochs: int = 100
    learning_rate: float = 1e-3
    random_state: int = 42
    num_loader_workers: int = 0

    # Probabilistic settings
    use_quantile_regression: bool = True
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    num_prediction_samples: int = 100

    # Training controls
    patience: int = 10
    min_delta: float = 1e-4
    gradient_clip_val: float = 0.1
    accelerator: str = "auto"
    devices: int = 1

    # Static covariates: one value per series
    static_covariates: Tuple[str, ...] = (
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
    )

    # Known future calendar covariates
    calendar_covariates: Tuple[str, ...] = (
        "YEAR",
        "MONTH",
        "DAY_OF_THE_WEEK",
        "DAY_OF_THE_MONTH",
    )

    # Known future festive covariates
    festival_covariates: Tuple[str, ...] = (
        "HARTALIK_TEEJ",
        "GANESH_CHATURTHI",
        "JANMASHTAMI",
        "VISHWAKARMA_PUJA",
        "KARWA_CHAUTH",
        "ONAM",
        "MARRIAGE_DAY",
        "N-16", "N-15", "N-14", "N-13", "N-12", "N-11", "N-10",
        "N-9", "N-8", "N-7", "N-6", "N-5", "N-4", "N-3", "N-2",
        "N-1", "N", "N+1", "N+2", "N+3", "N+4", "N+5", "N+6",
        "N+7", "N+8", "N+9",
        "D-3", "D-2", "D-1", "D", "D+1", "D+2", "D+3", "D+4",
        "D+5", "D+6",
    )

    @property
    def future_covariates(self) -> List[str]:
        return list(self.calendar_covariates + self.festival_covariates)

    @property
    def train_years(self) -> Tuple[int, ...]:
        return tuple(
            year
            for year in self.historical_years
            if year != self.validation_year
        )


# ---------------------------------------------------------------------------
# Reproducibility and logging
# ---------------------------------------------------------------------------

def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and Torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(output_dir: Path) -> logging.Logger:
    """Create console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "daily_tft_pipeline.log"

    logger = logging.getLogger("daily_tft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_input_data(config: ForecastConfig) -> pd.DataFrame:
    """Load CSV, Parquet, or Excel model data."""
    path = Path(config.input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=config.sheet_name)
    else:
        raise ValueError(
            "Unsupported input type. Use CSV, Parquet, XLSX, or XLS."
        )

    # Remove quotation marks accidentally embedded in Snowflake column names,
    # e.g. '"N-16"' becomes 'N-16'.
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
    )

    return df


def standardize_daily_data(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Standardize data types without hiding data-quality problems."""
    df = df.copy()

    if config.date_col in df.columns:
        df[config.date_col] = pd.to_datetime(
            df[config.date_col],
            errors="raise",
        )

    numeric_columns = [
        config.time_col,
        config.year_col,
        "MONTH",
        "DAY_OF_THE_MONTH",
        config.target_col,
    ]

    numeric_columns += list(config.festival_covariates)

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Festival indicators are known zeros when no festival marker applies.
    # Do not use df.fillna(0), because future NET_SALES must remain null.
    existing_festival_cols = [
        col for col in config.festival_covariates if col in df.columns
    ]
    df[existing_festival_cols] = (
        df[existing_festival_cols]
        .fillna(0.0)
        .astype("float32")
    )

    df = encode_weekday(df)

    df = df.sort_values(
        [config.group_col, config.time_col]
    ).reset_index(drop=True)

    return df


def encode_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DAY_OF_THE_WEEK into a numeric known-future covariate.

    Accepted examples:
    - MONDAY, Tuesday, friday
    - 0 to 6
    - 1 to 7

    Mapping used for text:
    Monday=0, ..., Sunday=6.
    """
    df = df.copy()
    col = "DAY_OF_THE_WEEK"

    if col not in df.columns:
        return df

    weekday_map = {
        "MONDAY": 0,
        "TUESDAY": 1,
        "WEDNESDAY": 2,
        "THURSDAY": 3,
        "FRIDAY": 4,
        "SATURDAY": 5,
        "SUNDAY": 6,
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }

    if (
        pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
    ):
        cleaned = df[col].astype(str).str.strip().str.upper()
        mapped = cleaned.map(weekday_map)

        # Try numeric strings for values not recognized as names.
        numeric_fallback = pd.to_numeric(cleaned, errors="coerce")
        df[col] = mapped.fillna(numeric_fallback)
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Fail early when a required field is absent."""
    required = {
        config.group_col,
        config.time_col,
        config.date_col,
        config.target_col,
        config.year_col,
        *config.static_covariates,
        *config.future_covariates,
    }

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_unique_series_time(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Ensure one observation per series and SERIES_INDEX."""
    duplicate_mask = df.duplicated(
        subset=[config.group_col, config.time_col],
        keep=False,
    )

    if duplicate_mask.any():
        sample = df.loc[
            duplicate_mask,
            [
                config.group_col,
                config.date_col,
                config.time_col,
                config.year_col,
            ],
        ].head(30)

        raise ValueError(
            "Duplicate series/index combinations found.\n"
            f"Sample:\n{sample.to_string(index=False)}"
        )


def validate_contiguous_index(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """
    Ensure SERIES_INDEX increments by exactly one inside each series.

    This is essential because, in this approach, Darts treats the retained
    Sep-Dec observations as a continuous artificial seasonal timeline.
    """
    index_gap = (
        df.groupby(config.group_col, sort=False)[config.time_col]
        .diff()
    )

    bad_mask = index_gap.notna() & index_gap.ne(1)

    if bad_mask.any():
        sample = df.loc[
            bad_mask,
            [
                config.group_col,
                config.date_col,
                config.year_col,
                config.time_col,
            ],
        ].copy()
        sample["INDEX_GAP"] = index_gap.loc[bad_mask].values

        raise ValueError(
            "Non-contiguous SERIES_INDEX found.\n"
            f"Sample:\n{sample.head(30).to_string(index=False)}"
        )


def validate_year_index_ranges(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Check expected row counts and index ranges for every series-year."""
    summary = (
        df.groupby([config.group_col, config.year_col], observed=True)
        .agg(
            minimum_index=(config.time_col, "min"),
            maximum_index=(config.time_col, "max"),
            row_count=(config.time_col, "size"),
        )
        .reset_index()
    )

    expected = pd.DataFrame(
        [
            {
                config.year_col: year,
                "expected_minimum_index": bounds[0],
                "expected_maximum_index": bounds[1],
                "expected_row_count": config.days_per_season,
            }
            for year, bounds in config.expected_index_ranges.items()
        ]
    )

    checked = summary.merge(
        expected,
        on=config.year_col,
        how="left",
        validate="many_to_one",
    )

    invalid = checked[
        checked["expected_row_count"].isna()
        | checked["minimum_index"].ne(checked["expected_minimum_index"])
        | checked["maximum_index"].ne(checked["expected_maximum_index"])
        | checked["row_count"].ne(checked["expected_row_count"])
    ]

    if not invalid.empty:
        raise ValueError(
            "Unexpected row counts or SERIES_INDEX ranges.\n"
            f"Sample:\n{invalid.head(30).to_string(index=False)}"
        )

    return checked


def validate_date_coverage(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Check that every series-year spans Sep 1-Dec 31."""
    summary = (
        df.groupby([config.group_col, config.year_col], observed=True)
        .agg(
            minimum_date=(config.date_col, "min"),
            maximum_date=(config.date_col, "max"),
            unique_dates=(config.date_col, "nunique"),
        )
        .reset_index()
    )

    summary["expected_minimum_date"] = pd.to_datetime(
        summary[config.year_col].astype(str) + "-09-01"
    )
    summary["expected_maximum_date"] = pd.to_datetime(
        summary[config.year_col].astype(str) + "-12-31"
    )

    invalid = summary[
        summary["minimum_date"].ne(summary["expected_minimum_date"])
        | summary["maximum_date"].ne(summary["expected_maximum_date"])
        | summary["unique_dates"].ne(config.days_per_season)
    ]

    if not invalid.empty:
        raise ValueError(
            "Incomplete or unexpected date coverage.\n"
            f"Sample:\n{invalid.head(30).to_string(index=False)}"
        )


def validate_static_covariates(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Ensure every declared static field is constant within a series."""
    nunique = (
        df.groupby(config.group_col, observed=True)[
            list(config.static_covariates)
        ]
        .nunique(dropna=False)
    )

    changing = (nunique > 1).stack()
    changing = changing[changing]

    if not changing.empty:
        raise ValueError(
            "Some static covariates change within a series.\n"
            f"Sample:\n{changing.head(30)}"
        )

    null_counts = df[list(config.static_covariates)].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        raise ValueError(
            "Null values found in static covariates:\n"
            f"{null_counts.to_string()}"
        )


def validate_covariates(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Known future covariates must be complete and numeric."""
    covariates = config.future_covariates

    null_counts = df[covariates].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        raise ValueError(
            "Null values found in future covariates:\n"
            f"{null_counts.to_string()}"
        )

    non_numeric = [
        col
        for col in covariates
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(
            "All Darts temporal covariates must be numeric. "
            f"Non-numeric columns: {non_numeric}"
        )


def validate_target_partition(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    """Ensure historical targets exist and future targets remain unknown."""
    historical_mask = df[config.year_col].isin(config.historical_years)
    future_mask = df[config.year_col].eq(config.forecast_year)

    historical_nulls = df.loc[historical_mask, config.target_col].isna().sum()

    if historical_nulls > 0:
        raise ValueError(
            f"Historical {config.target_col} has "
            f"{historical_nulls:,} null values."
        )

    future_populated = df.loc[future_mask, config.target_col].notna().sum()

    if future_populated > 0:
        raise ValueError(
            f"{future_populated:,} future {config.target_col} values are "
            "populated. Keep 2026 targets null unless they are genuine actuals."
        )

    negative_historical = (
        df.loc[historical_mask, config.target_col] < 0
    ).sum()

    if negative_historical > 0:
        raise ValueError(
            f"Historical target has {negative_historical:,} negative values."
        )


def run_all_validations(
    df: pd.DataFrame,
    config: ForecastConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run all structural checks before constructing Darts objects."""
    logger.info("Running data validation checks.")

    validate_required_columns(df, config)
    validate_unique_series_time(df, config)
    validate_contiguous_index(df, config)
    range_summary = validate_year_index_ranges(df, config)
    validate_date_coverage(df, config)
    validate_static_covariates(df, config)
    validate_covariates(df, config)
    validate_target_partition(df, config)

    logger.info(
        "Validation passed for %s global series.",
        df[config.group_col].nunique(),
    )
    return range_summary


# ---------------------------------------------------------------------------
# Darts TimeSeries construction
# ---------------------------------------------------------------------------

def create_target_series(
    historical_df: pd.DataFrame,
    config: ForecastConfig,
) -> List[TimeSeries]:
    """Create one target TimeSeries per business series."""
    columns = [
        config.time_col,
        config.group_col,
        config.target_col,
        *config.static_covariates,
    ]

    series_list = TimeSeries.from_group_dataframe(
        df=historical_df[columns],
        group_cols=config.group_col,
        time_col=config.time_col,
        value_cols=config.target_col,
        static_cols=list(config.static_covariates),
        freq=1,
        fill_missing_dates=False,
    )

    return list(series_list)


def create_future_covariate_series(
    df: pd.DataFrame,
    config: ForecastConfig,
) -> List[TimeSeries]:
    """Create one known-future covariate TimeSeries per business series."""
    columns = [
        config.time_col,
        config.group_col,
        *config.future_covariates,
    ]

    covariates = TimeSeries.from_group_dataframe(
        df=df[columns],
        group_cols=config.group_col,
        time_col=config.time_col,
        value_cols=config.future_covariates,
        freq=1,
        fill_missing_dates=False,
    )

    return list(covariates)


def get_series_id(series: TimeSeries, group_col: str) -> str:
    """Read the group identifier stored by from_group_dataframe()."""
    static_covariates = series.static_covariates

    if static_covariates is None or group_col not in static_covariates.columns:
        raise KeyError(
            f"{group_col} was not found in TimeSeries static covariates."
        )

    return str(static_covariates.iloc[0][group_col])


def sort_series_by_group(
    series_list: Sequence[TimeSeries],
    group_col: str,
) -> List[TimeSeries]:
    """Keep target and covariate lists in deterministic group order."""
    return sorted(
        series_list,
        key=lambda series: get_series_id(series, group_col),
    )


def assert_series_alignment(
    target_series: Sequence[TimeSeries],
    covariate_series: Sequence[TimeSeries],
    config: ForecastConfig,
) -> None:
    """Verify one-to-one ordering and index compatibility."""
    if len(target_series) != len(covariate_series):
        raise ValueError(
            "Target and covariate series counts differ: "
            f"{len(target_series)} versus {len(covariate_series)}."
        )

    for target, covariates in zip(target_series, covariate_series):
        target_id = get_series_id(target, config.group_col)
        covariate_id = get_series_id(covariates, config.group_col)

        if target_id != covariate_id:
            raise ValueError(
                f"Series ordering mismatch: {target_id} versus {covariate_id}."
            )

        if not covariates.time_index.isin(target.time_index).all():
            raise ValueError(
                f"Covariate index does not cover target index for {target_id}."
            )


# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------

def split_for_validation(
    target_series: Sequence[TimeSeries],
    config: ForecastConfig,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """
    Train on 2023-2024 and validate on 2025.

    With a one-based SERIES_INDEX:
      validation starts at 245.
    """
    validation_start = config.expected_index_ranges[
        config.validation_year
    ][0]

    train_list: List[TimeSeries] = []
    validation_list: List[TimeSeries] = []

    for series in target_series:
        train_list.append(
            series.slice(series.start_time(), validation_start - 1)
        )
        validation_list.append(
            series.slice(validation_start, series.end_time())
        )

    return train_list, validation_list


def trim_covariates_to_train_target(
    covariate_series: Sequence[TimeSeries],
    train_target_series: Sequence[TimeSeries],
    config: ForecastConfig,
) -> List[TimeSeries]:
    """Keep only the covariate history used while fitting the training target."""
    trimmed: List[TimeSeries] = []

    for covariates, target in zip(covariate_series, train_target_series):
        trimmed.append(
            covariates.slice(
                target.start_time(),
                target.end_time(),
            )
        )

    return trimmed


@dataclass
class FittedTransformers:
    target_scaler: Scaler
    covariate_scaler: Scaler
    static_transformer: StaticCovariatesTransformer


def fit_transformers(
    train_target_series: Sequence[TimeSeries],
    train_covariates: Sequence[TimeSeries],
) -> Tuple[List[TimeSeries], List[TimeSeries], FittedTransformers]:
    """
    Fit scaling only on the training period.

    Target scaler:
      global_fit=False -> one scaler per series.

    Covariate scaler:
      global_fit=True -> common scaling across all series. This is useful
      because calendar/festival variables have common meanings globally.

    Static transformer:
      encodes/scales static covariates across the global collection.
    """
    target_scaler = Scaler(global_fit=False, n_jobs=-1)
    covariate_scaler = Scaler(global_fit=True, n_jobs=-1)
    static_transformer = StaticCovariatesTransformer(n_jobs=-1)

    scaled_targets = target_scaler.fit_transform(list(train_target_series))
    scaled_targets = static_transformer.fit_transform(scaled_targets)

    scaled_covariates = covariate_scaler.fit_transform(
        list(train_covariates)
    )

    transformers = FittedTransformers(
        target_scaler=target_scaler,
        covariate_scaler=covariate_scaler,
        static_transformer=static_transformer,
    )

    return scaled_targets, scaled_covariates, transformers


def transform_targets(
    target_series: Sequence[TimeSeries],
    transformers: FittedTransformers,
) -> List[TimeSeries]:
    """Apply fitted target and static transformations."""
    scaled = transformers.target_scaler.transform(list(target_series))
    scaled = transformers.static_transformer.transform(scaled)
    return list(scaled)


def transform_covariates(
    covariates: Sequence[TimeSeries],
    transformers: FittedTransformers,
) -> List[TimeSeries]:
    """Apply the already fitted temporal-covariate scaler."""
    return list(
        transformers.covariate_scaler.transform(list(covariates))
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_tft_model(
    config: ForecastConfig,
    model_name: str,
) -> TFTModel:
    """Build the global daily TFT model."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        min_delta=config.min_delta,
        mode="min",
    )

    trainer_kwargs = {
        "accelerator": config.accelerator,
        "devices": config.devices,
        "callbacks": [early_stopping],
        "gradient_clip_val": config.gradient_clip_val,
        "enable_progress_bar": True,
    }

    model_kwargs = {
        "input_chunk_length": config.input_chunk_length,
        "output_chunk_length": config.output_chunk_length,
        "hidden_size": config.hidden_size,
        "lstm_layers": config.lstm_layers,
        "num_attention_heads": config.num_attention_heads,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "optimizer_kwargs": {"lr": config.learning_rate},
        "random_state": config.random_state,
        "add_relative_index": True,
        "save_checkpoints": True,
        "force_reset": True,
        "model_name": model_name,
        "pl_trainer_kwargs": trainer_kwargs,
    }

    if config.use_quantile_regression:
        if QuantileRegression is None:
            raise ImportError(
                "QuantileRegression could not be imported from Darts. "
                "Either update Darts or set use_quantile_regression=False."
            )

        model_kwargs["likelihood"] = QuantileRegression(
            quantiles=list(config.quantiles)
        )
    else:
        model_kwargs["loss_fn"] = torch.nn.HuberLoss()

    return TFTModel(**model_kwargs)


# ---------------------------------------------------------------------------
# Validation forecast and metrics
# ---------------------------------------------------------------------------

def clip_forecasts_at_zero(
    forecasts: Sequence[TimeSeries],
) -> List[TimeSeries]:
    """Sales cannot be negative."""
    return [
        forecast.map(lambda values: np.maximum(values, 0.0))
        for forecast in forecasts
    ]


def evaluate_validation_forecasts(
    actual_series: Sequence[TimeSeries],
    forecast_series: Sequence[TimeSeries],
    config: ForecastConfig,
) -> pd.DataFrame:
    """Calculate per-series MAE, RMSE, WAPE, and bias."""
    rows = []

    for actual, forecast in zip(actual_series, forecast_series):
        series_id = get_series_id(actual, config.group_col)

        actual_values = actual.values(copy=False).reshape(-1)
        forecast_values = forecast.values(copy=False).reshape(-1)

        absolute_error_sum = np.abs(
            actual_values - forecast_values
        ).sum()
        actual_sum = np.abs(actual_values).sum()

        wape = (
            absolute_error_sum / actual_sum
            if actual_sum > 0
            else np.nan
        )

        bias = (
            forecast_values.sum() - actual_values.sum()
        ) / actual_sum if actual_sum > 0 else np.nan

        rows.append(
            {
                config.group_col: series_id,
                "MAE": float(mae(actual, forecast)),
                "RMSE": float(rmse(actual, forecast)),
                "WAPE": float(wape) if not np.isnan(wape) else np.nan,
                "BIAS": float(bias) if not np.isnan(bias) else np.nan,
                "ACTUAL_TOTAL": float(actual_values.sum()),
                "FORECAST_TOTAL": float(forecast_values.sum()),
            }
        )

    return pd.DataFrame(rows)


def forecasts_to_dataframe(
    forecasts: Sequence[TimeSeries],
    reference_df: pd.DataFrame,
    config: ForecastConfig,
    value_name: str,
) -> pd.DataFrame:
    """Convert a list of Darts forecasts into a long pandas dataframe."""
    rows = []

    date_lookup = (
        reference_df[
            [config.group_col, config.time_col, config.date_col]
        ]
        .drop_duplicates()
        .set_index([config.group_col, config.time_col])[config.date_col]
        .to_dict()
    )

    for forecast in forecasts:
        series_id = get_series_id(forecast, config.group_col)
        values = forecast.values(copy=False).reshape(-1)

        for time_value, predicted_value in zip(
            forecast.time_index,
            values,
        ):
            time_value_int = int(time_value)
            rows.append(
                {
                    config.group_col: series_id,
                    config.time_col: time_value_int,
                    config.date_col: date_lookup.get(
                        (series_id, time_value_int)
                    ),
                    value_name: float(predicted_value),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End-to-end validation experiment
# ---------------------------------------------------------------------------

def run_validation_experiment(
    df: pd.DataFrame,
    config: ForecastConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest once:
      train = 2023-2024
      validation = 2025
    """
    historical_df = df[
        df[config.year_col].isin(config.historical_years)
    ].copy()

    target_series = sort_series_by_group(
        create_target_series(historical_df, config),
        config.group_col,
    )

    historical_covariates = sort_series_by_group(
        create_future_covariate_series(historical_df, config),
        config.group_col,
    )

    assert_series_alignment(
        target_series,
        historical_covariates,
        config,
    )

    train_targets, validation_targets = split_for_validation(
        target_series,
        config,
    )

    train_covariates = trim_covariates_to_train_target(
        historical_covariates,
        train_targets,
        config,
    )

    scaled_train_targets, scaled_train_covariates, transformers = (
        fit_transformers(
            train_targets,
            train_covariates,
        )
    )

    # For validation prediction, covariates must cover the encoder history and
    # the entire 2025 decoder horizon.
    scaled_historical_covariates = transform_covariates(
        historical_covariates,
        transformers,
    )

    model = build_tft_model(
        config,
        model_name="daily_festive_tft_validation",
    )

    logger.info(
        "Training validation model on %s series.",
        len(scaled_train_targets),
    )

    # val_series is included to monitor validation loss during fitting.
    # The validation target covers only 2025, while the validation covariates
    # cover the retained historical timeline through 2025.
    scaled_validation_targets = transform_targets(
        validation_targets,
        transformers,
    )

    model.fit(
        series=scaled_train_targets,
        future_covariates=scaled_train_covariates,
        val_series=scaled_validation_targets,
        val_future_covariates=scaled_historical_covariates,
        verbose=True,
        dataloader_kwargs={
            "num_workers": config.num_loader_workers,
        },
    )

    validation_horizon = config.days_per_season

    prediction_kwargs = {
        "n": validation_horizon,
        "series": scaled_train_targets,
        "future_covariates": scaled_historical_covariates,
    }

    if config.use_quantile_regression:
        prediction_kwargs["num_samples"] = config.num_prediction_samples

    scaled_forecasts = model.predict(**prediction_kwargs)

    # Each target scaler corresponds to the same series order used in fit.
    forecasts = transformers.target_scaler.inverse_transform(
        list(scaled_forecasts)
    )
    forecasts = clip_forecasts_at_zero(forecasts)

    metrics_df = evaluate_validation_forecasts(
        validation_targets,
        forecasts,
        config,
    )

    forecast_df = forecasts_to_dataframe(
        forecasts,
        reference_df=df,
        config=config,
        value_name="VALIDATION_FORECAST",
    )

    metrics_path = output_dir / "validation_metrics_2025.csv"
    forecasts_path = output_dir / "validation_forecasts_2025.csv"

    metrics_df.to_csv(metrics_path, index=False)
    forecast_df.to_csv(forecasts_path, index=False)

    logger.info("Saved validation metrics to %s", metrics_path)
    logger.info("Saved validation forecasts to %s", forecasts_path)

    return metrics_df, forecast_df


# ---------------------------------------------------------------------------
# Final model and 2026 forecast
# ---------------------------------------------------------------------------

def run_final_forecast(
    df: pd.DataFrame,
    config: ForecastConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Fit on 2023-2025 and forecast the full Sep-Dec 2026 season.
    """
    historical_df = df[
        df[config.year_col].isin(config.historical_years)
    ].copy()

    target_series = sort_series_by_group(
        create_target_series(historical_df, config),
        config.group_col,
    )

    training_covariates = sort_series_by_group(
        create_future_covariate_series(historical_df, config),
        config.group_col,
    )

    full_covariates = sort_series_by_group(
        create_future_covariate_series(df, config),
        config.group_col,
    )

    assert_series_alignment(target_series, training_covariates, config)
    assert_series_alignment(target_series, full_covariates, config)

    (
        scaled_targets,
        scaled_training_covariates,
        transformers,
    ) = fit_transformers(
        target_series,
        training_covariates,
    )

    scaled_full_covariates = transform_covariates(
        full_covariates,
        transformers,
    )

    model = build_tft_model(
        config,
        model_name="daily_festive_tft_final",
    )

    logger.info(
        "Training final model on 2023-2025 for %s series.",
        len(scaled_targets),
    )

    model.fit(
        series=scaled_targets,
        future_covariates=scaled_training_covariates,
        verbose=True,
        dataloader_kwargs={
            "num_workers": config.num_loader_workers,
        },
    )

    prediction_kwargs = {
        "n": config.days_per_season,
        "series": scaled_targets,
        "future_covariates": scaled_full_covariates,
    }

    if config.use_quantile_regression:
        prediction_kwargs["num_samples"] = config.num_prediction_samples

    scaled_forecasts = model.predict(**prediction_kwargs)

    forecasts = transformers.target_scaler.inverse_transform(
        list(scaled_forecasts)
    )
    forecasts = clip_forecasts_at_zero(forecasts)

    forecast_df = forecasts_to_dataframe(
        forecasts,
        reference_df=df,
        config=config,
        value_name="FORECAST_NET_SALES",
    )

    # Restrict output explicitly to 2026 indices.
    future_min, future_max = config.expected_index_ranges[
        config.forecast_year
    ]
    forecast_df = forecast_df[
        forecast_df[config.time_col].between(future_min, future_max)
    ].copy()

    output_path = output_dir / "daily_tft_forecast_2026.csv"
    forecast_df.to_csv(output_path, index=False)

    logger.info("Saved final 2026 forecast to %s", output_path)
    return forecast_df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def save_config(config: ForecastConfig, output_dir: Path) -> None:
    """Save the exact experiment configuration for reproducibility."""
    config_dict = vars(config).copy()

    # JSON does not support tuple keys/values directly in all contexts.
    config_dict["historical_years"] = list(config.historical_years)
    config_dict["quantiles"] = list(config.quantiles)
    config_dict["static_covariates"] = list(config.static_covariates)
    config_dict["calendar_covariates"] = list(config.calendar_covariates)
    config_dict["festival_covariates"] = list(config.festival_covariates)
    config_dict["expected_index_ranges"] = {
        str(year): list(bounds)
        for year, bounds in config.expected_index_ranges.items()
    }

    with (output_dir / "experiment_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config_dict, file, indent=2)


def run_pipeline(
    config: ForecastConfig,
    run_validation: bool = True,
    run_final: bool = True,
) -> None:
    """Run validation, model training, and forecasting."""
    output_dir = Path(config.output_dir)
    logger = configure_logging(output_dir)
    set_random_seeds(config.random_state)
    save_config(config, output_dir)

    logger.info("Loading input data from %s", config.input_path)
    df = load_input_data(config)
    df = standardize_daily_data(df, config)

    logger.info(
        "Loaded %,d rows, %,d columns, and %,d unique series.",
        len(df),
        len(df.columns),
        df[config.group_col].nunique()
        if config.group_col in df.columns
        else 0,
    )

    range_summary = run_all_validations(df, config, logger)
    range_summary.to_csv(
        output_dir / "series_year_index_validation.csv",
        index=False,
    )

    if run_validation:
        metrics_df, _ = run_validation_experiment(
            df,
            config,
            output_dir,
            logger,
        )

        logger.info(
            "Validation summary | median WAPE=%.4f | "
            "weighted total bias=%.4f",
            metrics_df["WAPE"].median(skipna=True),
            (
                metrics_df["FORECAST_TOTAL"].sum()
                - metrics_df["ACTUAL_TOTAL"].sum()
            )
            / max(metrics_df["ACTUAL_TOTAL"].sum(), 1e-9),
        )

    if run_final:
        run_final_forecast(
            df,
            config,
            output_dir,
            logger,
        )

    logger.info("Daily festive TFT pipeline completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one global Darts TFT model for daily festive sales."
        )
    )

    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to CSV, Parquet, XLSX, or XLS model data.",
    )
    parser.add_argument(
        "--output-dir",
        default="daily_tft_outputs",
        help="Directory for forecasts, metrics, logs, and config.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or zero-based sheet index.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the 2025 holdout experiment.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip final training and 2026 forecasting.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--input-chunk-length",
        type=int,
        default=60,
        help="Number of prior retained daily observations used by TFT.",
    )
    parser.add_argument(
        "--output-chunk-length",
        type=int,
        default=14,
        help="Number of steps TFT directly learns to forecast.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use Huber loss instead of quantile regression.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sheet_name: str | int
    try:
        sheet_name = int(args.sheet_name)
    except (TypeError, ValueError):
        sheet_name = args.sheet_name

    config = ForecastConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        sheet_name=sheet_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        input_chunk_length=args.input_chunk_length,
        output_chunk_length=args.output_chunk_length,
        use_quantile_regression=not args.deterministic,
    )

    run_pipeline(
        config=config,
        run_validation=not args.skip_validation,
        run_final=not args.skip_final,
    )


if __name__ == "__main__":
    main()
