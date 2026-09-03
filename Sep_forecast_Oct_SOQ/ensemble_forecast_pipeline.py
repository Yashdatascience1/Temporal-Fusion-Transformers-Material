"""
Ensemble forecast build: TFT (Series A) + Snowflake XGBoost (Series A + Series B).

Two output modes:
  ensemble : left-join SF forecast with TFT forecast, average where both exist,
             fall back to SF where TFT is missing.
  concat   : stack TFT Series A on top of SF Series B (no averaging).

Run:
    python ensemble_forecast_pipeline.py --output-table ENSEMBLE_FCST_SEP_OCT_26
    python ensemble_forecast_pipeline.py --output-table X --upload-mode concat --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Defaults (override via CLI)
# --------------------------------------------------------------------------- #
UTILS_PATH = r"C:\Users\G0004878\Desktop\TFT_Data\utils_files"
DEFAULT_TFT_CSV = (
    r"C:\Users\G0004878\Desktop\TFT_Data\Sep_forecast_Oct_SOQ\Step 2 - Modelling"
    r"\Final_output_Series_A_TFT_Sep_Oct.csv"
)
DEFAULT_DATABASE = "MOP_DATABASE"
DEFAULT_SCHEMA = "SOQ"
DEFAULT_FORECAST_VIEW = "MOP_DATABASE.SOQ.SNOWFLAKE_FORECAST_SEP_26_TO_OCT_26_SERIES_VIEW"
DEFAULT_INVALID_TABLE = "MOP_DATABASE.SOQ.INVALID_PARENT_DEALER_COMBINATIONS"
DEFAULT_LOG_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\logs"

KEY_COLS = ["MONTH_OF_SALE", "PARENT_DEALER_CODE_MODEL_FAMILY"]

logger = logging.getLogger("ensemble_forecast")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str, verbose: bool = False) -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"ensemble_forecast_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)

    return log_file


def banner(heading: str) -> None:
    logger.info("=" * 88)
    logger.info(heading.upper())
    logger.info("=" * 88)


def log_monthly_totals(df: pd.DataFrame, value_col: str, heading: str) -> pd.DataFrame:
    """Log month-wise totals in chronological order (not alphabetical)."""
    banner(heading)

    if df.empty:
        logger.warning("Dataframe is EMPTY - nothing to summarise.")
        return pd.DataFrame(columns=["MONTH", value_col])

    tmp = df.copy()
    tmp["_PERIOD"] = pd.to_datetime(tmp["MONTH_OF_SALE"]).dt.to_period("M")

    summary = (
        tmp.groupby("_PERIOD", as_index=False)
        .agg(TOTAL=(value_col, "sum"), ROWS=(value_col, "size"),
             NULLS=(value_col, lambda s: int(s.isna().sum())))
        .sort_values("_PERIOD")
    )
    summary["MONTH"] = summary["_PERIOD"].dt.strftime("%B - %Y")

    for _, r in summary.iterrows():
        logger.info(
            "  %-18s | total = %14s | rows = %7d | nulls = %5d",
            r["MONTH"], f"{r['TOTAL']:,.2f}", int(r["ROWS"]), int(r["NULLS"]),
        )
    logger.info("  %-18s | total = %14s | rows = %7d | nulls = %5d",
                "GRAND TOTAL", f"{summary['TOTAL'].sum():,.2f}",
                int(summary["ROWS"].sum()), int(summary["NULLS"].sum()))

    return summary[["MONTH", "TOTAL", "ROWS", "NULLS"]].rename(columns={"TOTAL": value_col})


def log_data_quality(df: pd.DataFrame, name: str, value_col: str) -> None:
    """Row counts, nulls, duplicate keys, negatives. Duplicates matter most: they
    fan out on the merge and inflate totals."""
    logger.info("[%s] shape = %s", name, df.shape)
    logger.info("[%s] columns = %s", name, list(df.columns))

    nulls = df[[c for c in KEY_COLS + [value_col] if c in df.columns]].isnull().sum()
    for c, n in nulls.items():
        level = logger.warning if n else logger.info
        level("[%s] nulls in %-35s : %d", name, c, int(n))

    if all(c in df.columns for c in KEY_COLS):
        dupes = int(df.duplicated(subset=KEY_COLS).sum())
        if dupes:
            logger.error(
                "[%s] %d DUPLICATE (month, series) rows - these will fan out on join "
                "and inflate totals. Investigate before trusting output.", name, dupes,
            )
        else:
            logger.info("[%s] no duplicate (month, series) keys.", name)

    if value_col in df.columns:
        neg = int((df[value_col] < 0).sum())
        if neg:
            logger.warning("[%s] %d rows with negative %s.", name, neg, value_col)
        logger.info("[%s] distinct series = %d",
                    name, df["PARENT_DEALER_CODE_MODEL_FAMILY"].nunique())


# --------------------------------------------------------------------------- #
# Snowflake session
# --------------------------------------------------------------------------- #
def build_session(database: str, schema: str):
    sys.path.append(UTILS_PATH)
    import Snowflake_configuration  # noqa: E402
    from snowflake.snowpark.session import Session  # noqa: E402

    logger.info("Opening Snowflake session ...")
    session = Session.builder.configs(Snowflake_configuration.ds1_role_json).create()
    session.use_database(database)
    session.use_schema(schema)
    logger.info("Session ready | database = %s | schema = %s", database, schema)
    return session


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #
def normalise_columns(df: pd.DataFrame, value_col_out: str) -> pd.DataFrame:
    """Bring any of the source shapes onto the canonical schema:
    MONTH_OF_SALE | PARENT_DEALER_CODE_MODEL_FAMILY | <value_col_out>"""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    if "SERIES" in df.columns and "PARENT_DEALER_CODE_MODEL_FAMILY" not in df.columns:
        df = df.rename(columns={"SERIES": "PARENT_DEALER_CODE_MODEL_FAMILY"})
    if "DATES" in df.columns and "MONTH_OF_SALE" not in df.columns:
        df = df.rename(columns={"DATES": "MONTH_OF_SALE"})
    if "PREDICTED_SALES" in df.columns and value_col_out != "PREDICTED_SALES":
        df = df.rename(columns={"PREDICTED_SALES": value_col_out})

    missing = [c for c in KEY_COLS + [value_col_out] if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing after normalisation: {missing} "
                       f"(found: {list(df.columns)})")

    # Strip stray double-quotes that wrap the series identifier in the view.
    n_quoted = int(df["PARENT_DEALER_CODE_MODEL_FAMILY"].astype(str).str.contains('"').sum())
    if n_quoted:
        logger.info("Stripping embedded double-quotes from %d series values.", n_quoted)
    df["PARENT_DEALER_CODE_MODEL_FAMILY"] = (
        df["PARENT_DEALER_CODE_MODEL_FAMILY"].astype(str).str.replace('"', "", regex=False).str.strip()
    )

    df["MONTH_OF_SALE"] = pd.to_datetime(df["MONTH_OF_SALE"])
    df[value_col_out] = pd.to_numeric(df[value_col_out], errors="coerce")
    return df[KEY_COLS + [value_col_out]]


def load_tft_series_a(csv_path: str) -> pd.DataFrame:
    banner("step 1 - load TFT forecast (series A)")
    logger.info("Reading %s", csv_path)

    df = pd.read_csv(csv_path, usecols=[0, 1, 2])
    logger.info("Raw TFT file shape = %s | raw columns = %s", df.shape, list(df.columns))

    df = normalise_columns(df, "TFT_PREDICTED_SALES")
    log_data_quality(df, "TFT_SERIES_A", "TFT_PREDICTED_SALES")
    log_monthly_totals(df, "TFT_PREDICTED_SALES", "Forecast by TFT for series A")
    return df


def load_sf_all_series(session, view: str) -> pd.DataFrame:
    banner("step 2 - load Snowflake XGBoost forecast (all series)")
    sql = f"SELECT * FROM {view}"
    logger.debug("SQL: %s", sql)

    df = session.sql(sql).to_pandas()
    logger.info("Raw SF view shape = %s | raw columns = %s", df.shape, list(df.columns))

    df = normalise_columns(df, "SF_PREDICTED_SALES")
    log_data_quality(df, "SF_ALL_SERIES", "SF_PREDICTED_SALES")
    log_monthly_totals(df, "SF_PREDICTED_SALES",
                       "Forecast by Snowflake XGBoost for all series")
    return df


def load_sf_series_b(session, view: str, invalid_table: str) -> pd.DataFrame:
    """Series B = series flagged in INVALID_PARENT_DEALER_COMBINATIONS.

    Quotes are stripped on BOTH sides inside SQL. Doing it in pandas after the
    fetch (as in the original) means the IN-clause compares quoted against
    unquoted values and can silently return zero rows.
    """
    banner("step 3 - load Snowflake XGBoost forecast (series B only)")
    sql = f"""
        SELECT *
        FROM {view}
        WHERE TRIM(REPLACE(SERIES, '"', '')) IN (
            SELECT TRIM(REPLACE(SERIES_NAME, '"', ''))
            FROM {invalid_table}
        )
    """
    logger.debug("SQL: %s", " ".join(sql.split()))

    df = session.sql(sql).to_pandas()
    logger.info("Raw series-B shape = %s", df.shape)

    if df.empty:
        logger.error(
            "Series B returned ZERO rows. Either the invalid-combination list is "
            "empty, or the join keys genuinely do not overlap. Do not assume this "
            "is correct - verify before uploading in concat mode."
        )

    df = normalise_columns(df, "SF_PREDICTED_SALES")
    log_data_quality(df, "SF_SERIES_B", "SF_PREDICTED_SALES")
    log_monthly_totals(df, "SF_PREDICTED_SALES",
                       "Forecast by Snowflake XGBoost for series B")
    return df


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def build_concat(tft_a: pd.DataFrame, sf_b: pd.DataFrame) -> pd.DataFrame:
    banner("step 4a - build CONCAT output (series A from TFT + series B from SF)")

    a = tft_a.rename(columns={"TFT_PREDICTED_SALES": "PREDICTED_SALES"}).copy()
    b = sf_b.rename(columns={"SF_PREDICTED_SALES": "PREDICTED_SALES"}).copy()
    a["SOURCE"] = "TFT_SERIES_A"
    b["SOURCE"] = "SF_SERIES_B"

    overlap = set(a["PARENT_DEALER_CODE_MODEL_FAMILY"]) & set(b["PARENT_DEALER_CODE_MODEL_FAMILY"])
    if overlap:
        logger.error(
            "%d series appear in BOTH the TFT (A) and SF (B) sets. Concatenating "
            "will double-count them. Sample: %s",
            len(overlap), sorted(overlap)[:5],
        )
    else:
        logger.info("Series A and series B are disjoint - no double counting.")

    out = pd.concat([b, a], ignore_index=True)
    log_data_quality(out, "CONCAT_OUTPUT", "PREDICTED_SALES")
    log_monthly_totals(
        out, "PREDICTED_SALES",
        "Forecast: series A (from TFT) and series B (from Snowflake)",
    )
    return out


def build_ensemble(sf_all: pd.DataFrame, tft_a: pd.DataFrame) -> pd.DataFrame:
    banner("step 4b - build ENSEMBLE output (TFT + Snowflake averaged on series A)")

    dropped = set(tft_a["PARENT_DEALER_CODE_MODEL_FAMILY"]) - set(
        sf_all["PARENT_DEALER_CODE_MODEL_FAMILY"]
    )
    if dropped:
        logger.warning(
            "%d TFT series are NOT present in the Snowflake view and will be "
            "DROPPED by the left join. Sample: %s",
            len(dropped), sorted(dropped)[:5],
        )

    rows_before = len(sf_all)
    df = sf_all.merge(tft_a, on=KEY_COLS, how="left")
    if len(df) != rows_before:
        logger.error(
            "Join changed row count %d -> %d. This means duplicate keys on the "
            "right side fanned out. Totals are NOT trustworthy.", rows_before, len(df),
        )
    else:
        logger.info("Join preserved row count (%d rows).", len(df))

    matched = int(df["TFT_PREDICTED_SALES"].notna().sum())
    logger.info("Rows with a TFT forecast (averaged) : %d (%.1f%%)",
                matched, 100 * matched / max(len(df), 1))
    logger.info("Rows without a TFT forecast (SF only): %d", len(df) - matched)

    # Null-safe average. The original overwrote this with a plain (SF+TFT)/2,
    # which turned every SF-only row into NaN and silently dropped it from sums.
    df["ENSEMBLE_SALES"] = np.where(
        df["TFT_PREDICTED_SALES"].isna(),
        df["SF_PREDICTED_SALES"],
        (df["TFT_PREDICTED_SALES"] + df["SF_PREDICTED_SALES"]) / 2.0,
    )

    residual_nulls = int(df["ENSEMBLE_SALES"].isna().sum())
    if residual_nulls:
        logger.error(
            "%d ENSEMBLE_SALES rows are still null (SF_PREDICTED_SALES was null "
            "too). These contribute 0 to the totals below.", residual_nulls,
        )

    log_data_quality(df, "ENSEMBLE_OUTPUT", "ENSEMBLE_SALES")
    log_monthly_totals(
        df, "ENSEMBLE_SALES",
        "Forecast: series A (ensembled TFT + Snowflake) and series B (Snowflake)",
    )

    # Side-by-side comparison so the ensemble effect is visible, not assumed.
    banner("ensemble vs component totals")
    for col in ["SF_PREDICTED_SALES", "TFT_PREDICTED_SALES", "ENSEMBLE_SALES"]:
        logger.info("  %-22s total = %14s", col, f"{df[col].sum():,.2f}")

    return df


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def upload(session, df: pd.DataFrame, table: str, write_mode: str, dry_run: bool) -> None:
    banner("step 5 - upload to Snowflake")
    logger.info("Target table = %s | write mode = %s | rows = %d | cols = %s",
                table, write_mode, len(df), list(df.columns))

    if df.empty:
        logger.error("Refusing to upload an empty dataframe.")
        return

    if dry_run:
        logger.warning("DRY RUN - skipping write. Preview of first 5 rows:")
        for line in df.head().to_string(index=False).splitlines():
            logger.warning("  %s", line)
        return

    if write_mode == "overwrite":
        logger.warning("Mode 'overwrite' will DROP and recreate %s.", table)

    session.create_dataframe(df).write.mode(write_mode).save_as_table(table)
    written = session.table(table).count()
    logger.info("Write complete. Row count in %s = %d", table, written)
    if written != len(df):
        logger.error("Row count mismatch: wrote %d, table holds %d.", len(df), written)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TFT + Snowflake ensemble forecast build")
    p.add_argument("--tft-csv", default=DEFAULT_TFT_CSV)
    p.add_argument("--forecast-view", default=DEFAULT_FORECAST_VIEW)
    p.add_argument("--invalid-table", default=DEFAULT_INVALID_TABLE)
    p.add_argument("--database", default=DEFAULT_DATABASE)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--output-table", required=True,
                   help="Target Snowflake table name.")
    p.add_argument("--upload-mode", choices=["ensemble", "concat"], default="ensemble",
                   help="Which dataframe to upload.")
    p.add_argument("--write-mode", choices=["overwrite", "append"], default="overwrite")
    p.add_argument("--dry-run", action="store_true",
                   help="Run everything and log it, but do not write to Snowflake.")
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_file = setup_logging(args.log_dir, args.verbose)

    banner("run started")
    logger.info("Log file: %s", log_file)
    for k, v in vars(args).items():
        logger.info("  param %-16s = %s", k, v)

    session = None
    try:
        session = build_session(args.database, args.schema)

        tft_a = load_tft_series_a(args.tft_csv)
        sf_all = load_sf_all_series(session, args.forecast_view)
        sf_b = load_sf_series_b(session, args.forecast_view, args.invalid_table)

        # Both are always built so the log carries both totals for comparison,
        # regardless of which one is uploaded.
        concat_df = build_concat(tft_a, sf_b)
        ensemble_df = build_ensemble(sf_all, tft_a)

        to_upload = ensemble_df if args.upload_mode == "ensemble" else concat_df
        logger.info("Selected '%s' dataframe for upload.", args.upload_mode)

        upload(session, to_upload, args.output_table, args.write_mode, args.dry_run)

        banner("run completed successfully")
        return 0

    except Exception:
        logger.exception("RUN FAILED - see traceback above.")
        return 1
    finally:
        if session is not None:
            try:
                session.close()
                logger.info("Snowflake session closed.")
            except Exception:
                logger.warning("Failed to close Snowflake session cleanly.", exc_info=True)


if __name__ == "__main__":
    sys.exit(main())
