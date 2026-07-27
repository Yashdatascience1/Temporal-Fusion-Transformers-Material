# =============================================================================
# STEP 3 — FAMILY -> SKU FORECAST DISAGGREGATION
# Refactored from pandas-in-Snowpark to native Snowpark DataFrame API.
#
# WHAT THIS SCRIPT ACTUALLY DOES (business logic, not mechanics):
#   The upstream forecast model predicts demand at DEALER x MODEL_FAMILY level
#   (e.g. "how many bikes of this family will Dealer X sell next month"), not
#   at the individual SKU (specific variant/color/trim) level. This script
#   takes that family-level number and splits ("disaggregates") it across the
#   individual active SKUs that belong to the family, using each SKU's
#   historical share of that dealer's family sales as the split ratio.
#   The core fan-out join that performs this split is inside
#   getStockDataMapping() — look for the comment marked "DISAGGREGATION STEP".
#
# WHY THIS REFACTOR MATTERS (mechanically):
#   The original script mixes session.sql()/session.table() (Snowpark) with
#   .to_pandas() immediately after almost every call. That means every
#   transformation after the first line of each function runs LOCALLY in the
#   Python process, not in Snowflake. For tables of any real size this is
#   slow and memory-bound. This version keeps everything as a Snowpark
#   DataFrame (lazy, pushed down to Snowflake as SQL) until the final
#   .write.save_as_table() calls.
#
# CONFIDENCE NOTE (read before trusting this in production):
#   I do not have your live Snowflake session or exact table schemas
#   (specifically: OBD2_MAPPING_VIEW, SKU_SUPERCEDENCE_MODEL_FAMILY, and the
#   prediction table's exact columns). Several joins below could hit
#   "ambiguous column" errors if two tables share a column name that pandas'
#   merge() was silently auto-suffixing (e.g. _x/_y) without you noticing.
#   I've flagged every such spot with "RISK:" comments. Snowpark will ERROR
#   LOUDLY on these instead of silently corrupting a column the way pandas
#   might have — treat that as a feature, but it means this WILL need at
#   least one debug pass against real tables before it's production-ready.
# =============================================================================

import snowflake.snowpark as snowpark
from snowflake.snowpark import Window
from snowflake.snowpark.functions import (
    col, lit, when, sum as sum_, count, count_distinct, split, trim,
    to_date, call_builtin, coalesce, round as round_, lag, stddev, concat,
    split_part,
)
from dateutil.relativedelta import relativedelta
import datetime

# -----------------------------------------------------------------------------
# CONFIG (unchanged from original — these are pure Python constants, no
# Snowpark involved, so there was nothing to refactor here)
# -----------------------------------------------------------------------------
PRED_LEVEL = "monthly"  # "weekly"

FORECAST_TABLE = 'MOP_DATABASE.SOQ.TEST_FORECASTS_WITH_MARKET_SHARE_SON_2025_VIEW_V2'
TEST_FORECAST_TABLE = 'MOP_DATABASE.SOQ.TEST_FORECASTS_PARENT_DEALER_MODEL_FAMILY_WITH_MARKET_SHARE_SON_2025'
TEST_DATA_TABLE = 'MOP_DATABASE.SOQ.TEST_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_SON_2025_UPDATED'
RAW_TEST_TABLE = TEST_DATA_TABLE + "_WEEKLY_REGULARISED"
PREDICTION_TABLE = 'MOP_DATABASE.SOQ.SOQ_PREDICTION_FINAL_VERSION'

IS_OBD = True  # consider stock/ECR of both current and previous OBD-variant SKU
OBD_MAPPING = 'MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'

CUSTOMER_TYPE_TO_CONSIDER = ['Individual']

ECR_GROUP_BY = ['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'X_MONTH_NAME', 'MODEL']

RUN_DATE = datetime.datetime.today().strftime('%Y%m%d')
RUN_VERSION = 33
MID_DATE = 15

BASE_SOQ_TABLE = "MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED"
DEMAND_VARIABILITY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAMILY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'
TRANSIT_TABLE = 'MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW'
SKU_SUPERCEDENCE_MODEL_FAMILY = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAY_2026_UPDATED_V2'

STOCK_DATE_TYPE = ["first"]
MONTHS = ['2026-06-01']

obd_flag = 'Y' if IS_OBD else 'N'


# -----------------------------------------------------------------------------
# fetchOBDData
# Original pandas: reads OBD_MAPPING, then OVERWRITES the 'SKU' column with
# the value of 'PREVIOUS_OBD_SKU'. Net effect: this table, once returned, is
# keyed to the *previous-generation* SKU code, while still carrying whatever
# CURRENT_OBD_SKU column the source table has. It's used downstream to map
# "if this old SKU sold, also count it as demand for the new SKU that
# replaced it" (OBD = presumably "Old/prior... SKU" superseding logic).
# -----------------------------------------------------------------------------
def fetchOBDData(session):
    obd_mapping = session.table(OBD_MAPPING)
    # Explicit select+alias instead of with_column("SKU", ...): with_column's
    # overwrite-vs-duplicate-column behavior when the target name already
    # exists is not something I can verify without your Snowpark version, so
    # this spells it out unambiguously instead. [Guessing on with_column
    # behavior — this select-based version avoids the question entirely.]
    other_cols = [c for c in obd_mapping.columns if c != "SKU"]
    obd_mapping = obd_mapping.select(*other_cols, col("PREVIOUS_OBD_SKU").alias("SKU"))
    return obd_mapping


# -----------------------------------------------------------------------------
# fetchStockData
# Pulls closing stock for a given date. If IS_OBD, stock sitting under the
# OLD sku code is folded into the NEW (current) sku code's stock number,
# because from a supply-planning view they're the same physical part.
# -----------------------------------------------------------------------------
def fetchStockData(session, date):
    stock_data = session.sql(f"""
        SELECT DEALER_CODE, MODEL, SKU, CLOSING_STOCK AS STK_AS_ON_DATE
        FROM ANALYTICS_DATABASE.ANALYTICS_SALES.STOCK_AVAILABILITY
        WHERE CAL_DATE = '{date}'
    """)
    if IS_OBD:
        obd_data = fetchOBDData(session)
        # RISK: obd_data may carry its own MODEL/DEALER_CODE-like columns.
        # Selecting only what we need before the join avoids an ambiguous
        # column error and mirrors what pandas' merge(..., on="SKU") does
        # implicitly (only SKU is the join key; other obd_data columns get
        # silently suffixed by pandas — Snowpark will not silently suffix,
        # it will error, unless you pass lsuffix/rsuffix to .join()).
        obd_data = obd_data.select("SKU", "CURRENT_OBD_SKU")
        stock_data = stock_data.join(obd_data, on="SKU", how="left")

        # fillna(SKU) -> coalesce: if there's no OBD mapping, the SKU is its
        # own "current" version.
        stock_data = stock_data.with_column(
            "CURRENT_OBD_SKU", coalesce(col("CURRENT_OBD_SKU"), col("SKU"))
        )

        stk_data = (
            stock_data
            .group_by("CURRENT_OBD_SKU", "DEALER_CODE", "MODEL")
            .agg(sum_("STK_AS_ON_DATE").alias("STK_AS_ON_DATE"))
        )
        stk_data = stk_data.rename(col("CURRENT_OBD_SKU"), "SKU")
        return stk_data
    return stock_data


# -----------------------------------------------------------------------------
# getParentDealerMapping
# Dealer hierarchy table -> derive the "parent" dealer code (a dealer group
# may have several outlets; SOQ planning happens at the parent level, not
# per-outlet). PAR_ORG_NAME is a hyphen-delimited string like
# "1234 - Some Dealer Group Name"; we want the piece before the first hyphen.
# -----------------------------------------------------------------------------
def getParentDealerMapping(session):
    df = session.sql("""
        SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE, PAR_ORG_NAME
        FROM FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH
        WHERE X_DEALER_CODE_HIER IS NOT NULL
    """)
    # split_part(str, delim, 1) == python str.split("-")[0]
    # [likely correct — split_part is a standard Snowflake SQL function and
    # is wrapped in snowpark.functions in reasonably recent Snowpark
    # versions; if your installed version doesn't expose it, swap for
    # call_builtin("split_part", col("PAR_ORG_NAME"), lit("-"), lit(1))]
    df = df.with_column(
        "PARENT_DEALER_CODE", trim(split_part(col("PAR_ORG_NAME"), lit("-"), lit(1)))
    )
    return df


# -----------------------------------------------------------------------------
# fetchSKUSupercedence
# Reference table mapping SKU -> UNIQUE FAMILY CODE, MODEL, SKUSTATUS (active
# / discontinued), used both to know which SKUs are "live" for a family and
# to know which family a SKU rolls up into.
# -----------------------------------------------------------------------------
def fetchSKUSupercedence(session):
    df = session.table(SKU_SUPERCEDENCE_MODEL_FAMILY).drop("UPDATED_ON")
    print("Supercedence columns:", df.columns)
    return df


# -----------------------------------------------------------------------------
# percentsku (original) -> add_percent_proportion (this version)
# Business logic for splitting family-level forecast across SKUs:
#   - If the family had ZERO sales across all its active SKUs recently,
#     split the forecast EQUALLY across active SKUs (no data to weight by).
#   - Else if THIS SKU specifically had zero sales, it gets 0% of the split.
#   - Else this SKU gets its share = (this SKU's sales) / (family's total
#     active-SKU sales).
# This was a row-wise pandas .apply(); it's a pure column expression, so it
# maps directly and losslessly to Snowpark's when/otherwise — no UDF needed.
# -----------------------------------------------------------------------------
def add_percent_proportion(df):
    return df.with_column(
        "PERCENT_PROPORTION",
        round_(
            when(col("TOTAL_DEALER_ACTIVE_SKU_SALES") == 0, lit(1) / col("NUM_ACTIVE_SKUS"))
            .when(col("DEALER_SKU_SALES") == 0, lit(0))
            .otherwise(col("DEALER_SKU_SALES") / col("TOTAL_DEALER_ACTIVE_SKU_SALES")),
            5,
        ),
    )


# -----------------------------------------------------------------------------
# getStockDate — pure Python date math, nothing to refactor. Unchanged.
# "first": last day of the month BEFORE the run month.
# "end": last day of the month BEFORE the planning month.
# "mid": the MID_DATE-th day of the month before planning month.
# -----------------------------------------------------------------------------
def getStockDate(date_period, planning_month):
    run_month = datetime.datetime.strptime(planning_month, "%Y-%m-%d") - relativedelta(months=1)
    if date_period == "end":
        return (
            datetime.datetime.strptime(planning_month, "%Y-%m-%d").replace(day=1) - relativedelta(days=1)
        ).strftime("%Y-%m-%d")
    if date_period == "first":
        return (run_month.replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
    if date_period == "mid":
        return run_month.replace(day=MID_DATE).strftime("%Y-%m-%d")


# -----------------------------------------------------------------------------
# write_to_snowflake
# Original took a pandas df and re-uploaded it via session.create_dataframe().
# Since everything is now already a Snowpark DataFrame end-to-end, this is
# just a thin wrapper around .write — kept only so call sites don't change.
# -----------------------------------------------------------------------------
def write_to_snowflake(df, table_name, mode="append"):
    df.write.mode(mode).save_as_table(table_name)


# -----------------------------------------------------------------------------
# processForecastTable
# 1. (weekly mode only) collapse week-level forecast rows up to month level.
# 2. Pull the family-level forecast for the target month.
# 3. Parse the SERIES key (format: "PARENTDEALER_MODELFAMILY_FAMILYCODE",
#    quote-wrapped on the first and third pieces) into separate columns.
# 4. Persist to the prediction table, tagged with this run's date/version.
# -----------------------------------------------------------------------------
def processForecastTable(session, run_date, month, run_version):
    if PRED_LEVEL == "weekly":
        test_df = session.table(TEST_FORECAST_TABLE)
        dates_df = session.table(RAW_TEST_TABLE).select(
            "Date", "Regularized_Date", "MONTH_DATE"
        ).distinct()
        test_df = test_df.rename(col("TS"), "Regularized_Date")
        test_df = test_df.join(dates_df, on="Regularized_Date", how="left")
        test_df = test_df.group_by("SERIES", "MONTH_DATE").agg(
            sum_("FORECAST").alias("FORECAST")
        )
        # negative forecasts clipped to 0, else rounded to nearest int
        test_df = test_df.with_column(
            "PREDICTED_SALES",
            when(col("FORECAST") < 0, lit(0)).otherwise(round_(col("FORECAST"), 0)),
        )
        test_df = test_df.select(
            col("MONTH_DATE").alias("DATES"), col("SERIES"), col("PREDICTED_SALES")
        )
        write_to_snowflake(test_df, FORECAST_TABLE, "overwrite")

    df = session.sql(f"SELECT * FROM {FORECAST_TABLE} WHERE DATES='{month}'")

    # SERIES looks like: "1234"_FAMILYNAME_"CODE"  (underscore-delimited,
    # outer quote chars on pieces 1 and 3 only, per original .strip('"')).
    df = df.with_column(
        "PARENT_DEALER_CODE", trim(split_part(col("SERIES"), lit("_"), lit(1)), lit('"'))
    )
    df = df.with_column("MODEL_FAMILY", split_part(col("SERIES"), lit("_"), lit(2)))
    df = df.with_column(
        "FAMILY_CODE", trim(split_part(col("SERIES"), lit("_"), lit(3)), lit('"'))
    )
    df = df.with_column(
        "UNIQUE FAMILY CODE", concat(col("MODEL_FAMILY"), lit("<>"), col("FAMILY_CODE"))
    )
    df = df.with_column("RUN_DATE", lit(run_date))
    df = df.with_column("RUN_VERSION", lit(run_version))

    df.write.mode("append").save_as_table(PREDICTION_TABLE)


def fetchPredictionData(session, run_date, month, run_version):
    return session.sql(f"""
        SELECT * FROM {PREDICTION_TABLE}
        WHERE RUN_DATE='{run_date}' AND RUN_VERSION={run_version}
    """)


# -----------------------------------------------------------------------------
# getECR ("Effective Customer Retail"? — actual retail sales feed)
# NET_SALES = invoiced - cancellations - returns are ALREADY signed in the
# source columns (original code just sums them, doesn't subtract), so I've
# preserved that assumption rather than "fixing" it — I don't know the sign
# convention of CANCELLED_SALES/RETURNED_SALES in your source table.
# [Guessing: preserved original arithmetic as-is; verify sign convention.]
# -----------------------------------------------------------------------------
def getECR(session, customer_type_to_include, start_date, end_date):
    customer_list = ",".join([f"'{t}'" for t in customer_type_to_include])
    query = f"""
        SELECT * FROM ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS
        WHERE X_CUSTOMER_TYPE IN ({customer_list})
        AND CAL_DATE >= '{start_date}' AND CAL_DATE < '{end_date}'
    """
    df = session.sql(query)
    df = df.with_column("DATE", to_date(col("CAL_DATE")))
    # TO_CHAR(date, 'YYYYMM') via call_builtin since date_format's exact
    # wrapper/format-string dialect differs across Snowpark versions.
    # [Guessing on exact snowpark.functions coverage — call_builtin is the
    # safe universal escape hatch to any native Snowflake SQL function.]
    df = df.with_column("X_MONTH_NAME", call_builtin("to_char", col("DATE"), lit("YYYYMM")))
    df = df.with_column(
        "NET_SALES", col("INVOICED_SALES") + col("CANCELLED_SALES") + col("RETURNED_SALES")
    )
    return df


# -----------------------------------------------------------------------------
# ECRAggregation
# Pulls the last 3 months of retail sales, maps each sale to its family
# (via SKU supercedence) and parent dealer, and — if OBD is on — re-keys the
# SKU column to the CURRENT obd sku (renaming the raw sku to ORIGINAL_SKU
# for traceability).
# -----------------------------------------------------------------------------
def ECRAggregation(session, run_date, sku_supercedence, parent_dealer_mapping):
    current_date = datetime.datetime.strptime(run_date, '%Y%m%d').replace(day=1)
    start_date = (current_date - relativedelta(months=3)).strftime("%Y-%m-%d")
    end_date = current_date.strftime("%Y-%m-%d")

    ecr_data = getECR(session, CUSTOMER_TYPE_TO_CONSIDER, start_date, end_date)

    # RISK: sku_supercedence and ecr_data may both have non-join columns
    # with the same name beyond SKU/MODEL. If .join() throws an ambiguous
    # column error, add lsuffix="_SKU" / rsuffix="_SUP" here.
    ecr_data = ecr_data.join(sku_supercedence, on=["SKU", "MODEL"], how="left")
    ecr_data = ecr_data.join(
        parent_dealer_mapping.select("DEALER_CODE", "PARENT_DEALER_CODE"),
        on="DEALER_CODE", how="left",
    )

    if IS_OBD:
        obd_data = fetchOBDData(session).select("SKU", "CURRENT_OBD_SKU")
        ecr_data = ecr_data.join(obd_data, on="SKU", how="left")
        ecr_data = ecr_data.with_column(
            "CURRENT_OBD_SKU", coalesce(col("CURRENT_OBD_SKU"), col("SKU"))
        )
        ecr_data = ecr_data.rename(col("SKU"), "ORIGINAL_SKU")
        ecr_data = ecr_data.rename(col("CURRENT_OBD_SKU"), "SKU")

    return ecr_data


# -----------------------------------------------------------------------------
# getStockDataMapping
# *** THIS IS THE ACTUAL FAMILY -> SKU DISAGGREGATION FUNCTION ***
# `data` comes in as one row per (dealer, family) forecast. The join marked
# "DISAGGREGATION STEP" below fans each of those rows out into N rows — one
# per active SKU in that family — which is what turns a family-level number
# into SKU-level candidate rows (the actual $ split happens later, in main(),
# via add_percent_proportion).
# -----------------------------------------------------------------------------
def getStockDataMapping(session, months, data, date_period, parent_dealer_mapping, sku_supercedence):
    stock_date = getStockDate(date_period, months)

    stock_data = fetchStockData(session, stock_date)
    stock_data = stock_data.join(
        parent_dealer_mapping.select("DEALER_CODE", "PARENT_DEALER_CODE"),
        on="DEALER_CODE", how="left",
    )
    stock_data = stock_data.join(sku_supercedence, on=["SKU", "MODEL"], how="left")

    stocks_sku = stock_data.group_by("PARENT_DEALER_CODE", "SKU").agg(
        sum_("STK_AS_ON_DATE").alias("STK_AS_ON_DATE")
    )

    # Count of active SKUs per family — needed both for the equal-split
    # fallback in add_percent_proportion and to know which families even
    # have any active SKU at all (families with none get dropped later).
    num_active = (
        sku_supercedence
        .filter(col("SKUSTATUS") == "active")
        .group_by("UNIQUE FAMILY CODE")
        .agg(count_distinct(col("SKU")).alias("NUM_ACTIVE_SKUS"))
    )

    data = data.join(num_active, on="UNIQUE FAMILY CODE", how="left")

    sku_data = sku_supercedence.filter(col("SKUSTATUS") == "active")

    # ============================ DISAGGREGATION STEP =======================
    # Fan-out join: each (dealer, family) forecast row is replicated once per
    # active SKU belonging to that family. This is where "family level"
    # becomes "SKU level" structurally (the split RATIO is applied later).
    # RISK: `data` (prediction columns) and `sku_data` (supercedence columns)
    # may share a column name beyond "UNIQUE FAMILY CODE" — if so this join
    # will raise an ambiguous-column error where pandas' merge() would have
    # silently suffixed it. Add lsuffix/rsuffix if that happens.
    # ==========================================================================
    soq_data_sku = data.join(sku_data, on="UNIQUE FAMILY CODE", how="left")

    soq_data_sku = soq_data_sku.with_column(
        "PARENT_DEALER_CODE", col("PARENT_DEALER_CODE").cast("string")
    )
    stocks_sku = stocks_sku.with_column(
        "PARENT_DEALER_CODE", col("PARENT_DEALER_CODE").cast("string")
    )

    soq_data_sku = soq_data_sku.join(stocks_sku, on=["PARENT_DEALER_CODE", "SKU"], how="left")

    # Debug/audit snapshot — kept from the original so you can inspect which
    # rows had no active-SKU match.
    soq_data_sku.write.mode("overwrite").save_as_table(
        "MOP_DATABASE.SOQ.INACTIVE_FAMILIES_ACTIVE_FAMILIES_MERGE_TEMP"
    )

    # Drop families that have zero active SKUs — nothing to disaggregate to.
    soq_data_sku = soq_data_sku.filter(col("NUM_ACTIVE_SKUS").is_not_null())

    return soq_data_sku


# -----------------------------------------------------------------------------
# calculateDemandVariability (original, pandas groupby().apply()) ->
# add_demand_variability (this version, native window functions)
#
# Original per-group logic: sort by month, take month-over-month diffs,
# return the std-dev of those diffs. If a group has 2 or fewer data points,
# hardcode variability = 1 (not enough history to measure volatility).
#
# This is the one function that genuinely cannot become a single column
# expression the way percentsku could — it needs an ORDERED window per
# group. Two-pass approach: LAG() to get the diff, then STDDEV() of the
# diffs, aggregated back down to one row per group.
#
# [likely]: Snowflake's STDDEV() = STDDEV_SAMP() = pandas .std() default
# (ddof=1), so this should match numerically. Not verified against your
# actual data — recommend spot-checking a handful of groups against the old
# pandas output before switching over.
# -----------------------------------------------------------------------------
def add_demand_variability(df, group_cols, order_col="X_MONTH_NAME", value_col="NET_SALES"):
    w = Window.partition_by(*group_cols).order_by(col(order_col))
    diffed = df.with_column("SALES_DIFF", col(value_col) - lag(col(value_col)).over(w))

    counts = df.group_by(*group_cols).agg(count(lit(1)).alias("N_MONTHS"))
    stds = (
        diffed.filter(col("SALES_DIFF").is_not_null())
        .group_by(*group_cols)
        .agg(stddev(col("SALES_DIFF")).alias("DEMAND_VARIABILITY_RAW"))
    )

    result = counts.join(stds, on=group_cols, how="left")
    result = result.with_column(
        "DEMAND_VARIABILITY",
        when(col("N_MONTHS") > 2, coalesce(col("DEMAND_VARIABILITY_RAW"), lit(1))).otherwise(lit(1)),
    )
    return result.drop("N_MONTHS", "DEMAND_VARIABILITY_RAW")


def createDemandVariability(session, months, run_date=RUN_DATE, run_version=RUN_VERSION):
    group_by = ['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'X_MONTH_NAME', 'SKU']
    start_date = (
        datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1) - relativedelta(months=12)
    ).strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1).strftime("%Y-%m-%d")

    ecr_data = getECR(session, CUSTOMER_TYPE_TO_CONSIDER, start_date, end_date)

    parent_dealer_mapping = getParentDealerMapping(session)
    sku_supercedence = fetchSKUSupercedence(session)

    ecr_data = ecr_data.join(sku_supercedence, on=["SKU", "MODEL"], how="left")
    ecr_data = ecr_data.join(
        parent_dealer_mapping.select("DEALER_CODE", "PARENT_DEALER_CODE"),
        on="DEALER_CODE", how="left",
    )

    if IS_OBD:
        obd_data = fetchOBDData(session).select("SKU", "CURRENT_OBD_SKU")
        ecr_data = ecr_data.join(obd_data, on="SKU", how="left")
        ecr_data = ecr_data.with_column(
            "CURRENT_OBD_SKU", coalesce(col("CURRENT_OBD_SKU"), col("SKU"))
        )
        ecr_data = ecr_data.rename(col("SKU"), "ORIGINAL_SKU")
        ecr_data = ecr_data.rename(col("CURRENT_OBD_SKU"), "SKU")

    total_sales_by_month = ecr_data.group_by(*group_by).agg(
        sum_("NET_SALES").cast("float").alias("NET_SALES")
    )

    demand_variability = add_demand_variability(
        total_sales_by_month,
        group_cols=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU'],
    )
    demand_variability = demand_variability.with_column("PLANNING_MONTH", lit(months))
    demand_variability = demand_variability.with_column("ECR_START_DATE", lit(start_date))
    demand_variability = demand_variability.with_column("ECR_END_DATE", lit(end_date))
    demand_variability = demand_variability.with_column("RUN_DATE", lit(run_date))
    demand_variability = demand_variability.with_column("RUN_VERSION", lit(run_version))
    demand_variability = demand_variability.with_column("IS_OBD", lit(obd_flag))

    demand_variability.write.mode("append").save_as_table(DEMAND_VARIABILITY_TABLE)


def createDemandVariabilityByFamily(session, months, run_date=RUN_DATE, run_version=RUN_VERSION):
    group_by = ['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'X_MONTH_NAME']
    start_date = (
        datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1) - relativedelta(months=12)
    ).strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1).strftime("%Y-%m-%d")

    ecr_data = getECR(session, CUSTOMER_TYPE_TO_CONSIDER, start_date, end_date)

    parent_dealer_mapping = getParentDealerMapping(session)
    sku_supercedence = fetchSKUSupercedence(session)

    ecr_data = ecr_data.join(sku_supercedence, on=["SKU", "MODEL"], how="left")
    ecr_data = ecr_data.join(
        parent_dealer_mapping.select("DEALER_CODE", "PARENT_DEALER_CODE"),
        on="DEALER_CODE", how="left",
    )

    if IS_OBD:
        obd_data = fetchOBDData(session).select("SKU", "CURRENT_OBD_SKU")
        ecr_data = ecr_data.join(obd_data, on="SKU", how="left")
        ecr_data = ecr_data.with_column(
            "CURRENT_OBD_SKU", coalesce(col("CURRENT_OBD_SKU"), col("SKU"))
        )
        ecr_data = ecr_data.rename(col("SKU"), "ORIGINAL_SKU")
        ecr_data = ecr_data.rename(col("CURRENT_OBD_SKU"), "SKU")

    total_sales_by_month = ecr_data.group_by(*group_by).agg(
        sum_("NET_SALES").cast("float").alias("NET_SALES")
    )

    demand_variability = add_demand_variability(
        total_sales_by_month,
        group_cols=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'],
    )
    demand_variability = demand_variability.with_column("PLANNING_MONTH", lit(months))
    demand_variability = demand_variability.with_column("ECR_START_DATE", lit(start_date))
    demand_variability = demand_variability.with_column("ECR_END_DATE", lit(end_date))
    demand_variability = demand_variability.with_column("RUN_DATE", lit(run_date))
    demand_variability = demand_variability.with_column("RUN_VERSION", lit(run_version))
    demand_variability = demand_variability.with_column("IS_OBD", lit(obd_flag))

    demand_variability.write.mode("append").save_as_table(DEMAND_VARIABILITY_FAMILY_TABLE)


# -----------------------------------------------------------------------------
# main
# Orchestration, unchanged in structure/order from the original. This is
# where the disaggregation output (soq_data_sku, SKU-level) gets its sales
# history joined in, the split ratio applied, transit data attached, and the
# final SOQ base table written.
# -----------------------------------------------------------------------------
def main(session: snowpark.Session):
    parent_dealer_mapping = getParentDealerMapping(session)
    sku_supercedence = fetchSKUSupercedence(session)

    for months in MONTHS:
        processForecastTable(session, RUN_DATE, months, RUN_VERSION)

        data = session.sql(f"""
            SELECT DISTINCT * FROM {PREDICTION_TABLE}
            WHERE RUN_DATE='{RUN_DATE}' AND DATES='{months}' AND RUN_VERSION={RUN_VERSION}
        """)
        data = data.with_column("PARENT_DEALER_CODE", col("PARENT_DEALER_CODE").cast("string"))
        data = data.with_column("DATES_STR", call_builtin("to_char", col("DATES"), lit("YYYY-MM-DD")))
        print("Prediction row count:", data.count())

        for stock_period in STOCK_DATE_TYPE:
            soq_data_sku = getStockDataMapping(
                session, months, data, stock_period, parent_dealer_mapping, sku_supercedence
            )
            ecr_data = ECRAggregation(session, RUN_DATE, sku_supercedence, parent_dealer_mapping)

            # Sales at FAMILY level (last 3 months) per dealer — the
            # denominator context for the equal-split fallback.
            dealer_family_sales = (
                ecr_data.group_by("PARENT_DEALER_CODE", "UNIQUE FAMILY CODE")
                .agg(sum_("NET_SALES").alias("DEALER_FAMILY_CODE_NET_SALES"))
            )
            # Sales at SKU level (last 3 months) per dealer — the numerator
            # for each SKU's split ratio.
            dealer_family_sku_sales = (
                ecr_data.group_by("PARENT_DEALER_CODE", "UNIQUE FAMILY CODE", "SKU")
                .agg(sum_("NET_SALES").alias("DEALER_SKU_SALES"))
            )

            soq_data_sku = soq_data_sku.with_column(
                "PARENT_DEALER_CODE", col("PARENT_DEALER_CODE").cast("integer").cast("string")
            )

            soq_data_sku = soq_data_sku.join(
                dealer_family_sales, on=["PARENT_DEALER_CODE", "UNIQUE FAMILY CODE"], how="left"
            )
            soq_data_sku = soq_data_sku.with_column(
                "DEALER_FAMILY_CODE_NET_SALES", coalesce(col("DEALER_FAMILY_CODE_NET_SALES"), lit(0))
            )

            soq_data_sku = soq_data_sku.join(
                dealer_family_sku_sales,
                on=["PARENT_DEALER_CODE", "UNIQUE FAMILY CODE", "SKU"], how="left",
            )

            active_family_dealer_sku_sales = (
                soq_data_sku.group_by("PARENT_DEALER_CODE", "UNIQUE FAMILY CODE")
                .agg(sum_("DEALER_SKU_SALES").alias("TOTAL_DEALER_ACTIVE_SKU_SALES"))
            )
            soq_data_sku = soq_data_sku.join(
                active_family_dealer_sku_sales,
                on=["PARENT_DEALER_CODE", "UNIQUE FAMILY CODE"], how="left",
            )
            soq_data_sku = soq_data_sku.with_column(
                "DEALER_SKU_SALES", coalesce(col("DEALER_SKU_SALES"), lit(0))
            )

            # OBD edge case: a family may have sales, but a specific SKU
            # within it (post OBD re-keying) may show zero — computed
            # separately per (dealer, SKU_UNIQUE_FAMILY_CODE) to catch that.
            dealer_sku_total = (
                soq_data_sku.group_by("PARENT_DEALER_CODE", "SKU_UNIQUE_FAMILY_CODE")
                .agg(sum_("DEALER_SKU_SALES").alias("DEALER_ACTIVE_SKU_TOTAL_SALES"))
            )
            soq_data_sku = soq_data_sku.join(
                dealer_sku_total, on=["PARENT_DEALER_CODE", "SKU_UNIQUE_FAMILY_CODE"], how="left"
            )

            # ---- THE ACTUAL SPLIT ----
            soq_data_sku = add_percent_proportion(soq_data_sku)
            soq_data_sku = soq_data_sku.with_column(
                "PREDICTED_SALES_SKU",
                when(col("NUM_ACTIVE_SKUS") == 1, col("PREDICTED_SALES"))
                .otherwise(col("PERCENT_PROPORTION") * col("PREDICTED_SALES")),
            )

            transit_data = session.table(TRANSIT_TABLE)
            soq_data_sku = soq_data_sku.drop(
                "DEALER_ACTIVE_SKU_TOTAL_SALES", "TOTAL_DEALER_ACTIVE_SKU_SALES"
            )
            print("soq_data_sku row count:", soq_data_sku.count())

            soq_final_data = soq_data_sku.join(
                transit_data, on=["PARENT_DEALER_CODE", "SKU"], how="left"
            )
            print("soq_final_data row count:", soq_final_data.count())

            stock_date = getStockDate(stock_period, months)
            soq_final_data = soq_final_data.with_column("STOCK_DATE_PERIOD", lit(stock_period))
            soq_final_data = soq_final_data.with_column("STOCK_DATE", lit(stock_date))
            soq_final_data = soq_final_data.with_column("PLANNING_MONTH", lit(months))
            soq_final_data = soq_final_data.with_column("RUN_DATE", lit(RUN_DATE))

            # drop_duplicates() -> distinct() (both dedupe on ALL columns)
            soq_final_data = soq_final_data.distinct()

            soq_final_data = soq_final_data.with_column("RUN_VERSION", lit(RUN_VERSION))
            soq_final_data = soq_final_data.with_column("IS_OBD", lit(obd_flag))

            soq_final_data.write.mode("append").save_as_table(BASE_SOQ_TABLE)

    for months in MONTHS:
        print(months)
        createDemandVariability(session, months, RUN_DATE, RUN_VERSION)
        createDemandVariabilityByFamily(session, months, RUN_DATE, RUN_VERSION)

    return session.table(DEMAND_VARIABILITY_FAMILY_TABLE)
