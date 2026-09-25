from snowflake.snowpark import functions as F

START, END = '2026-09-01', '2026-09-24'
keys = ["PARENT_DEALER_CODE_MODEL_FAMILY", "SKU", "CAL_DATE"]

# ------------------------------------------------------------
# 0. Build comparison_df (predictions vs actuals, same keys)
# ------------------------------------------------------------
pred_window = (join_with_predictions_sf
    .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
    .filter(F.col("CAL_DATE").between(F.lit(START), F.lit(END)))
    .group_by(keys)
    .agg(F.sum("SKU_PREDICTION").alias("SKU_PREDICTION")))

actual_window = (actual_sales_df
    .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
    .filter(F.col("CAL_DATE").between(F.lit(START), F.lit(END)))
    .group_by(keys)
    .agg(F.sum("NET_SALES").alias("NET_SALES")))

comparison_df = (pred_window.join(actual_window, on=keys, how="full")
    .with_column("SKU_PREDICTION", F.coalesce(F.col("SKU_PREDICTION"), F.lit(0)))
    .with_column("NET_SALES", F.coalesce(F.col("NET_SALES"), F.lit(0))))

# ------------------------------------------------------------
# 1. Attach ABC category from the proportions table (Jun-Aug basis)
# ------------------------------------------------------------
abc_lookup = final_output.select(
    "PARENT_DEALER_CODE_MODEL_FAMILY", "SKU", "ABC_CATEGORY"
).distinct()

comp = comparison_df.join(
    abc_lookup, on=["PARENT_DEALER_CODE_MODEL_FAMILY", "SKU"], how="left"
).with_column(
    "ABC_CATEGORY", F.coalesce(F.col("ABC_CATEGORY"), F.lit("UNMAPPED"))
)

# ------------------------------------------------------------
# 2. Choose evaluation grain
#    "daily"  -> dealer-family x SKU x day
#    "period" -> dealer-family x SKU, summed over Sep 1-24
# ------------------------------------------------------------
GRAIN = "period"

if GRAIN == "period":
    comp = comp.group_by("PARENT_DEALER_CODE_MODEL_FAMILY", "SKU", "ABC_CATEGORY").agg(
        F.sum("SKU_PREDICTION").alias("SKU_PREDICTION"),
        F.sum("NET_SALES").alias("NET_SALES"),
    )

# ------------------------------------------------------------
# 3. Keep only rows with actual sales, then compute errors
# ------------------------------------------------------------
comp = comp.filter(F.col("NET_SALES") > 0)

comp = comp.with_column(
    "ABS_ERROR", F.abs(F.col("NET_SALES") - F.col("SKU_PREDICTION"))
).with_column(
    "APE", F.col("ABS_ERROR") / F.col("NET_SALES")
)

# ------------------------------------------------------------
# 4. Accuracy metrics per category + overall
# ------------------------------------------------------------
def accuracy_metrics(df):
    return df.agg(
        F.count(F.lit(1)).alias("N_ROWS"),
        F.sum("NET_SALES").alias("ACTUAL"),
        F.sum("SKU_PREDICTION").alias("PREDICTED"),
        F.avg("APE").alias("MAPE"),
        (F.sum("ABS_ERROR") / F.sum("NET_SALES")).alias("WAPE"),
        (F.sum("SKU_PREDICTION") / F.sum("NET_SALES") - F.lit(1)).alias("BIAS"),
    )

by_category = accuracy_metrics(comp.group_by("ABC_CATEGORY"))
overall = accuracy_metrics(comp).with_column("ABC_CATEGORY", F.lit("ALL"))

accuracy = by_category.union_all_by_name(overall).with_column(
    "ACCURACY_1_MINUS_MAPE", F.greatest(F.lit(0), F.lit(1) - F.col("MAPE"))
).with_column(
    "ACCURACY_1_MINUS_WAPE", F.greatest(F.lit(0), F.lit(1) - F.col("WAPE"))
).sort("ABC_CATEGORY")

accuracy.show()
accuracy_pd = accuracy.to_pandas()   # optional: for pandas / Excel export