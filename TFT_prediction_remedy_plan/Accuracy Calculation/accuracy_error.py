keys = ["PARENT_DEALER_CODE", "SKU", "CAL_DATE"]

# Date range and type of predictions
print(pred_df.schema)
pred_df.select(F.min("CAL_DATE"), F.max("CAL_DATE"), F.sum("SKU_PREDICTION")).show()

# Duplicate keys (fan-out would inflate sums)
print(pred_df.group_by(keys).count().filter(F.col("COUNT") > 1).count())
print(actual_sales_df.group_by(keys).count().filter(F.col("COUNT") > 1).count())

# Parent dealer overlap between the two pipelines
a = actual_sales_df.select("PARENT_DEALER_CODE").distinct()
p = pred_df.select("PARENT_DEALER_CODE").distinct()
print(a.count(), p.count(), a.join(p, on="PARENT_DEALER_CODE").count())

# SKU overlap
a_s = actual_sales_df.select("SKU").distinct()
p_s = pred_df.select("SKU").distinct()
print(a_s.count(), p_s.count(), a_s.join(p_s, on="SKU").count())

START, END = '2026-09-01', '2026-09-24'
keys = ["PARENT_DEALER_CODE_MODEL_FAMILY", "SKU", "CAL_DATE"]

pred_window = (join_with_predictions_sf
    .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
    .filter(F.col("CAL_DATE").between(F.lit(START), F.lit(END)))
    .group_by(keys)
    .agg(F.sum("SKU_PREDICTION").alias("SKU_PREDICTION")))

# actual_sales_df here = the one from step 14 of script 1 (same OBD mapping + parent_map)
actual_window = (actual_sales_df
    .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
    .group_by(keys)
    .agg(F.sum("NET_SALES").alias("NET_SALES")))

comparison = (pred_window.join(actual_window, on=keys, how="full")
    .with_column("SKU_PREDICTION", F.coalesce(F.col("SKU_PREDICTION"), F.lit(0)))
    .with_column("NET_SALES", F.coalesce(F.col("NET_SALES"), F.lit(0))))

comparison.select(
    F.sum("SKU_PREDICTION").alias("SKU_PREDICTION_SUM"),
    F.sum("NET_SALES").alias("ACTUAL_SALES_SUM")
).show()