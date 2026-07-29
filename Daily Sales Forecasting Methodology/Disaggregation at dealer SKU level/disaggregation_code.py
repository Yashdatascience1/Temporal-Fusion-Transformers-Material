from snowflake.snowpark import Window
from snowflake.snowpark.functions import col, when, lit, count, round as round_

w = Window.partition_by("PARENT_DEALER_CODE", "MODEL_FAMILY_CODE")

final_sales_df = final_sales_df.with_column(
    "NUM_ACTIVE_SKUS", count(lit(1)).over(w)
)

final_sales_df = final_sales_df.with_column(
    "INDIVIDUAL_WEIGHTS",
    round_(
        when(col("SKU_FAMILY_SALES_ACROSS_3_MONTHS") == 0, lit(1) / col("NUM_ACTIVE_SKUS"))
        .when(col("SKU_SALES_ACROSS_3_MONTHS") == 0, lit(0))
        .otherwise(col("SKU_SALES_ACROSS_3_MONTHS") / col("SKU_FAMILY_SALES_ACROSS_3_MONTHS")),
        5,
    ),
)