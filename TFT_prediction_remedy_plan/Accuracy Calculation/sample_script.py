# def fetchSKUSupercedence_snowpark(session,SKU_SUPERCEDENCE_MODEL_FAMILY):
def fetchSKUSupercedence_snowpark(session):
    data = session.table("MOP_DATABASE.SOQ.SKU_SUPERCEDENCE")
    data_1 = session.table("MOP_DATABASE.SOQ.MODEL_FAMILY_MAPPING") 
    result = data.join(data_1, on="MODEL", how="left") 
    result = result.with_column("SKU_UNIQUE_FAMILY_CODE", F.col("UNIQUEFAMILYCODE")) 

    for old_col in result.columns:
        new_col = old_col.replace('"','')
        result = result.rename(old_col, new_col) 
    
    result = result.with_column("MODEL_FAMILY_CODE",
                            F.concat(F.col("MODEL_FAMILY"), F.lit('<>'),
                            F.substring(F.col("UNIQUEFAMILYCODE"),
                                F.charindex(F.lit('<>'), F.col("UNIQUEFAMILYCODE")) + F.lit(2))
                                )) 
    
    result = result.rename("UNIQUEFAMILYCODE", "UNIQUE FAMILY CODE") 
    # result.write.mode("overwrite").save_as_table(SKU_SUPERCEDENCE_MODEL_FAMILY) 
    return result

result = fetchSKUSupercedence_snowpark(session)


def return_models_for_forecasting(session, use_selected_models, logger):
    """
    Returns a list of model names or None based on user filter parameter settings.
    """
    if not use_selected_models:
        # logger.info("Model filter disabled — considering ALL models in ECR sales.")
        return None

    models_for_forecasting = session.table('MOP_DATABASE.SOQ.MODELS_FOR_FORECASTING').to_pandas()
    name_of_models = models_for_forecasting["MODEL_NAME"].tolist()
    # logger.info("Model filter enabled — %s models selected for forecasting.", len(name_of_models))
    return name_of_models

name_of_models = return_models_for_forecasting(session,True,None)

def get_ecr_sales_snowpark(session, customer_types, start_date, name_of_models,end_date):
    ecr_sales = session.table("ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS") \
        .filter(F.col("X_CUSTOMER_TYPE").in_(customer_types)) \
        .filter((F.col("CAL_DATE") >= F.lit(start_date)) & (F.col("CAL_DATE") <= F.lit(end_date)))
        
    if name_of_models is not None:
        ecr_sales = ecr_sales.filter(F.col("MODEL").isin(name_of_models))
        
    ecr_sales = ecr_sales.with_column("NET_SALES", 
    F.when(
        (F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES")) < 0, 
        F.lit(0)
    ).otherwise(
        F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES")
    )
)


    return ecr_sales


ecr_sales = get_ecr_sales_snowpark(session,start_date='2026-06-01',end_date='2026-08-31',customer_types=['Individual'],name_of_models=name_of_models)

ecr_sales = ecr_sales.select('DEALER_CODE','CAL_DATE','SKU','NET_SALES')

active_sku_supercedence = result.filter(F.lower(F.col("SKUSTATUS")) == 'active')

active_sku_supercedence = active_sku_supercedence.select("SKU","MODEL_FAMILY_CODE")

#OBD_mapping
obd_data = session.table("MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW") 
sku_supercedence = session.table("MOP_DATABASE.SOQ.SKU_SUPERCEDENCE")

obd_data_joined = obd_data.join(
    sku_supercedence.select("SKU", "SKUSTATUS"), 
    obd_data["CURRENT_OBD_SKU"] == sku_supercedence["SKU"], 
    how='left'
)

obd_data_active_skus = obd_data_joined.filter(F.lower(F.col("SKUSTATUS")) == 'active') \
                                        .select("CURRENT_OBD_SKU", "PREVIOUS_OBD_SKU") 

ecr_sales = ecr_sales.join(obd_data_active_skus, ecr_sales["SKU"] == obd_data_active_skus["PREVIOUS_OBD_SKU"], how="left")
ecr_sales = ecr_sales.with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))

joined_df = ecr_sales.join(right=active_sku_supercedence,on=["SKU"])

joined_df = joined_df.select("DEALER_CODE","CAL_DATE","MODEL_FAMILY_CODE","SKU","NET_SALES")

dealer_master = session.sql("""SELECT PARENT_DEALER_CODE,DEALER_CODE
                            FROM ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER
                            WHERE NOT REGEXP_LIKE(DEALER_CODE,'^17.*')
                            AND LOWER(ORG_STATUS) = 'active'
                            AND DEALER_DFNC = 'Y'
                            """)

family_level_daily_sales = dealer_master.join(joined_df,on=["DEALER_CODE"])

family_level_daily_sales = family_level_daily_sales.with_column("PARENT_DEALER_CODE_MODEL_FAMILY",F.concat(F.trim(F.col("PARENT_DEALER_CODE")),F.lit('<>'),F.col("MODEL_FAMILY_CODE")))


family_sales_3_months = family_level_daily_sales.group_by("PARENT_DEALER_CODE_MODEL_FAMILY").agg(F.sum("NET_SALES").alias("LAST_3_MONTHS_FAMILY_SALES"))


sku_level_sales_3_months = family_level_daily_sales.group_by("PARENT_DEALER_CODE_MODEL_FAMILY","SKU").agg(F.sum("NET_SALES").alias("LAST_3_MONTHS_SKU_SALES"))


family_and_sku_with_last_3_months_sales = family_sales_3_months.join(right=sku_level_sales_3_months,on=["PARENT_DEALER_CODE_MODEL_FAMILY"])


