#Returns a list of model names or None based on user filter parameter settings.
#This helps us to filter out the models for which forecasting is not required
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


def fetchSKUSupercedence_snowpark(session, SKU_SUPERCEDENCE_MODEL_FAMILY):
    data = session.table("MOP_DATABASE.SOQ.SKU_SUPERCEDENCE") 
    data_1 = session.table("MOP_DATABASE.SOQ.MODEL_FAMILY_MAPPING") 
    result = data.join(data_1, on="MODEL", how="left") 
    result = result.with_column("SKU_UNIQUE_FAMILY_CODE", col("UNIQUEFAMILYCODE")) 

    for old_col in result.columns:
        new_col = old_col.replace('"','')
        result = result.rename(old_col, new_col) 
    
    result = result.with_column("MODEL_FAMILY_CODE",
                            F.concat(F.col("MODEL_FAMILY"), F.lit('<>'),
                            F.substring(F.col("UNIQUEFAMILYCODE"),
                                F.charindex(F.lit('<>'), F.col("UNIQUEFAMILYCODE")) + F.lit(2))
                                )) 
    
    result = result.rename("UNIQUEFAMILYCODE", "UNIQUE FAMILY CODE") 
    result.write.mode("overwrite").save_as_table(SKU_SUPERCEDENCE_MODEL_FAMILY) 
    return result


def get_ecr_sales_snowpark(session, customer_types, start_date, name_of_models,end_date):
    ecr_sales = session.table("ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS") \
        .filter(col("X_CUSTOMER_TYPE").in_(customer_types)) \
        .filter((col("CAL_DATE") >= F.lit(start_date)) & (col("CAL_DATE") <= F.lit(end_date)))
        
    if name_of_models is not None:
        ecr_sales = ecr_sales.filter(col("MODEL").isin(name_of_models))
        
    ecr_sales = ecr_sales.with_column("NET_SALES", 
        col("INVOICED_SALES") + col("CANCELLED_SALES") + col("RETURNED_SALES")) 
    return ecr_sales


def process_ecr_aggregation_snowpark(session, agg_type, customer_types,run_date, name_of_models, SKU_SUPERCEDENCE_MODEL_FAMILY):
    start_date = (run_date + relativedelta(day=1, months=-3)).date()

    end_date = start_date + relativedelta(months=3, days=-1)
    ecr_sales = get_ecr_sales_snowpark(session, customer_types, start_date, name_of_models,end_date)
    
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
    ecr_sales = ecr_sales.with_column("SKU", F.coalesce(col("CURRENT_OBD_SKU"), col("SKU"))) 
    
    sku_map = session.table(SKU_SUPERCEDENCE_MODEL_FAMILY).filter(F.lower(F.col("SKUSTATUS"))=='active')
    ecr_sales = ecr_sales.join(sku_map, ["MODEL", "SKU"], how="inner")
    
    parent_map = session.table("FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH") \
        .select(col("X_DEALER_CODE_HIER").alias("DEALER_CODE"), col("PAR_ORG_NAME")).distinct() 

    ecr_sales = ecr_sales.join(parent_map, "DEALER_CODE", how="left") 
    ecr_sales = ecr_sales.with_column("PARENT_DEALER_CODE", split_part(col("PAR_ORG_NAME"), F.lit("-"), F.lit(1))) 

    ecr_sales = ecr_sales.distinct()
    
    if agg_type == "monthly":
        final_agg = ecr_sales.group_by(
            "PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE", 
            F.year("CAL_DATE").alias("CAL_YEAR"), 
            F.month("CAL_DATE").alias("CAL_MONTH")
        ).agg(F.sum("NET_SALES").alias("NET_SALES")) 
        
        final_agg = final_agg.with_column("DATE", F.date_from_parts(col("CAL_YEAR"), col("CAL_MONTH"), F.lit(1))) 

        join_with_ecr_sales = final_agg.join(ecr_sales.select("SKU","MODEL_FAMILY_CODE","SKUSTATUS"),on=["MODEL_FAMILY_CODE"],how='left')
        join_with_ecr_sales = join_with_ecr_sales.filter(F.lower(F.col("SKUSTATUS"))=="active")
    return final_agg,join_with_ecr_sales


final_agg,join_with_ecr_sales = process_ecr_aggregation_snowpark(session,"monthly",CUSTOMER_TYPE_TO_CONSIDER,run_date, name_of_models, SKU_SUPERCEDENCE_MODEL_FAMILY)

