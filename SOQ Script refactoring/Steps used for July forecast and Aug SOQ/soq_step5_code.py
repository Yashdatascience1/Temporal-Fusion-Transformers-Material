import snowflake.snowpark as snowpark
from snowflake.snowpark.session import Session  # <-- Changed from get_active_session
import sys 
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

# Append path to your configuration files
sys.path.append(r"C:\Users\G0004878\Desktop\TFT_Data\utils_files")
import Snowflake_configuration
snowflake_conn_prop = Snowflake_configuration.ds1_role_json

RUN_DATE = datetime.datetime.today().strftime('%Y%m%d') 
run_version = 42

SOQ_TABLE = f'''SELECT * FROM MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2 WHERE RUN_DATE='{RUN_DATE}' and run_version = {run_version} '''
DEALER_MAPPING_TABLE = 'ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER'
OUTPUT_TABLE = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_WITH_AO_VERSION_4'

def main(session: snowpark.Session): 
    soq_data = session.sql(SOQ_TABLE).to_pandas()
    is_obd = soq_data['IS_OBD'].tolist()
    soq_data.drop(['IS_OBD'], axis=1, inplace=True)
    
    dealer_data = session.table(DEALER_MAPPING_TABLE).to_pandas()
    dealer_data = dealer_data[['DEALER_CODE', 'AREA_OFFICE', 'ZONE']]
    dealer_data.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']
    
    soq_data = pd.merge(soq_data, dealer_data, on=['PARENT_DEALER_CODE'], how='left')
    soq_data['IS_OBD'] = is_obd
    
    agg_data_sp = session.create_dataframe(soq_data)
    agg_data_sp.write.mode("append").save_as_table(OUTPUT_TABLE)
    return agg_data_sp

# --- LOCAL COMPUTER ENTRY POINT ---
if __name__ == "__main__":
    print("Connecting to Snowflake...")
    try:
        # Create the local session using your credentials file
        session = Session.builder.configs(snowflake_conn_prop).create()
        session.use_database('MOP_DATABASE')
        session.use_schema('SOQ')
        print("Session established successfully. Running main process...")
        
        # Execute the processing function
        result = main(session)
        print(f"Data successfully appended to {OUTPUT_TABLE}!")
        
        # In Snowpark, printing a DataFrame object locally just shows a reference string.
        # Use .show() to print the rows out in your local console window:
        print("\nPreview of processed data:")
        result.limit(5).show()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Close connection cleanly
        if 'session' in locals():
            session.close()
            print("Snowflake connection closed.")