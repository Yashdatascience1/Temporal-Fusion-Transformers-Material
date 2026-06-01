prediction_df = pandas_df.loc['2025-02-01':'2026-07-01']

# Function to encode the year as a normalized value
def encode_year(idx):
  return (idx.year - 2000) / 50

def encode_days_in_month(index):
  return index.days_in_month.to_numpy().reshape(-1,1)

# Set up the add_encoders dictionary to specify how different time-related encoders and transformers should be applied
add_encoders = {
    'cyclic': {'past': ['month'], 'future': ['month']},
    'position': {'past': ['relative'], 'future': ['relative']},
    'custom': {
        'past': [encode_year, encode_days_in_month],
        'future': [encode_year, encode_days_in_month]
    },
    'transformer': Scaler()
}

#Extracting type of columns according to the datatypes
# 1. Targets/Metrics (The numbers we want to predict)
target_cols = pandas_df.select_dtypes(include=['number']).columns.tolist()
target_cols.append('PARENT_DEALER_CODE_MODEL_FAMILY')

# 2. Time Dimension
time_cols = pandas_df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

# 3. Static/Categorical Covariates (The identifiers)
# We exclude numbers and dates to find the "ID" strings
static_cols = pandas_df.select_dtypes(exclude=['number', 'datetime', 'datetime64']).columns.tolist()

print(f"Targets: {target_cols}")
print(f"Time Column: {time_cols}")
print(f"Static Identifiers: {static_cols}")


#Separating the covariates
target_col = ["NET_SALES"]

#future covariates
future_covariates = [i for i in target_cols if i!='NET_SALES']

#actual_static_cols
actual_static_cols = [i for i in static_cols if i!='PARENT_DEALER_CODE_MODEL_FAMILY']


static_covariates = actual_static_cols.copy()

static_covariates

target_plus_static_cols = target_col + static_cols


try:
    future_covariates.remove('PARENT_DEALER_CODE_MODEL_FAMILY')
except:
    pass

static_plus_future_cov = static_cols + future_covariates
static_plus_future_cov

#Step 1 - Preparing the lookback data for the model
lookback_data_pandas_df = prediction_df.loc['2025-02-01':'2026-04-01',target_plus_static_cols]

#Step 2 - Preparing the lookahead data for the model
lookahead_data_pandas_df = prediction_df.loc['2025-02-01':'2026-07-01',static_plus_future_cov]

#Step 3 - Creating the darts time-series object from lookback data for the model
lookback_data_darts_df = TimeSeries.from_group_dataframe(df=lookback_data_pandas_df,
                                                                  group_cols=["PARENT_DEALER_CODE_MODEL_FAMILY"],
                                                                  static_cols=static_covariates,value_cols=["NET_SALES"],freq='MS')

#Step 4 - Creating the darts time-series object from lookahead data for the model 
lookahead_data_darts_df = TimeSeries.from_group_dataframe(
    df=lookahead_data_pandas_df,
    group_cols="PARENT_DEALER_CODE_MODEL_FAMILY",
    static_cols=static_covariates, 
    value_cols=future_covariates,
    freq='MS'
)


future_covariates_scaler = joblib.load(r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model\scaled_objects_pickled_version\future_covariates_scaler.pkl")

transformer = joblib.load(r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model\scaled_objects_pickled_version\static_transformer.pkl")

target_scaler = joblib.load(r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model\scaled_objects_pickled_version\target_scaler.pkl")


scaled_temporal = future_covariates_scaler.transform(lookahead_data_darts_df)

final_scaled_lookahead_data = transformer.transform(scaled_temporal)

target_scaled_data = target_scaler.transform(lookback_data_darts_df)

final_scaled_lookback_data = transformer.transform(target_scaled_data)

from darts.models import TFTModel
checkpoint_path = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model\tft_net_sales_2026-05-18_02_18_29\checkpoints\tft-best-2026-05-18_02_18_29-epoch=09-val_loss=0.0396.ckpt"

WORK_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model"
MODEL_NAME = f"tft_net_sales_2026-05-18_02_18_29"

loaded_model = TFTModel(
    input_chunk_length=12,
    output_chunk_length=3,
    batch_size=512,
    dropout=0.1,
    likelihood=None,
    loss_fn=torch.nn.MSELoss(),
    n_epochs=0, 
    random_state=42,
    add_encoders=add_encoders,
    model_name=MODEL_NAME,
    work_dir=WORK_DIR,
    save_checkpoints=False,
    force_reset=True
)

loaded_model.fit(
    series=final_scaled_lookback_data, 
    future_covariates=final_scaled_lookahead_data,
    epochs=1)


from darts.models.forecasting.tft_model import _TFTModule
loaded_model.model = _TFTModule.load_from_checkpoint(checkpoint_path)

forecast_series = loaded_model.predict(
    n=3, 
    series=final_scaled_lookback_data, 
    future_covariates=final_scaled_lookahead_data
)

print("Forecast generated successfully!")

# Build output DataFrame for Apr'26 to Jun'26 predictions
records = []

for forecast, source_series in zip(actual_pred_series, lookahead_data_darts_df):
    # val_list retains original static covariates — use it as the label source
    series_name = source_series.static_covariates['PARENT_DEALER_CODE_MODEL_FAMILY'].values[0]

    months          = forecast.time_index
    forecast_values = forecast.values().flatten()

    for month, pred in zip(months, forecast_values):
        records.append({
            'MONTH_OF_SALE'                   : month,
            'PARENT_DEALER_CODE_MODEL_FAMILY'  : series_name,
            'PREDICTED_SALES'                  : round(float(pred), 2)
        })

df_final_output = pd.DataFrame(records)
df_final_output['MONTH_OF_SALE'] = pd.to_datetime(df_final_output['MONTH_OF_SALE']).dt.strftime('%Y-%m-%d')
df_final_output = df_final_output.sort_values(['PARENT_DEALER_CODE_MODEL_FAMILY', 'MONTH_OF_SALE']).reset_index(drop=True)

print(f'Output shape : {df_final_output.shape}')
print(f'Months       : {df_final_output["MONTH_OF_SALE"].unique()}')
print(f'Series count : {df_final_output["PARENT_DEALER_CODE_MODEL_FAMILY"].nunique()}')
df_final_output.head(10)

