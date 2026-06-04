import pandas as pd
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.dataprocessing.transformers import StaticCovariatesTransformer
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
import os

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

#Only including the series for which at least 15 months of data is available
#Because input_chunk_length (12) + output_chunk_length (3) is 15
pandas_df = pd.read_csv(r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Input_data_preparation\Preparing the input data\Filtered_data_for_training.csv",index_col = ['MONTH_OF_SALE'],parse_dates=True)

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

pandas_df=pandas_df.reset_index().sort_values(by=["PARENT_DEALER_CODE_MODEL_FAMILY","MONTH_OF_SALE"]).set_index("MONTH_OF_SALE")

pandas_df_with_target_and_static_covariates = pandas_df.loc[:,['PARENT_DEALER_CODE_MODEL_FAMILY','NET_SALES']+static_covariates]


pandas_df_with_future_covariates = pandas_df.loc[:,future_covariates]


#Step 4 - Creating the darts timeseries object for target and static covariates
darts_df_with_static_covariates = TimeSeries.from_group_dataframe(df=pandas_df_with_target_and_static_covariates,
                                                                  group_cols=["PARENT_DEALER_CODE_MODEL_FAMILY"],
                                                                  static_cols=static_covariates,value_cols=["NET_SALES"],freq='MS')


#Step 5 - Creating the darts timeseries object with future covariates

#Removing PARENT_DEALER_CODE_MODEL_FAMILY from future_covariates
try:
    future_covariates.remove('PARENT_DEALER_CODE_MODEL_FAMILY')
except:
    pass

darts_df_with_future_covariates = TimeSeries.from_group_dataframe(df = pandas_df_with_future_covariates,
                                    group_cols="PARENT_DEALER_CODE_MODEL_FAMILY",
                                    freq = 'MS',
                                    value_cols = future_covariates
                                    )

train_list = []
val_list = []

for ts in darts_df_with_static_covariates:
    train = ts.slice(pd.Timestamp('2023-04-01'), pd.Timestamp('2025-12-01'))
    val = ts.slice(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-03-01'))
    
    train_list.append(train)
    val_list.append(val)

train_future_covariates_list = []
validation_future_covariates_list = []

for ts in darts_df_with_future_covariates:
    train = ts.slice(pd.Timestamp('2023-04-01'), pd.Timestamp('2025-12-01'))
    val = ts.slice(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-03-01'))
    train_future_covariates_list.append(train)
    validation_future_covariates_list.append(val)



target_scaler = Scaler(n_jobs=-1)
future_covariates_scaler = Scaler(n_jobs=-1)

transformer = StaticCovariatesTransformer(n_jobs=-1)

#Scale the target training data
scaled_target_series = target_scaler.fit_transform(train_list)

scaled_target_series_with_static_covariates_training = transformer.fit_transform(scaled_target_series)



# #Scale the static covariates in training data
# scaled_static_covariates_training = transformer.fit_transform(train_list)

# #Scale the future covariates in training data
# # scaled_future_covariates = future_covariates_scaler.fit_transform(darts_df_with_future_covariates)

scaled_future_covariates_training = future_covariates_scaler.fit_transform(train_future_covariates_list)
scaled_future_covariates_validation = future_covariates_scaler.transform(validation_future_covariates_list)


# #Scale the target validation data
scaled_target_series_validation = target_scaler.transform(val_list)
scaled_target_series_with_static_covariates_validation = transformer.transform(scaled_target_series_validation)

# #Scale the static covariates in validation data
# scaled_static_covariates_validation = transformer.transform(val_list)

from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from darts.models import TFTModel

loss_logger=LossLogger()


# =========================
# 2. DEFINE PATHS & MODEL NAME
# =========================
now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

WORK_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model"
MODEL_NAME = f"tft_net_sales_{now}"

MODEL_DIR = os.path.join(WORK_DIR, MODEL_NAME)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Create directories if not present
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("MODEL_DIR:", MODEL_DIR)
print("CHECKPOINT_DIR:", CHECKPOINT_DIR)


# =========================
# 3. CUSTOM CHECKPOINT CLASS
# =========================
class DateStampedCheckpoint(ModelCheckpoint):

    @property
    def state_key(self) -> str:
        return f"DateStampedCheckpoint_{self.monitor}_{self.dirpath}"


checkpoint_callback = DateStampedCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename=f"tft-best-{now}-{{epoch:02d}}-{{val_loss:.4f}}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    verbose=True
)


# =========================
# 4. EARLY STOPPING
# =========================
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)


# =========================
# 5. MODEL DEFINITION
# =========================
model = TFTModel(
    input_chunk_length=12,
    output_chunk_length=3,
    batch_size=512,
    dropout=0.1,
    likelihood=None,
    loss_fn=torch.nn.MSELoss(),
    n_epochs=100,
    random_state=42,
    add_encoders=add_encoders,

    model_name=MODEL_NAME,
    work_dir=WORK_DIR,

    # IMPORTANT: avoid conflict with Darts internal checkpointing
    save_checkpoints=False,
    force_reset=True,

    pl_trainer_kwargs={
        "callbacks": [
            loss_logger,
            checkpoint_callback,
            early_stop_callback
        ],
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1
    }
)


# =========================
# 6. LR FINDER
# =========================
print("\nRunning LR Finder...")

lr_finder = model.lr_find(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training,
)

suggested_lr = lr_finder.suggestion()
print("Suggested Learning Rate:", suggested_lr)

# Apply LR
model.lr = suggested_lr

from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

current_date = datetime.now().strftime("%Y-%m-%d")

now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

WORK_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\Multi_series\12_month_forecast\Pipeline\Model"
MODEL_NAME = f"tft_net_sales_{now}"

# =========================
# 7. TRAINING WITH VALIDATION
# =========================
print("\nStarting Training...")

model.fit(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training,
    val_series=scaled_target_series_with_static_covariates_validation,
    val_future_covariates=scaled_future_covariates_validation,
    verbose=True
)


import torch
from darts.models import TFTModel

# 1. Get the path to your best custom checkpoint file
best_ckpt_path = checkpoint_callback.best_model_path
print("Loading directly from raw PyTorch Lightning checkpoint:", best_ckpt_path)

# 2. Re-instantiate a fresh template model with IDENTICAL architecture
# Crucial: Override n_epochs to 0 here so the upcoming fit step is instantaneous
loaded_model = TFTModel(
    input_chunk_length=12,
    output_chunk_length=3,
    batch_size=512,
    dropout=0.1,
    likelihood=None,
    loss_fn=torch.nn.MSELoss(),
    n_epochs=0, # Set to 0 to prevent training
    random_state=42,
    add_encoders=add_encoders,
    model_name=MODEL_NAME,
    work_dir=WORK_DIR,
    save_checkpoints=False,
    force_reset=True
)

# 3. Force public initialization via an instant 0-epoch fit
print("Initializing model structure natively...")
loaded_model.fit(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training
)

# 4. Safely load the raw weights from the .ckpt file directly into the structure
checkpoint = torch.load(best_ckpt_path, map_location="cpu")
loaded_model.model.load_state_dict(checkpoint["state_dict"])

# 5. Restore the original parameter just in case you use it for further retraining
loaded_model.n_epochs = 100

print("Success! Model structure initialized and weights successfully injected.")


# Generate the forecast
forecast_series = loaded_model.predict(
    n=3, # Predicting your 3-month output chunk
    series=scaled_target_series_with_static_covariates_training, # Gives the model the past 12 months of history
    future_covariates=scaled_future_covariates_validation # Gives the model the known future features (Festivals, Marriage days, etc.)
)

print("Forecast generated successfully!")

# Inverse-transform predictions back to original scale
pred_series_inv = target_scaler.inverse_transform(forecast_series)
val_inv         = target_scaler.inverse_transform(scaled_target_series_with_static_covariates_validation)

# Build output DataFrame — one row per series per month
records = []

for actual, forecast, original_series in zip(val_inv, pred_series_inv, val_list):

    # Get the series label from unscaled val_list (retains original string value)
    series_name = original_series.static_covariates['PARENT_DEALER_CODE_MODEL_FAMILY'].values[0]

    months          = forecast.time_index
    actual_values   = actual.values().flatten()
    forecast_values = forecast.values().flatten()

    for month, act, pred in zip(months, actual_values, forecast_values):
        records.append({
            'MONTH_OF_SALE'                  : month,
            'PARENT_DEALER_CODE_MODEL_FAMILY' : series_name,
            'ACTUAL_SALES'                   : round(float(act),  2),
            'PREDICTED_SALES'                : round(float(pred), 2)
        })

df_output = pd.DataFrame(records)
df_output['MONTH_OF_SALE'] = pd.to_datetime(df_output['MONTH_OF_SALE']).dt.strftime('%Y-%m-%d')
df_output = df_output.sort_values(['PARENT_DEALER_CODE_MODEL_FAMILY', 'MONTH_OF_SALE']).reset_index(drop=True)

print(f'Output shape : {df_output.shape}')
print(f'Months       : {df_output["MONTH_OF_SALE"].unique()}')
print(f'Series count : {df_output["PARENT_DEALER_CODE_MODEL_FAMILY"].nunique()}')
df_output.head(10)

