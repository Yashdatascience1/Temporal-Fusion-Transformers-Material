# =============================================================================
# TFT TRAINING WITH CUSTOM FESTIVE-WEIGHTED LOSS
# Train : 2023-04-01 → 2025-12-31
# Val   : 2026-01-01 → 2026-06-30  (with 122-day warm-up glued in front)
# Test  : predict next 180 days after 2026-06-30
# =============================================================================

import os, json, glob, gc
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping

# =============================================================================
# SECTION 1: CONFIG — must match the chunking pipeline notebook
# =============================================================================

local_train_dir = "./local_train_data"   # use absolute path if notebooks differ in cwd
local_test_dir  = "./local_test_data"
group_keys_path = "./saved_group_keys/group_keys.json"

time_col   = 'CAL_DATE'
group_col  = 'PARENT_DEALER_CODE_MODEL_FAMILY'
target_col = 'NET_SALES'
FREQ       = 'D'

TRAIN_END = pd.Timestamp("2025-12-31")
VAL_START = pd.Timestamp("2026-01-01")
VAL_END   = pd.Timestamp("2026-06-30")

INPUT_CHUNK_LENGTH  = 365
OUTPUT_CHUNK_LENGTH = 184   # Jul 1 → Dec 31 2026 inclusive = 184 days
TEST_HORIZON        = 184

static_covariates = [
    'PARENT_DEALER_CODE', 'MODEL_FAMILY', 'MODEL_NAME', 'BRAKE_TYPE',
    'IGNITION_TYPE', 'WHEEL_TYPE', 'COLOUR', 'DEALER_CITY',
    'X_CITY_CATEGORY', 'ZONAL_OFFICE_NAME'
]

future_covariates = [
    'NEW_YEAR','LOHRI','MAKAR_SANKRANTI','REPUBLIC_DAY','VASANT_PANCHAMI',
    'MAHA_SHIVRATRI','EID_UL_FITR','HOLIKA_DAHAN','HOLI','HANUMAN_JAYANTI',
    'AKSHAYA_TRITYA','BUDDHA_PURNIMA','GANGA_DUSSEHRA','JAGANNATH_RATHYATRA',
    'GURU_PURNIMA','NAG_PANCHAMI','RAKSHA_BANDHAN','HARTALIK_TEEJ',
    'GANESH_CHATURTHI','JANMASHTAMI','VISHWAKARMA_PUJA','KARWA_CHAUTH',
    'ONAM','MARRIAGE_DAY',
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

# Columns whose non-zero value triggers the extra MAPE penalty
penalty_cols = [
    'N-16','N-15','N-14','N-13','N-12','N-11','N-10','N-9','N-8','N-7',
    'N-6','N-5','N-4','N-3','N-2','N-1','N','N+1','N+2','N+3','N+4',
    'N+5','N+6','N+7','N+8','N+9','N+10',
    'D-3','D-2','D-1','D','D+1','D+2','D+3','D+4','D+5','D+6',
    'C','C+1','C+2','C+3','C+4','C+5','C+6'
]

# =============================================================================
# SECTION 2: CUSTOM LOSS — adapted for Darts
# =============================================================================
# Darts calls: criterion(output, target) — only two arguments, no covariates.
# Workaround: the target series carries TWO components:
#   component 0 = NET_SALES (scaled)
#   component 1 = FESTIVE_FLAG (binary, unscaled)
# The loss splits them. The flag's own prediction error is excluded from the
# loss entirely — the model still outputs 2 components, but component 1's
# predictions are ignored (and discarded at inference).

class HuberMaeFeatureLoss(nn.HuberLoss):
    """
    Total loss = Huber(y_hat, y) + flag * MAE(y_hat, y)
    computed only on component 0. Component 1 of the target is the flag.
    On festive days (flag=1) the absolute error is counted twice in effect:
    once inside Huber, once as the added MAE penalty.
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__(reduction='none', delta=delta)
        self.user_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Shapes: (batch, timesteps, n_components)
        # Component 0 = NET_SALES, component 1 = FESTIVE_FLAG
        y_hat = input[..., 0]
        y     = target[..., 0]
        flag  = target[..., 1]

        huber_base = super().forward(y_hat, y)

        mae = torch.abs(y - y_hat)

        total_loss = huber_base + flag * mae

        if self.user_reduction == 'mean':
            return total_loss.mean()
        elif self.user_reduction == 'sum':
            return total_loss.sum()
        return total_loss

print("="*60)
print("SECTION 3: LOADING DATA & BUILDING SERIES")
print("="*60)

needed_cols = [time_col, group_col, target_col] + static_covariates + penalty_cols

train_targets_stacked = []
val_targets_stacked   = []
series_keys           = []

chunk_files = sorted(glob.glob(os.path.join(local_train_dir, "chunk_*.parquet")))
test_chunk_files = sorted(glob.glob(os.path.join(local_test_dir, "chunk_*.parquet")))
print(f"Train chunk files: {len(chunk_files)} | Test chunk files: {len(test_chunk_files)}")

# ---------------------------------------------------------------------------
# STEP 3A: Build ONE shared covariate series (covariates are date-only —
# identical across all series, so one TimeSeries serves all 117K).
# ---------------------------------------------------------------------------
print("Building shared covariate calendar...")

cov_cols = [time_col] + future_covariates

# One chunk is enough to get all dates in the train range
cal_train = pd.read_parquet(chunk_files[0], columns=cov_cols)
cal_train[time_col] = pd.to_datetime(cal_train[time_col])
cal_train = cal_train.drop_duplicates(subset=time_col)

cal_test = pd.read_parquet(test_chunk_files[0], columns=cov_cols)
cal_test[time_col] = pd.to_datetime(cal_test[time_col])
cal_test = cal_test.drop_duplicates(subset=time_col)

calendar_df = (
    pd.concat([cal_train, cal_test])
    .drop_duplicates(subset=time_col)
    .sort_values(time_col)
    .reset_index(drop=True)
)
del cal_train, cal_test
gc.collect()

print(f"Calendar range: {calendar_df[time_col].min()} → {calendar_df[time_col].max()} "
      f"({len(calendar_df)} days)")

shared_future_cov = TimeSeries.from_dataframe(
    calendar_df, time_col=time_col, value_cols=future_covariates,
    freq=FREQ, fill_missing_dates=False
).astype(np.float32)

# ---------------------------------------------------------------------------
# STEP 3B: Build target series per group (only target + flag + statics — small)
# ---------------------------------------------------------------------------
val_window_days = (VAL_END - VAL_START).days + 1     # 181
warmup_days = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH - val_window_days
warmup_start = VAL_START - pd.Timedelta(days=warmup_days)

for ci, chunk_path in enumerate(chunk_files):
    df_chunk = pd.read_parquet(chunk_path, columns=needed_cols)
    df_chunk[time_col] = pd.to_datetime(df_chunk[time_col])

    for key, g in df_chunk.groupby(group_col):
        g = g.sort_values(time_col).reset_index(drop=True)

        g["FESTIVE_FLAG"] = (g[penalty_cols] != 0).any(axis=1).astype(np.float32)

        train_mask = g[time_col] <= TRAIN_END
        val_mask   = (g[time_col] >= warmup_start) & (g[time_col] <= VAL_END)

        g_train = g[train_mask]
        g_val   = g[val_mask]

        if len(g_train) < INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH:
            continue
        if len(g_val) < INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH:
            g_val = None

        static_df = g[static_covariates].iloc[[0]].reset_index(drop=True)

        ts_train = TimeSeries.from_dataframe(
            g_train, time_col=time_col,
            value_cols=[target_col, "FESTIVE_FLAG"],
            static_covariates=static_df,
            freq=FREQ, fill_missing_dates=False
        ).astype(np.float32)
        train_targets_stacked.append(ts_train)

        if g_val is not None:
            ts_val = TimeSeries.from_dataframe(
                g_val, time_col=time_col,
                value_cols=[target_col, "FESTIVE_FLAG"],
                static_covariates=static_df,
                freq=FREQ, fill_missing_dates=False
            ).astype(np.float32)
        else:
            ts_val = None
        val_targets_stacked.append(ts_val)

        series_keys.append(key)

    del df_chunk
    gc.collect()
    print(f"Chunk {ci+1}/{len(chunk_files)} processed. Series so far: {len(series_keys)}")

# Covariate lists: same shared object referenced N times — negligible RAM
train_future_covs = [shared_future_cov] * len(train_targets_stacked)
full_future_covs  = [shared_future_cov] * len(train_targets_stacked)

print(f"\nTotal series built : {len(train_targets_stacked)}")
n_with_val = sum(v is not None for v in val_targets_stacked)
print(f"Series with valid val strip: {n_with_val}")

# =============================================================================
# SECTION 4: SCALING
# =============================================================================
# Scale ONLY NET_SALES (component 0). The flag must stay binary — scaling it
# would corrupt the loss mask. Strategy: scale the 1-component NET_SALES
# series per-series, then stack the unscaled flag back on.

print("\n" + "="*60)
print("SECTION 4: SCALING")
print("="*60)

train_sales = [ts.univariate_component(0) for ts in train_targets_stacked]
train_flags = [ts.univariate_component(1) for ts in train_targets_stacked]

target_scaler = Scaler()  # per-series min-max (fits each series independently)
train_sales_scaled = target_scaler.fit_transform(train_sales)

# Re-stack: scaled sales + raw flag (preserve static covariates from original)
scaled_train_targets = []
for s, f, orig in zip(train_sales_scaled, train_flags, train_targets_stacked):
    stacked = s.stack(f)
    stacked = stacked.with_static_covariates(orig.static_covariates)
    scaled_train_targets.append(stacked)

# Validation targets — transform with the SAME fitted scalers.
# Darts' per-series Scaler requires transform() to receive the same-length
# list in the same order as fit(). Series without a val strip get their train
# series as a placeholder; those slots are discarded afterwards.
val_sales_aligned = [
    (val_targets_stacked[i].univariate_component(0)
     if val_targets_stacked[i] is not None
     else train_sales[i])
    for i in range(len(train_sales))
]
val_sales_scaled_aligned = target_scaler.transform(val_sales_aligned)

scaled_val_targets = []
for i in range(len(train_sales)):
    if val_targets_stacked[i] is None:
        scaled_val_targets.append(None)
        continue
    v_flag = val_targets_stacked[i].univariate_component(1)
    stacked = val_sales_scaled_aligned[i].stack(v_flag)
    stacked = stacked.with_static_covariates(train_targets_stacked[i].static_covariates)
    scaled_val_targets.append(stacked)

# Future covariates: festival flags are already 0/1 ramps — leave unscaled.

# Static covariates are strings → encode
static_transformer = StaticCovariatesTransformer()
scaled_train_targets = static_transformer.fit_transform(scaled_train_targets)

# Apply same static transform to val (only non-None)
_val_present = [ts for ts in scaled_val_targets if ts is not None]
_val_present_transformed = static_transformer.transform(_val_present)
_it = iter(_val_present_transformed)
scaled_val_targets = [next(_it) if ts is not None else None for ts in scaled_val_targets]

# Build fit() val lists — only series that have a val strip, with matching covariates
fit_val_series = []
fit_val_covs   = []
for i in range(len(scaled_train_targets)):
    if scaled_val_targets[i] is not None:
        fit_val_series.append(scaled_val_targets[i])
        fit_val_covs.append(full_future_covs[i])

print(f"Scaled train series : {len(scaled_train_targets)}")
print(f"Val series for fit  : {len(fit_val_series)}")

# =============================================================================
# SECTION 5: MODEL — GPU setup as in iteration3
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: MODEL")
print("="*60)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_float32_matmul_precision('high')

now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
MODEL_NAME = f"daily_tft_festive_loss_{now}"
WORK_DIR = os.getcwd()
print("Model name:", MODEL_NAME)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="min",
)

model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=4,
    num_attention_heads=16,
    dropout=0.05,

    batch_size=256,
    n_epochs=100,

    likelihood=None,                      # deterministic → loss_fn is used
    loss_fn=HuberMaeFeatureLoss(delta=1.0, reduction='mean'),

    random_state=42,
    add_relative_index=True,

    save_checkpoints=True,                # ← best + last checkpoints saved automatically
    force_reset=True,
    model_name=MODEL_NAME,

    skip_interpolation=True,

    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
        "precision": "bf16-mixed",
    }
)

# =============================================================================
# SECTION 6: TRAIN
# =============================================================================

print("\n" + "="*60)
print("SECTION 6: TRAINING")
print("="*60)

model.fit(
    series=scaled_train_targets,
    future_covariates=train_future_covs,

    val_series=fit_val_series,
    val_future_covariates=fit_val_covs,

    dataloader_kwargs={
        "num_workers": 4,
        "pin_memory": True,
    },
    verbose=True
)

print("\nTraining complete.")

# =============================================================================
# SECTION 7: LOAD BEST CHECKPOINT & PREDICT 180 DAYS
# =============================================================================

print("\n" + "="*60)
print("SECTION 7: BEST CHECKPOINT + TEST PREDICTION")
print("="*60)

# Best checkpoint = lowest val_loss (saved automatically by save_checkpoints=True)
best_model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)
print("Best checkpoint loaded.")

# Prediction input: full history up to VAL_END (train + val actuals, scaled)
predict_input = []
predict_covs  = []
for i in range(len(scaled_train_targets)):
    if scaled_val_targets[i] is not None:
        # glue train + val (val includes warm-up overlap — slice it off first)
        val_only = scaled_val_targets[i].drop_before(TRAIN_END)  # strictly after TRAIN_END
        full_hist = scaled_train_targets[i].append(val_only)
    else:
        full_hist = scaled_train_targets[i]
    predict_input.append(full_hist)
    predict_covs.append(full_future_covs[i])

preds_scaled = best_model.predict(
    n=TEST_HORIZON,
    series=predict_input,
    future_covariates=predict_covs,
    verbose=True
)

# Inverse-transform: predictions have 2 components — keep only NET_SALES (comp 0)
pred_sales_scaled = [p.univariate_component(0) for p in preds_scaled]
pred_sales = target_scaler.inverse_transform(pred_sales_scaled)

# Clip negatives
pred_sales = [p.map(lambda x: np.clip(x, 0, None)) for p in pred_sales]

print(f"\nPredicted {len(pred_sales)} series × {TEST_HORIZON} days.")
print(f"First prediction window: {pred_sales[0].start_time()} → {pred_sales[0].end_time()}")

# Save predictions to parquet
pred_rows = []
for key, p in zip(series_keys, pred_sales):
    pdf = p.to_dataframe().reset_index()
    pdf.columns = [time_col, 'PREDICTED_NET_SALES']
    pdf[group_col] = key
    pred_rows.append(pdf)

pred_df = pd.concat(pred_rows, ignore_index=True)
pred_df.to_parquet("./predictions_180d.parquet", index=False)
print("Predictions saved → ./predictions_180d.parquet")
