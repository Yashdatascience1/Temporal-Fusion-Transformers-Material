print("\nStarting Training with Validation...")
model.fit(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training,
    val_series=scaled_target_series_with_static_covariates_validation,
    val_future_covariates=scaled_future_covariates_validation,
    dataloader_kwargs={
        "num_workers": 4,         # Parallellize data processing on GPU
        "pin_memory": True        # Fast page-locked VRAM transfers
    },
    verbose=True
)

print(f"\n✅ Training Complete. Best model saved at:\n--> {os.path.join(CHECKPOINT_DIR, 'best_model.ckpt')}")

loss_logger = LossLogger()

if torch.cuda.is_bf16_supported():
    print("Awesome! bf16 is supported. Using bf16-mixed.")
    precision_setting = "bf16-mixed"
else:
    print("Warning: bf16 is not supported on this GPU. Falling back to 16-mixed.")
    precision_setting = "16-mixed"

model = TFTModel(
    input_chunk_length=16,
    output_chunk_length=2,
    batch_size=256,
    dropout=0.1,
    likelihood=None,
    loss_fn=torch.nn.HuberLoss(delta=1.0),
    n_epochs=100,
    random_state=42,
    add_encoders=add_encoders,
    model_name=MODEL_NAME,
    work_dir=WORK_DIR,
    use_reversible_instance_norm=True,
    
    # CRITICAL CHANGE: Tell Darts to handle its native model manifest building
    save_checkpoints=True,          
    force_reset=True,
    
    pl_trainer_kwargs={
        "callbacks": [
            loss_logger,
            early_stop_callback
        ],
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "accelerator": "gpu", 
        "devices": [0],
        "precision": precision_setting
    }
)

print("\nRunning LR Finder...")
lr_finder = model.lr_find(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training
)

suggested_lr = lr_finder.suggestion()
print("Suggested Learning Rate:", suggested_lr)
model.lr = suggested_lr

import numpy as np
import pandas as pd

# Festive peak months inside your training window (Apr'23–Dec'25)
festive_months = [pd.Timestamp('2023-11-01'),   # Diwali Nov 12, 2023
                  pd.Timestamp('2024-11-01'),   # Diwali Nov 1, 2024
                  pd.Timestamp('2025-10-01')]   # Diwali Oct 20, 2025

rows = []
for ts in scaled_target_series_with_static_covariates_training[:5000]:
    s = ts.pd_series()
    if s.max() == 0:
        continue
    argmax_month = s.idxmax()
    for fm in festive_months:
        if fm in s.index:
            rows.append({
                'festive_month': fm,
                'scaled_value': s.loc[fm],
                'is_series_max': argmax_month == fm
            })

df = pd.DataFrame(rows)
print(df.groupby('festive_month')['scaled_value'].describe())
print("\nShare of series where the festive month IS the historical max:")
print(df.groupby('festive_month')['is_series_max'].mean())