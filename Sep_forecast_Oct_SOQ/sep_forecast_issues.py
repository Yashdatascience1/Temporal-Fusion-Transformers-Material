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


                count      mean       std  min  25%       50%       75%  max
festive_month                                                               
2023-11-01     5000.0  0.425770  0.378868  0.0  0.0  0.357143  0.777778  1.0
2024-11-01     5000.0  0.253393  0.260057  0.0  0.0  0.200000  0.400000  1.0
2025-10-01     5000.0  0.451615  0.390078  0.0  0.0  0.400000  0.869841  1.0

Share of series where the festive month IS the historical max:
festive_month
2023-11-01    0.1672
2024-11-01    0.0228
2025-10-01    0.1884
Name: is_series_max, dtype: float64


# ===========================
                count      mean       std  min       25%       50%       75%  \
festive_month                                                                  
2023-11-01     5000.0  0.425770  0.378868  0.0  0.000000  0.357143  0.777778   
2024-10-01     5000.0  0.482513  0.365082  0.0  0.142857  0.454545  0.827167   
2025-10-01     5000.0  0.451615  0.390078  0.0  0.000000  0.400000  0.869841   

               max  
festive_month       
2023-11-01     1.0  
2024-10-01     1.0  
2025-10-01     1.0  

Share of series where the festive month IS the historical max:
festive_month
2023-11-01    0.1672
2024-10-01    0.1636
2025-10-01    0.1884
Name: is_series_max, dtype: float64


#################################
def on_train_epoch_end(self, trainer, pl_module):
    print(f"epoch {trainer.current_epoch}: "
          f"allocated {torch.cuda.memory_allocated()/1e9:.2f}GB, "
          f"reserved {torch.cuda.memory_reserved()/1e9:.2f}GB")
    train_loss = trainer.callback_metrics.get("train_loss")
    if train_loss is not None:
        self.train_losses.append(float(train_loss.detach().cpu()))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


