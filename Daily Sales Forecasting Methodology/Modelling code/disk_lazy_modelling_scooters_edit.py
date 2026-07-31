model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,

    hidden_size=32,
    lstm_layers=4,
    num_attention_heads=4,          # was 16 -> 4x less attention memory, and
                                    # 32/4 = 8 dims per head instead of 2
    dropout=0.05,

    batch_size=64,                  # was 256 -> 4x less memory per step
    n_epochs=100,

    likelihood=None,
    loss_fn=HuberMaeFeatureLoss(delta=1.0, reduction='mean'),

    random_state=42,
    add_relative_index=True,

    save_checkpoints=True,
    force_reset=True,
    model_name=MODEL_NAME,
    skip_interpolation=True,

    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [early_stopping],
        "gradient_clip_val": 0.1,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 4,   # 64 x 4 = effective batch 256
    },
)

import gc, torch

# free anything left on the card by a previous failed fit
for name in ("trainer", "model"):
    pass  # keep `model` — only clear stale CUDA allocations
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

free, total = torch.cuda.mem_get_info()
print(f"GPU free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")
if free / total < 0.8:
    print("WARNING: less than 80% of the card is free -- restart the kernel before training.")


model.fit(
    series=train_seq,
    future_covariates=train_cov_seq,
    val_series=val_seq,
    val_future_covariates=val_cov_seq,
    max_samples_per_ts=400,          # REINSTATED
    dataloader_kwargs={
        "num_workers": 0,
        "pin_memory": True,
    },
    verbose=True,
)

