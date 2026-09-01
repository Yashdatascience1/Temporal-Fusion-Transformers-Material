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

