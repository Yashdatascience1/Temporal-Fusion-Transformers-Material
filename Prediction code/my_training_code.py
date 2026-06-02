loss_logger = LossLogger()

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
    
    # CRITICAL CHANGE: Tell Darts to handle its native model manifest building
    save_checkpoints=True,          
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

print("\nRunning LR Finder...")
lr_finder = model.lr_find(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training,
)

suggested_lr = lr_finder.suggestion()
print("Suggested Learning Rate:", suggested_lr)
model.lr = suggested_lr

print("\nStarting Training with Validation...")
model.fit(
    series=scaled_target_series_with_static_covariates_training,
    future_covariates=scaled_future_covariates_training,
    val_series=scaled_target_series_with_static_covariates_validation,
    val_future_covariates=scaled_future_covariates_validation,
    verbose=True
)

print(f"\n✅ Training Complete. Best model saved at:\n--> {os.path.join(CHECKPOINT_DIR, 'best_model.ckpt')}")

