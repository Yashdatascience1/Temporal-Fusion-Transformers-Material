model = TFTModel.load_from_checkpoint(MODEL_NAME, best=True,WORK_DIR = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Iterations in October\_2_Modelling")
print(f"Loaded best checkpoint: {MODEL_NAME}")

ValueError: Could not find base model save file `_model.pth.tar` in c:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Iterations in October\_3_Prediction\darts_logs\daily_tft_negbin_emb_2026-10-09_11_37_06.