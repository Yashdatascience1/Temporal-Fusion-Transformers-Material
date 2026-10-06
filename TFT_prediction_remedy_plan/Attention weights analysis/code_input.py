   import torch, pandas as pd

   st = pd.concat(ctx.val_statics, ignore_index=True)     # raw static values, one row per series
   emb = [(n, m.num_embeddings) for n, m in ctx.model.model.named_modules()
          if isinstance(m, torch.nn.Embedding)]
   rows = [r for _, r in emb]
   print(pd.DataFrame({
       "column": st.columns,
       "categories_in_data": st.nunique().values,
       "embedding_rows": rows[:len(st.columns)] + [None] * (len(st.columns) - len(rows)),
   }))

     import glob, pandas as pd
  root = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Iterations in September"
  for f in glob.glob(root + r"\**\series_cache\static_covariates.parquet", recursive=True):
      d = pd.read_parquet(f)
      print(f, "| series:", len(d))
      print(d.astype(str).nunique().to_dict(), "\n")