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