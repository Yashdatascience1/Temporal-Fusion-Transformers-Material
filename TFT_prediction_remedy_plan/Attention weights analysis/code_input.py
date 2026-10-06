import torch, pandas as pd
st = pd.concat(ctx.val_statics, ignore_index=True)
emb = [m.num_embeddings for m in ctx.model.model.modules() if isinstance(m, torch.nn.Embedding)]
chk = pd.DataFrame({"column": st.columns, "categories": st.nunique().values, "embedding_rows": emb[:len(st.columns)]})
chk["ok"] = chk["embedding_rows"] == chk["categories"] + 1
print(chk)

cov = ctx.shared_cov.pd_dataframe().loc["2026-09-01":"2026-12-07"]
for kind, tbl in [("decoder", out["ALL"]["decoder_time"])]:
    for v in ["D-2", "DOW_SIN", "N+3", "D", "IS_MONTH_START"]:
        active = (cov[v] != 0).to_numpy()
        w = tbl[v].to_numpy()
        print(f"{v:15s} active: {w[active].mean():.3f}   zero: {w[~active].mean():.3f}")