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


import torch, pandas as pd

st = pd.concat(ctx.val_statics, ignore_index=True)
emb = [(n, m.num_embeddings) for n, m in ctx.model.model.named_modules()
       if isinstance(m, torch.nn.Embedding)]
print("static columns in data      :", st.shape[1])
print("embedding tables in model   :", len(emb))
print(emb)

k = max(st.shape[1], len(emb))
chk = pd.DataFrame({
    "column":         list(st.columns)         + [None] * (k - st.shape[1]),
    "categories":     list(st.nunique().values) + [None] * (k - st.shape[1]),
    "embedding_rows": [r for _, r in emb]      + [None] * (k - len(emb)),
})
chk["expected_rows"] = chk["categories"] + 1
chk["ok"] = chk["embedding_rows"] == chk["expected_rows"]
print(chk)