idx2 = stratified_indices(labels, n_per_group=100, seed=1)
out2 = extract_tft_interpretation(ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
        indices=idx2, group_labels=labels, target_names=[ctx.target_col],
        future_names=ctx.future_covariates, static_names=ctx.static_covariates)
a = out["ALL"]["decoder_importance"]; b = out2["ALL"]["decoder_importance"].reindex(a.index)
print(pd.concat([a.rename("sample0"), b.rename("sample1")], axis=1).head(10))
print("rank correlation:", a.rank().corr(b.rank()))

fam = "XOOM"
idx_f = [i for i in idx if labels[i] == fam]
solo = extract_tft_interpretation(ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
        indices=idx_f, group_labels=labels, target_names=[ctx.target_col],
        future_names=ctx.future_covariates, static_names=ctx.static_covariates)
s = solo["ALL"]["decoder_importance"]
print("max abs difference:", (s - out[fam]["decoder_importance"].reindex(s.index)).abs().max())