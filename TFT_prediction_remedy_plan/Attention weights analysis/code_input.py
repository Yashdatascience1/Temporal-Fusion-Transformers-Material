from tft_interpret import *

labels = ctx.labels()                                    # one MODEL_FAMILY label per val series
print(pd.Series(labels).value_counts())                  # group sizes: spot tiny groups

idx = stratified_indices(labels, n_per_group=10)         # small first
out = extract_tft_interpretation(
    ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
    indices=idx, group_labels=labels,
    target_names=[ctx.target_col],
    future_names=ctx.future_covariates,
    static_names=ctx.static_covariates,
)


r = out["ALL"]
print(r["n_series"])
print(r["decoder_importance"].head(10))
print(r["encoder_importance"].head(10))
print(r["static_importance"])
print(r["attention"].sum(axis=0).head())   # each horizon column should sum to ~1 across window positions

idx = stratified_indices(labels, n_per_group=100)
out = extract_tft_interpretation(ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
        indices=idx, group_labels=labels, target_names=[ctx.target_col],
        future_names=ctx.future_covariates, static_names=ctx.static_covariates)
save_results(out, "tft_interp")
plot_importance(out["ALL"]["decoder_importance"], title="Decoder importance")
plot_importance(out["ALL"]["encoder_importance"], title="Encoder importance")
plot_attention(out["ALL"]["attention"], title="Attention")