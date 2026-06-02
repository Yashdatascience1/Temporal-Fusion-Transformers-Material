N = 100

explainer = TFTExplainer(
    loaded_model,
    background_series=final_scaled_lookback_data[:N],
    background_future_covariates=final_scaled_lookahead_data[:N],
)
result = explainer.explain()

# CHECKPOINT 1 — look at this output, find the right accessor names
print([m for m in dir(result) if not m.startswith('_')])

# Substitute the actual method names you found:
dec_imp = result.get_decoder_importance()   # <-- adjust if needed
enc_imp = result.get_encoder_importance()   # <-- adjust if needed

# CHECKPOINT 2 — confirm structure before sorting
print(type(dec_imp))
if isinstance(dec_imp, list):
    import pandas as pd
    dec_imp = pd.concat(dec_imp, axis=0)
    enc_imp = pd.concat(enc_imp, axis=0)

# Top features
dec_top = dec_imp.mean(axis=0).sort_values(ascending=False)
enc_top = enc_imp.mean(axis=0).sort_values(ascending=False)

print("DECODER (this is the one that matters for Jun/Jul/Aug):")
print(dec_top.head(15))
print("\nENCODER:")
print(enc_top.head(15))

dec_imp.to_csv('decoder_importance.csv')
enc_imp.to_csv('encoder_importance.csv')