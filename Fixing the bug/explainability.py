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


from darts.explainability import TFTExplainer
import pandas as pd

N = 100

# ── Real 2026 run (the failing one) ───────────────────────────────────
explainer = TFTExplainer(
    loaded_model,
    background_series              = final_scaled_lookback_data[:N],
    background_future_covariates   = final_scaled_lookahead_data[:N]
)

result_real = explainer.explain(
    foreground_series              = final_scaled_lookback_data[:N],
    foreground_future_covariates   = final_scaled_lookahead_data[:N]
)

# ── Swap run — 2025 covariate values in 2026 horizon ──────────────────
# (Build swapped_final_lookahead using the swap code from earlier)
result_swap = explainer.explain(
    foreground_series              = final_scaled_lookback_data[:N],
    foreground_future_covariates   = swapped_final_lookahead[:N]
)

# ── Extract and compare decoder importances ───────────────────────────
dec_real = result_real.get_decoder_importance()
dec_swap = result_swap.get_decoder_importance()

if isinstance(dec_real, list):
    dec_real = pd.concat(dec_real).groupby(level=0).mean()
    dec_swap = pd.concat(dec_swap).groupby(level=0).mean()

comparison = pd.DataFrame({
    'real_2026_covariates'  : dec_real.mean(axis=0),
    'swap_2025_covariates'  : dec_swap.mean(axis=0)
}).sort_values('real_2026_covariates', ascending=False)

comparison['shift'] = comparison['swap_2025_covariates'] - comparison['real_2026_covariates']

print("Decoder importance — real 2026 vs 2025 covariate swap:")
print(comparison.round(4))
print("\nFeatures that gain most importance when 2025 covariates are injected:")
print(comparison.sort_values('shift', ascending=False).head(10).round(4))