# ----- CELL 7: where does JAGANNATH_RATHYATRA's decoder weight go, day by day? -----------------
# Needs cells 1-6 run first (uses best, ser, covs, cal_df, _raw_weights, clean_name, with_flag, nov6, TARGET_DATE).
FLAG = "JAGANNATH_RATHYATRA"
for _n in ["best", "ser", "covs", "cal_df", "_raw_weights", "clean_name", "with_flag", "nov6",
           "TARGET_DATE", "ROLES", "binary_covariates"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-6 first"


def per_day_weights(cov_list):
    """Decoder variable-selection weights, mean over series: DataFrame (92 forecast days x 78 variables), % per day."""
    w = _raw_weights(ser, cov_list)                       # one explainer pass; leaves per-step weights on the model
    cols = [clean_name(c) for c in w["decoder_importance"].columns]
    t = best.model._decoder_sparse_weights.detach().cpu().numpy()
    t = t.reshape(t.shape[0], t.shape[1], -1)             # (series, 92 days, 78 variables)
    assert t.shape[0] == len(ser) and t.shape[2] == len(cols), f"unexpected weight shape {t.shape}"
    m = t.mean(axis=0)
    m = m / m.sum(axis=1, keepdims=True) * 100.0          # each day sums to 100%
    return pd.DataFrame(m, index=pd.date_range(ROLES["forecast_start"], periods=m.shape[0], freq="D"), columns=cols)


# ---- 1. the flag's weight on every forecast day, flag OFF (the real forecast) ---------------------------
W_off = per_day_weights(covs)
s = W_off[FLAG]
print(f"{FLAG} decoder weight over the 92 days: mean {s.mean():.1f}%  median {s.median():.1f}%  "
      f"min {s.min():.2f}%  max {s.max():.1f}%")
print(f"days with weight above 10%: {int((s > 10).sum())} of {len(s)}")
print("\nhighest-weight days (%):");  print(s.nlargest(8).round(1).to_string())
print("\nlowest-weight days (%):");   print(s.nsmallest(8).round(2).to_string())
print("\nmean weight by weekday (0 = Monday):");  print(s.groupby(s.index.dayofweek).mean().round(1).to_string())
print("\nmean weight by month:");                 print(s.groupby(s.index.month).mean().round(1).to_string())

# ---- 2. is it a 'resting slot' that empties when a real festival flag is active? -----------------------
festival_cols = [c for c in binary_covariates if not c.startswith("IS_") and c != "POST_GST_CUT"]
n_active = cal_df.loc[W_off.index, festival_cols].sum(axis=1)
print(f"\ncorrelation of the flag's weight with the number of festival flags ON that day: {s.corr(n_active):.2f}")
print(f"mean weight on days with no festival flag on : {s[n_active == 0].mean():.1f}%  ({int((n_active == 0).sum())} days)")
print(f"mean weight on days with a festival flag on  : {s[n_active > 0].mean():.1f}%  ({int((n_active > 0).sum())} days)")

# ---- 3. what happens to the weight on Nov 6 when the flag really fires --------------------------------
W_on = per_day_weights(with_flag(FLAG, nov6))
rank_off = int(W_off.loc[TARGET_DATE].rank(ascending=False)[FLAG])
rank_on = int(W_on.loc[TARGET_DATE].rank(ascending=False)[FLAG])
print(f"\nNov 6: weight with the flag OFF = {W_off.loc[TARGET_DATE, FLAG]:.2f}% (rank {rank_off} of {W_off.shape[1]})"
      f"  ->  flag ON = {W_on.loc[TARGET_DATE, FLAG]:.2f}% (rank {rank_on})")
print("top 5 inputs on Nov 6 with the flag ON (%):"); print(W_on.loc[TARGET_DATE].nlargest(5).round(1).to_string())

print("""
How to read it:
- High on days with no festival flag, low on festival days -> the flag is a resting slot: weight sits there when nothing
  else is happening, and a high resting weight is why switching it on has a large effect.
- Weight jumps when the flag fires (OFF -> ON)             -> the selection layer reacts to the flag itself.
- High on a few specific days only                         -> look at what those days share (weekday, month-end).
- Flat near 11% every day                                  -> none of the above; the weight is a fixed property of the layer.""")
