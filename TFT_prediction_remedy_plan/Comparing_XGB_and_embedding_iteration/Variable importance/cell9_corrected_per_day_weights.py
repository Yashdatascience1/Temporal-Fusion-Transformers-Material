# ----- CELL 9: per-day decoder weights with CORRECT labels -----------------------------------------
# Cells 4, 7 and 8(b) labelled the raw per-step weights with the column order of Darts' importance table. Darts sorts
# that table's columns by the first series' weights, so those labels were wrong. Here the weights are labelled with the
# model's own variable order, and checked against the (correctly labelled) table before anything is printed.
# Needs cells 1-6 run first (uses best, ser, covs, cal_df, _stack, clean_name, group_of, with_flag, nov6, TARGET_DATE).
FLAG = "JAGANNATH_RATHYATRA"
for _n in ["best", "ser", "covs", "cal_df", "_stack", "clean_name", "group_of", "with_flag", "nov6",
           "TARGET_DATE", "ROLES", "binary_covariates", "TFTExplainer"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-6 first"


def per_day_weights(cov_list):
    """Decoder weights, mean over series: DataFrame (92 forecast days x 78 variables), % per day, correct labels."""
    explainer = TFTExplainer(best, background_series=ser, background_future_covariates=cov_list)
    result = explainer.explain()                                     # one predict over all series
    t = best.model._decoder_sparse_weights.detach().cpu().numpy()    # (series, 92, 1, 78) in MODEL order
    t = t.reshape(t.shape[0], t.shape[1], -1)
    names = [clean_name(explainer._name_mapping[n]) for n in best.model.decoder_variables]
    assert t.shape[0] == len(ser) and t.shape[2] == len(names), f"unexpected weight shape {t.shape}"
    m = t.mean(axis=0)
    m = m / m.sum(axis=1, keepdims=True) * 100.0
    W = pd.DataFrame(m, index=pd.date_range(ROLES["forecast_start"], periods=m.shape[0], freq="D"), columns=names)

    # built-in check: the table's own labels must agree with these labels once averaged over the 92 days
    tab_ = _stack(result.get_feature_importances()["decoder_importance"]).mean(axis=0)
    tab_.index = [clean_name(c) for c in tab_.index]
    tab_ = tab_ / tab_.sum() * 100.0
    rho = W.mean(axis=0).corr(tab_.reindex(W.columns), method="spearman")
    print(f"   label check: Spearman between table and per-day weights (92-day means) = {rho:.3f}")
    assert rho > 0.9, "labels still disagree with the table; send me this traceback"
    return W


# ---- 1. Nov 6, flag OFF (the real forecast) -------------------------------------------------------------------
W_off = per_day_weights(covs)
d6 = W_off.loc[TARGET_DATE].sort_values(ascending=False)
tbl = pd.DataFrame({"nov6_pct": d6, "mean_92_days_pct": W_off.mean(axis=0).reindex(d6.index),
                    "group": [group_of("decoder", v) for v in d6.index]})
print(f"\nDECODER weights on {TARGET_DATE.date()}, top 20 (equal share = {100 / len(d6):.2f}%):")
print(tbl.head(20).round(2).to_string())
print("\nby group, Nov 6 vs 92-day mean (% summed):")
print(tbl.groupby("group")[["nov6_pct", "mean_92_days_pct"]].sum().round(2)
      .sort_values("nov6_pct", ascending=False).to_string())
on_flags = [c for c in binary_covariates if cal_df.loc[TARGET_DATE, c] == 1.0]
off_flags = [c for c in binary_covariates if cal_df.loc[TARGET_DATE, c] == 0.0]
print("\nbinary flags ON on Nov 6:", {c: round(float(d6[c]), 2) for c in on_flags})
print(f"share of Nov 6 weight sitting on the {len(off_flags)} binary flags that are 0 that day: {d6[off_flags].sum():.1f}%")
for v in ["D-2", "PETROL_PRICE_INDEX", "TOTAL_SPEND_PER_10K", "add_relative_index"]:
    if v in d6.index:
        print(f"   {v:22s} Nov 6 = {d6[v]:.2f}%   92-day mean = {W_off[v].mean():.2f}%")

# ---- 2. the Rath Yatra flag, day by day ---------------------------------------------------------------------
s = W_off[FLAG]
festival_cols = [c for c in binary_covariates if not c.startswith("IS_") and c != "POST_GST_CUT"]
n_active = cal_df.loc[W_off.index, festival_cols].sum(axis=1)
print(f"\n{FLAG}: mean {s.mean():.1f}%  median {s.median():.1f}%  min {s.min():.2f}%  max {s.max():.1f}%   "
      f"days above 10%: {int((s > 10).sum())} of {len(s)}")
print("highest-weight days (%):"); print(s.nlargest(6).round(1).to_string())
print("lowest-weight days (%):");  print(s.nsmallest(6).round(1).to_string())
print(f"correlation with number of festival flags ON that day: {s.corr(n_active):.2f}")
print(f"mean weight on days with no festival flag on: {s[n_active == 0].mean():.1f}%  "
      f"| with a festival flag on: {s[n_active > 0].mean():.1f}%")
print("mean weight by weekday (0 = Monday):", s.groupby(s.index.dayofweek).mean().round(1).to_dict())

# ---- 3. switch the flag ON for Nov 6 only and see how the weights move ----------------------------------------
W_on = per_day_weights(with_flag(FLAG, nov6))
print(f"\nNov 6: {FLAG} weight OFF = {W_off.loc[TARGET_DATE, FLAG]:.2f}%  ->  ON = {W_on.loc[TARGET_DATE, FLAG]:.2f}%")
chg = (W_on.loc[TARGET_DATE] - W_off.loc[TARGET_DATE]).sort_values(key=np.abs, ascending=False)
print("five biggest weight changes on Nov 6 (percentage points):"); print(chg.head(5).round(2).to_string())

print("""
How to read it:
- Rath Yatra mean near 11% with few high or low days -> the table was right: the model gives this never-firing flag a
  large, almost fixed weight (a learned prior), and it is the same on quiet and festival days.
- Weight much higher on quiet days than festival days -> the 'resting slot' idea.
- Weight jumps when the flag is ON                    -> the selection layer reacts to the flag itself.""")
