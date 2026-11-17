# ----- CELL 8: which window does the cell-3 decoder table describe? --------------------------------
# Needs cells 1-7 run first (uses imp, cal_df, dead, binary_covariates, per_day_weights, covs, ROLES).
for _n in ["imp", "cal_df", "dead", "per_day_weights", "covs", "ROLES"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-7 first"

VAL_START, VAL_END = pd.Timestamp(ROLES["val_start"]), pd.Timestamp(ROLES["val_end"])
tab = imp[imp.part == "decoder"].set_index("variable")["importance_pct"]       # cell-3 table (explainer)

# ---- (a) do the table's high-weight 'never fire in Oct-Dec' flags fire inside Jun 15 - Sep 30, 2026? ----
rows = []
for f in dead:
    fires = cal_df.index[(cal_df[f] == 1) & (cal_df.index >= VAL_START) & (cal_df.index <= VAL_END)]
    rows.append(dict(flag=f, table_weight_pct=tab.get(f, np.nan), fires_in_val_period=len(fires) > 0,
                     val_dates=", ".join(d.strftime("%Y-%m-%d") for d in fires[:3])))
a = pd.DataFrame(rows).sort_values("table_weight_pct", ascending=False)
print(f"(a) the {len(dead)} flags that never fire in Oct-Dec, ordered by cell-3 table weight:")
print(a.head(12).round(2).to_string(index=False))
print("\nmean table weight if the flag fires inside the validation period :",
      round(a[a.fires_in_val_period].table_weight_pct.mean(), 2), f"({int(a.fires_in_val_period.sum())} flags)")
print("mean table weight if it does not                                 :",
      round(a[~a.fires_in_val_period].table_weight_pct.mean(), 2), f"({int((~a.fires_in_val_period).sum())} flags)")

# ---- (b) decoder weights averaged over the 92 forecast days (per-step weights, as in cells 4 and 7) ----
W = per_day_weights(covs)
fc_mean = W.mean(axis=0)
cmp = pd.DataFrame({"table_pct": tab, "forecast_days_pct": fc_mean}).dropna()
cmp["table_minus_forecast"] = cmp.table_pct - cmp.forecast_days_pct
print(f"\n(b) same 78 variables, two sources. Spearman between them: "
      f"{cmp.table_pct.corr(cmp.forecast_days_pct, method='spearman'):.2f}")
print("share of weight on the never-fire flags: table "
      f"{tab[tab.index.isin(dead)].sum():.1f}%  vs  forecast-day weights {fc_mean[fc_mean.index.isin(dead)].sum():.1f}%")
print("\nlargest gaps, table HIGHER than forecast-day weights:")
print(cmp.nlargest(8, "table_minus_forecast").round(2).to_string())
print("\nlargest gaps, table LOWER than forecast-day weights:")
print(cmp.nsmallest(8, "table_minus_forecast").round(2).to_string())
cmp.to_csv(os.path.join(BASE, "decoder_weights_table_vs_forecast_days.csv"))

print("""
How to read it:
- Flags that fire in Jun 15-Sep 30 carry far more table weight than those that do not -> the table describes the validation
  windows (where those flags are ON), not the Oct-Dec forecast. Trust the forecast-day weights (b) for Oct-Dec.
- No such pattern, and (b) is close to the table                                       -> the two sources agree and the
  Rath Yatra gap needs another explanation; send me both printouts.""")
