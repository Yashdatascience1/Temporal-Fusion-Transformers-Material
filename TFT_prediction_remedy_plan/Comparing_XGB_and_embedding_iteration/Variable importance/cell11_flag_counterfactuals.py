# ----- CELL 11: how much does the Nov 6 forecast move when other inputs change? (no retraining) ---------------------
# Needs cells 1-6 and cell 10 run first. Same idea as cell 10, applied to more inputs, one at a time:
#   A. festival flags that are 0 on Nov 6 (Rath Yatra, Janmashtami, the next-highest-weight ones): switch ON for Nov 6 only
#   B. flags that really are 1 on Nov 6 (D-2 and the others): switch OFF for Nov 6 only
#   C. the two spend columns separately: set to the same days last year
import torch

NAMED_ON   = ["JAGANNATH_RATHYATRA", "JANMASHTAMI"]     # flags you asked about
EXTRA_ON   = []                                          # add more flag names here if you want
N_AUTO     = 6                                           # plus the N_AUTO highest-table-weight never-firing flags

for _n in ["best", "ser", "covs", "cal_df", "imp", "binary_covariates", "series_covariates", "TARGET_DATE", "H0",
           "FC_START", "nov6", "set_days", "scenario", "run", "base", "base_b", "dead", "GST", "SPEND"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-6 and cell 10 first"
for _f in NAMED_ON + EXTRA_ON:
    assert _f in binary_covariates, f"{_f} is not a binary flag; try one of {[c for c in binary_covariates if _f[:4] in c]}"

weight = imp[imp.part == "decoder"].set_index("variable")["importance_pct"]       # cell-3 table, % of decoder weight
asked  = list(dict.fromkeys(NAMED_ON + EXTRA_ON))
pool   = (imp[(imp.part == "decoder") & (imp.group == "other binary flags") & imp.variable.isin(dead)
              & ~imp.variable.isin(asked)].sort_values("importance_pct", ascending=False))
flags_on  = asked + list(pool.variable.head(N_AUTO))
flags_off = [c for c in binary_covariates if cal_df.loc[TARGET_DATE, c] == 1.0 and c != GST]
print(f"A. switched ON for Nov 6 only : {flags_on}")
print(f"B. switched OFF for Nov 6 only: {flags_off}   ({GST} was tested in cell 10)")
print(f"C. spend columns, one at a time: {SPEND}\n")

_c2 = {}
def edit_days(tag, col, value):
    out = []
    for c in covs:                                   # many series share one covariate object: edit each once
        k = (tag, id(c))
        if k not in _c2:
            _c2[k] = set_days(c, col, nov6, value)
        out.append(_c2[k])
    return out


noise = abs(base_b[H0] / base[H0] - 1)               # chance wobble on Nov 6 (other random draws)
THRESH = max(0.05, 3 * noise)                        # smaller changes than this are treated as zero
rows = []
def record(var, change, cov_list):
    d = run(cov_list)
    x = d[H0] / base[H0]
    rows.append(dict(variable=var, change=change, table_weight_pct=weight.get(var, np.nan), nov6=d[H0], nov6_x=x,
                     total_x=d.sum() / base.sum(), moves_nov6=abs(x - 1) > THRESH))

for f in flags_on:
    record(f, "ON for Nov 6", edit_days(f"on_{f}", f, 1.0))
for f in flags_off:
    record(f, "OFF for Nov 6", edit_days(f"off_{f}", f, 0.0))
for s in SPEND:
    record(s, "same days last year", scenario(f"lag_{s}", lag_cols=[s]))

res = pd.DataFrame(rows)
print(f"baseline Nov 6 = {base[H0]:.1f} (sum over {len(ser)} series); sampling wobble {noise * 100:.1f}%; "
      f"a change smaller than {THRESH * 100:.0f}% counts as no change.")
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(BASE, "flag_counterfactuals.csv"), index=False)

on = res[res.change == "ON for Nov 6"].dropna(subset=["table_weight_pct"])
if len(on) >= 4:
    rho = on.table_weight_pct.corr(np.abs(np.log(on.nov6_x)), method="spearman")
    print(f"\nAmong the {len(on)} flags switched ON: Spearman between table weight and size of response = {rho:.2f} "
          "(tiny sample, read loosely)")

print("""
How to read it:
- nov6_x above 1 -> the forecast for Nov 6 grew; below 1 -> it shrank. total_x is the same ratio for all 92 days.
- A flag that is 0 all quarter and still moves Nov 6 is something the model reacts to, but switching a festival ON on a
  day it never falls is an unnatural input. It shows sensitivity, not that the festival really boosts Nov 6.
- Compare table_weight_pct with nov6_x: a big weight with nov6_x near 1 means the weight is not an effect size.""")
