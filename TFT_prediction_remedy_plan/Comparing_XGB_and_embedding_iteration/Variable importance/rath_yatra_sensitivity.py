# ----- CELL 6: does the forecast react to JAGANNATH_RATHYATRA? (no retraining) -----------------
# Needs cells 1-5 run first. Reuses from cell 5: ser, covs, predict_samples, cal_df, H0, TARGET_DATE.
# Reuses from cell 3: best, imp.
import torch

FLAG       = "JAGANNATH_RATHYATRA"
N_CONTROLS = 3          # how many high-weight AND low-weight never-firing flags to use as controls
SEED       = 0

for _n in ["best", "ser", "covs", "imp", "cal_df", "predict_samples", "H0", "TARGET_DATE",
           "binary_covariates", "ROLES", "TimeSeries"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-5 first"

fc_days = pd.date_range(ROLES["forecast_start"], ROLES["forecast_end"], freq="D")
nov6    = pd.DatetimeIndex([TARGET_DATE])
assert len(fc_days) == best.output_chunk_length

# ---- which flags never fire in the forecast window, and when FLAG fires in history ---------------
fc_cal = cal_df.loc[fc_days[0]:fc_days[-1]]
dead   = [c for c in binary_covariates if fc_cal[c].max() == 0]
assert FLAG in dead, f"{FLAG} is not constant 0 over the forecast window"
firing = cal_df.index[cal_df[FLAG] == 1]
print(f"{FLAG} equals 1 on {len(firing)} day(s) in the whole calendar: "
      f"{[d.date().isoformat() for d in firing]}")
print("  -> setting it to 1 on many forecast days is far outside what the model saw in training.\n")

# ---- controls: other never-firing single-day festivals, highest and lowest decoder weight ---------
weight = imp[imp.part == "decoder"].set_index("variable")["importance_pct"]
pool = (imp[(imp.part == "decoder") & (imp.group == "other binary flags")
            & imp.variable.isin(dead) & (imp.variable != FLAG)]
        .sort_values("importance_pct", ascending=False))
controls = list(pool.variable.head(N_CONTROLS)) + list(pool.variable.tail(N_CONTROLS))
print("decoder weight of", FLAG, f"= {weight.get(FLAG, np.nan):.2f}%")
print("controls (never fire in Oct-Dec):", {c: round(float(weight[c]), 2) for c in controls}, "\n")


# ---- helpers ---------------------------------------------------------------------------------------
def set_days(cov, col, dates, value=1.0):
    """Copy of a covariate series with one column set to `value` on the given dates."""
    cols = list(cov.components)
    vals = cov.values().copy()
    rows = cov.time_index.get_indexer(dates)
    assert (rows >= 0).all(), "dates fall outside the covariate range"
    vals[rows, cols.index(col)] = value
    return TimeSeries.from_times_and_values(cov.time_index, vals, columns=cols)


def with_flag(col, dates):
    cache, out = {}, []                     # many series share one covariate object: edit each once
    for c in covs:
        if id(c) not in cache:
            cache[id(c)] = set_days(c, col, dates)
        out.append(cache[id(c)])
    return out


def run(cov_list, seed=SEED):
    torch.manual_seed(seed)                 # same random draws for every scenario -> fair comparison
    _, v = predict_samples(ser, cov_list)   # (series, days, samples)
    return v.mean(axis=2).sum(axis=0)       # mean forecast, summed over the series, per day


# ---- run the scenarios ---------------------------------------------------------------------------------
base   = run(covs, SEED)
base_b = run(covs, SEED + 1)                # identical inputs, different draws = sampling noise floor
rows   = []


def record(name, daily, w=np.nan):
    ratio = (daily + 1.0) / (base + 1.0)    # +1 only guards against division by zero
    k = int(np.argmax(np.abs(np.log(ratio))))
    rows.append(dict(scenario=name, decoder_weight_pct=w, nov6=daily[H0], nov6_x=daily[H0] / base[H0],
                     total_x=daily.sum() / base.sum(), biggest_day=fc_days[k].date(), biggest_day_x=ratio[k]))


record("baseline, other random draws (noise floor)", base_b)
record(f"{FLAG}=1 on Nov 6 only", run(with_flag(FLAG, nov6)), weight.get(FLAG, np.nan))
record(f"{FLAG}=1 on all 92 days", run(with_flag(FLAG, fc_days)), weight.get(FLAG, np.nan))
for f in controls:
    record(f"control {f}=1 on all 92 days", run(with_flag(f, fc_days)), weight.get(f, np.nan))

res = pd.DataFrame(rows)
print(f"baseline: Nov 6 = {base[H0]:.1f} units over {len(ser)} series; 92-day total = {base.sum():.0f}")
print(res.round(3).to_string(index=False))

noise = abs(base_b[H0] / base[H0] - 1) * 100
print(f"\nSampling noise on Nov 6 is about {noise:.1f}%, and {abs(base_b.sum() / base.sum() - 1) * 100:.1f}% on the 92-day total. "
      "Treat changes of that size as zero.")

ctrl = res[res.scenario.str.contains("all 92 days")].copy()
ctrl["abs_log_total_x"] = np.abs(np.log(ctrl["total_x"]))
if ctrl["decoder_weight_pct"].notna().sum() >= 4:
    rho = ctrl["decoder_weight_pct"].corr(ctrl["abs_log_total_x"], method="spearman")
    print(f"Spearman between decoder weight and size of response (n={len(ctrl)} flags): {rho:.2f}  (tiny sample, read loosely)")

print("""
How to read it:
- Rath Yatra rows within the noise floor      -> the model ignores the flag; its decoder weight is a parking slot.
- Large change on 'all 92 days' only          -> the model reacts to the flag, but this scenario is out of distribution,
                                                 so it shows sensitivity, not that the weight was informative.
- Large change on 'Nov 6 only'                -> the model lets one flag move a single day; compare with D-2 (-88%).
- Controls with LOW weight moving as much as high-weight ones -> weight does not predict response.""")
