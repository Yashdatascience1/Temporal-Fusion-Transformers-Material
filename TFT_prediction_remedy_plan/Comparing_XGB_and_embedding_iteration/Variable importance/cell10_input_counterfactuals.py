# ----- CELL 10: which of the three new inputs pulls the forecast down? (no retraining) ---------------------------
# Needs cells 1-6 run first (uses best, ser, covs, cal_df, predict_samples, H0, FC_START, TARGET_DATE, ROLES,
# series_covariates, z_petrol_2025, TimeSeries). Part A checks whether each input is inside the range seen in
# history. Part B re-runs the forecast with one input at a time put back to an in-range value.
import torch

SEED         = 0
MODEL_NOV6   = 79_000                       # what your model predicted for Nov 6 (all series)
ACTUAL_NOV6  = 300_000                      # what was actually sold
NEEDED_X     = ACTUAL_NOV6 / MODEL_NOV6     # the factor the forecast would have to grow by (about 3.8)

PETROL, GST = "PETROL_PRICE_INDEX", "POST_GST_CUT"
SPEND = list(series_covariates)

for _n in ["best", "ser", "covs", "cal_df", "predict_samples", "H0", "FC_START", "TARGET_DATE", "ROLES",
           "series_covariates", "z_petrol_2025", "TimeSeries"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-6 first"
comps = list(covs[0].components)
for _c in [PETROL, GST] + SPEND:
    assert _c in comps, f"{_c} is not a covariate column; columns are {comps[:12]} ..."
N_FC = best.output_chunk_length
LAG  = 365                                  # 2026-10-01 minus 365 days = 2025-10-01, so months line up
print(f"spend columns: {SPEND}")

# ---- PART A: are the three inputs inside the range the model saw in history? -----------------------------------
hist_end = FC_START - pd.Timedelta(days=1)
fc_end   = FC_START + pd.Timedelta(days=N_FC - 1)
rows, seen = [], set()
for c in covs:
    if id(c) in seen:
        continue
    seen.add(id(c))
    df = c.pd_dataframe()
    for col in [PETROL] + SPEND:
        h, f = df.loc[:hist_end, col], df.loc[FC_START:fc_end, col]
        sd = h.std()
        rows.append(dict(col=col, shift_in_hist_sd=(f.mean() - h.mean()) / sd if sd > 0 else np.nan,
                         days_above_hist_max=(f > h.max()).mean(), days_below_hist_min=(f < h.min()).mean()))
ra = (pd.DataFrame(rows).groupby("col")
      .agg(n_paths=("shift_in_hist_sd", "size"), median_shift_in_hist_sd=("shift_in_hist_sd", "median"),
           share_days_above_hist_max=("days_above_hist_max", "mean"),
           share_days_below_hist_min=("days_below_hist_min", "mean")))
print("\nA. forecast-window values versus the same column's history (before Oct 1, 2026):")
print(ra.round(3).to_string())
g = cal_df[GST]
print(f"\n{GST}: ON on {g.loc[:hist_end].mean() * 100:.1f}% of history days, on {g.loc[FC_START:fc_end].mean() * 100:.1f}% "
      f"of forecast days; first ON date = {g.index[g == 1][0].date() if (g == 1).any() else 'never'}")


# ---- PART B: counterfactuals ----------------------------------------------------------------------------------
def make(c, flat=None, lag_cols=()):
    """Copy of covariate series: `flat` columns set to a value, `lag_cols` columns copied from 365 days earlier,
    over the 92 forecast days only."""
    cols, vals = list(c.components), c.values().copy()
    i0 = int((FC_START - c.start_time()).days)
    assert i0 - LAG >= 0 and i0 + N_FC <= len(vals), "covariate series too short for this edit"
    for col, v in (flat or {}).items():
        vals[i0:i0 + N_FC, cols.index(col)] = v
    for col in lag_cols:
        j = cols.index(col)
        vals[i0:i0 + N_FC, j] = vals[i0 - LAG:i0 - LAG + N_FC, j]
    return TimeSeries.from_times_and_values(c.time_index, vals, columns=cols)


_cache = {}
def scenario(tag, **kw):
    out = []
    for c in covs:                          # many series share one covariate object: edit each once
        k = (tag, id(c))
        if k not in _cache:
            _cache[k] = make(c, **kw)
        out.append(_cache[k])
    return out


def run(cov_list, seed=SEED):
    torch.manual_seed(seed)                 # same random draws for every scenario -> fair comparison
    _, v = predict_samples(ser, cov_list)
    return v.mean(axis=2).sum(axis=0)       # mean forecast summed over the sample's series, per day


base   = run(covs)
base_b = run(covs, SEED + 1)                # same inputs, other draws = noise floor
scen = {
    "1 petrol -> training mean (0)":        scenario("p0", flat={PETROL: 0.0}),
    "2 petrol -> Oct-2025 level":           scenario("p25", flat={PETROL: z_petrol_2025}),
    "3 spend -> same days last year":       scenario("s", lag_cols=SPEND),
    f"4 {GST} -> 0":                        scenario("g", flat={GST: 0.0}),
    "5 petrol(mean) + spend(last year)":    scenario("ps", flat={PETROL: 0.0}, lag_cols=SPEND),
    f"6 petrol + spend + {GST}=0":          scenario("psg", flat={PETROL: 0.0, GST: 0.0}, lag_cols=SPEND),
}
out = [dict(scenario="baseline (as forecast)", nov6=base[H0], nov6_x=1.0, total_x=1.0, gap_closed_pct=0.0),
       dict(scenario="baseline, other draws (noise)", nov6=base_b[H0], nov6_x=base_b[H0] / base[H0],
            total_x=base_b.sum() / base.sum(), gap_closed_pct=np.nan)]
for name, cv in scen.items():
    d = run(cv)
    x = d[H0] / base[H0]
    out.append(dict(scenario=name, nov6=d[H0], nov6_x=x, total_x=d.sum() / base.sum(),
                    gap_closed_pct=100 * np.log(x) / np.log(NEEDED_X)))
res = pd.DataFrame(out)
print(f"\nB. Nov 6 on {len(ser)} series (ratios matter; the {len(ser)}-series total is not your 79K).")
print(f"   To reach actual sales the Nov 6 forecast must grow {NEEDED_X:.2f}x ({MODEL_NOV6:,} -> {ACTUAL_NOV6:,}).")
print(res.round(3).to_string(index=False))
res.to_csv(os.path.join(BASE, "input_counterfactuals.csv"), index=False)

print("""
How to read it (gap_closed_pct = how much of the 3.8x gap that change closes, on a log scale; 100 = fully closed):
- One input closes most of the gap (rows 1-4)           -> that input is the main culprit.
- Each closes a part and row 6 gets close to 100        -> they act together; retrain without all three.
- Row 6 stays far below 100                              -> the cause is deeper than these inputs (or the model now
                                                            relies on them to set the level).
- Part A: petrol or spend far above history max          -> out-of-range extrapolation, same story as petrol.
- Row 4 is read loosely: a flag at 0 after the cut date was never seen in training.""")
