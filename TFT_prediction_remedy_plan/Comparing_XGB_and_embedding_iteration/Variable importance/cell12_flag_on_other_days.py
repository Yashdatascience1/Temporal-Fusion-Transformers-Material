# ----- CELL 12: is the Rath Yatra response specific to Nov 6, or does it happen on any day? (no retraining) -------------
# Needs cells 1-6 and cell 10 run first (uses covs, ser, set_days, run, base, base_b, cal_df, binary_covariates, FC_START,
# TARGET_DATE, ROLES). For each test day, the flag is switched ON for that one day only and we look at that day's forecast.
import torch

FLAGS      = ["JAGANNATH_RATHYATRA", "PONGAL"]    # PONGAL pushed Nov 6 DOWN in cell 11, Rath Yatra pushed it UP
N_QUIET    = 3                                     # quiet Fridays to test (Nov 6 2026 is a Friday)
N_PEAK     = 2                                     # other highest-forecast days to test

for _n in ["best", "ser", "covs", "cal_df", "binary_covariates", "set_days", "run", "base", "base_b", "FC_START",
           "TARGET_DATE", "ROLES"]:
    assert _n in globals(), f"{_n} is not defined: run cells 1-6 and cell 10 first"
for _f in FLAGS:
    assert _f in binary_covariates, f"{_f} is not a binary flag"

n_days  = best.output_chunk_length
fc_days = pd.date_range(ROLES["forecast_start"], periods=n_days, freq="D")
festival_cols = [c for c in binary_covariates if not c.startswith("IS_") and c != "POST_GST_CUT"]
n_active = cal_df.loc[fc_days, festival_cols].sum(axis=1)
near_fest = n_active.rolling(5, center=True, min_periods=1).max() > 0       # a festival flag within +-2 days

# ---- choose the test days ------------------------------------------------------------------------------------------
quiet_pool = [d for d in fc_days if d.dayofweek == 4 and not near_fest[d] and base[(d - FC_START).days] <= np.median(base)]
assert len(quiet_pool) >= N_QUIET, f"only {len(quiet_pool)} quiet Fridays found; lower N_QUIET"
pick = np.linspace(0, len(quiet_pool) - 1, N_QUIET).round().astype(int)     # spread across Oct-Dec
quiet = [quiet_pool[i] for i in pick]
peak_order = [fc_days[k] for k in np.argsort(-base) if fc_days[k] != TARGET_DATE]
peaks = peak_order[:N_PEAK]
tests = [(TARGET_DATE, "Nov 6 (D-2 is ON)")] + [(d, "other high-forecast day") for d in peaks] + \
        [(d, "quiet Friday, no festival nearby") for d in quiet]
print("test days:", [(d.date().isoformat(), lab) for d, lab in tests], "\n")

# ---- run: flag ON for one day at a time -------------------------------------------------------------------------------
_c3 = {}
def on_day(flag, date):
    out = []
    for c in covs:                                   # many series share one covariate object: edit each once
        k = (flag, date, id(c))
        if k not in _c3:
            _c3[k] = set_days(c, flag, pd.DatetimeIndex([date]), 1.0)
        out.append(_c3[k])
    return out


rows = []
for flag in FLAGS:
    for date, label in tests:
        k = (date - FC_START).days
        d = run(on_day(flag, date))
        rows.append(dict(flag=flag, date=date.date(), day_type=label, weekday=date.day_name()[:3],
                         baseline_units=base[k], with_flag_units=d[k], x=d[k] / base[k], added_units=d[k] - base[k],
                         noise_pct=abs(base_b[k] / base[k] - 1) * 100))
res = pd.DataFrame(rows)
print("Forecast on the test day with the flag ON for that day only (units summed over the sample's series):")
print(res.round(2).to_string(index=False))
res.to_csv(os.path.join(BASE, "flag_on_other_days.csv"), index=False)

print("""
How to read it (x = forecast with the flag ON / baseline forecast, on that same day):
- x about the same on quiet days and on Nov 6 -> the flag works as a generic 'special day' lift. Nothing about Nov 6 is
  specific, and the model is not using it as a festival signal.
- x near 1 on quiet days but large on Nov 6   -> the response needs something else that Nov 6 has (D-2 or the other flags on).
- added_units about equal on every day         -> the flag adds a fixed amount; x much larger on quiet days.
- noise_pct is the chance wobble of that day; ignore changes of that size. Quiet days are small numbers, so noisier.""")
