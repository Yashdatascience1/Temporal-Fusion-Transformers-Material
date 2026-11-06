# ======================================================================================
# Section 11 (new cell, run at the END of 02_modelling_xgb_features.ipynb)
# TFTExplainer variable importance for ALL variables of iteration 4:
#   static (8 embedded columns) | encoder (past NET_SALES + 77 future covariates + rel. index)
#   | decoder (77 future covariates + rel. index)
#
# Uses objects already defined in that notebook: MODEL_NAME, val_seq, val_cov, ROLES,
# static_covariates, binary_covariates, series_covariates, PENALTY_COLS, target_col, BASE.
# ======================================================================================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.models import TFTModel
from darts.explainability import TFTExplainer

N_EXPLAIN     = 1000     # val_seq is already shuffled (seed 42), so the first N are a random sample
RUN_STABILITY = True     # re-run on a second, disjoint sample and compare rankings
TOP_K_PLOT    = 25

CALENDAR_COLS = ROLES["calendar_covariates"]
PETROL_COLS   = ROLES["shared_continuous_covariates"]
OTHER_FLAGS   = [c for c in binary_covariates if c not in PENALTY_COLS]

# ---- model: best checkpoint (the one run_metadata.json points to) --------------------
best = TFTModel.load_from_checkpoint(MODEL_NAME, best=True)


# ---- helpers -------------------------------------------------------------------------
def clean_name(name):
    """Darts tags variables with a role suffix and (possibly) internal static names."""
    name = str(name)
    m = re.fullmatch(r"static_covariate_(\d+)", name)
    if m:                                         # internal name -> real static column
        return static_covariates[int(m.group(1))]
    return re.sub(r"_(fut_cov|past_cov|static_cov|target|futcov|pastcov)$", "", name)


def group_of(part, var):
    if part == "static":
        return "static (embedded)"
    if var == target_col:
        return "past sales (target history)"
    if "relative_index" in var:
        return "relative index"
    if var in series_covariates:
        return "spend (per-series)"
    if var in PETROL_COLS:
        return "petrol"
    if var in CALENDAR_COLS:
        return "calendar (dow, month-end)"
    if var in PENALTY_COLS:
        return "festive-window flags"
    if var in OTHER_FLAGS:
        return "other binary flags"
    return "UNMATCHED"


def explain_sample(lo, hi):
    ser = [val_seq[i] for i in range(lo, hi)]
    cov = [val_cov[i] for i in range(lo, hi)]
    # decoder needs covariates through VAL_END + 92 days = FORECAST_END
    assert all(c.end_time() >= s.end_time() + pd.Timedelta(days=best.output_chunk_length)
               for s, c in zip(ser[:20], cov[:20])), "future covariates do not reach FORECAST_END"
    explainer = TFTExplainer(best, background_series=ser, background_future_covariates=cov)
    result = explainer.explain()
    try:
        w = result.get_feature_importances()
    except AttributeError:                        # other Darts builds expose getters instead
        w = dict(encoder_importance=result.get_encoder_importance(),
                 decoder_importance=result.get_decoder_importance(),
                 static_covariates_importance=result.get_static_covariates_importance())
    rows = []
    for key, part in [("encoder_importance", "encoder"), ("decoder_importance", "decoder"),
                      ("static_covariates_importance", "static")]:
        df = w.get(key)
        if df is None or len(df) == 0:
            continue
        s = df.mean(axis=0)
        s = s * (100.0 / s.sum())                 # force % within each part, whatever the raw scale
        for name, val in s.items():
            v = clean_name(name)
            rows.append(dict(part=part, variable=v, importance_pct=float(val), group=group_of(part, v)))
    out = pd.DataFrame(rows)
    out["uniform_pct"] = 100.0 / out.groupby("part")["variable"].transform("count")
    out["x_uniform"] = out["importance_pct"] / out["uniform_pct"]
    return out, result, explainer


# ---- run -----------------------------------------------------------------------------
imp, result, explainer = explain_sample(0, N_EXPLAIN)
imp = imp.sort_values(["part", "importance_pct"], ascending=[True, False]).reset_index(drop=True)

bad = imp[imp["group"] == "UNMATCHED"]
if len(bad):
    print("*** UNMATCHED variable names (fix clean_name / group_of):", bad["variable"].tolist()[:10])

print(f"explained {N_EXPLAIN} series; encoder window ends {pd.Timestamp(ROLES['val_end']).date()}, "
      f"decoder window = {ROLES['forecast_start']} -> {ROLES['forecast_end']}")
print(imp.groupby("part").size().to_dict(), "variables per part")
print(imp.groupby("part")["importance_pct"].sum().round(2).to_dict(), "(sum per part, should be 100)")

# every variable, side by side
wide = (imp.pivot_table(index=["group", "variable"], columns="part",
                        values="importance_pct", aggfunc="first")
          .reset_index())
print("\n=== ALL variables (% of VSN weight within each part) ===")
print(wide.sort_values(["group", "decoder"], ascending=[True, False]).round(3).to_string(index=False))

# group level (69 flags dilute each other, so groups are easier to read)
grp = (imp.groupby(["part", "group"])["importance_pct"].sum()
          .unstack("part").fillna(0).round(2))
n_vars = imp.groupby(["part", "group"]).size().unstack("part").fillna(0).astype(int).add_prefix("n_vars_")
print("\n=== By group (% summed over the group's variables) ===")
print(grp.join(n_vars).to_string())

imp.to_csv(os.path.join(BASE, "tft_variable_importance_long.csv"), index=False)
wide.to_csv(os.path.join(BASE, "tft_variable_importance_wide.csv"), index=False)

# ---- plots ---------------------------------------------------------------------------
parts = [p for p in ["static", "encoder", "decoder"] if p in imp["part"].unique()]
fig, axes = plt.subplots(1, len(parts), figsize=(7 * len(parts), 8))
axes = np.atleast_1d(axes)
for ax, p in zip(axes, parts):
    g = imp[imp["part"] == p].nlargest(TOP_K_PLOT, "importance_pct").sort_values("importance_pct")
    ax.barh(g["variable"], g["importance_pct"])
    ax.axvline(g["uniform_pct"].iloc[0], ls="--", c="grey", lw=1)       # equal-weight baseline
    ax.set_title(f"{p}: top {min(TOP_K_PLOT, len(g))}  (dashed = equal weights)")
    ax.set_xlabel("% of VSN weight")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "tft_variable_importance.png"), dpi=150)
plt.show()

# ---- stability: do two disjoint samples of series agree on the ranking? ----------------
if RUN_STABILITY:
    imp_b, _, _ = explain_sample(N_EXPLAIN, 2 * N_EXPLAIN)
    for p in parts:
        a = imp[imp.part == p].set_index("variable")["importance_pct"]
        b = imp_b[imp_b.part == p].set_index("variable")["importance_pct"].reindex(a.index)
        print(f"rank agreement (Spearman) between two samples, {p:8s}: {a.corr(b, method='spearman'):.3f}")
