"""
Compare NET_SALES across years for every festive flag column (N-16..N+10,
D-3..D+6), including the 2026 prediction.

Each flag fires on one day per year. So for a given column -- say D-2 -- this
pulls the actual total on that day in 2023, 2024, 2025, and the predicted
total on that day in 2026. That makes years comparable on a FESTIVAL-RELATIVE
axis instead of a calendar axis, which matters because Diwali moves:
Nov 12 (2023) -> Oct 31 (2024) -> Oct 20 (2025) -> Nov 8 (2026).

Writes an interactive HTML file with four views.

Run:  python 07_festive_profile.py
"""

import os, glob, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- CONFIG ----------------
MODEL_NAME = "PUT_YOUR_MODEL_NAME_HERE"

BASE    = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
OUT_DIR = os.path.join(os.getcwd(), "predictions_2026")
HTML    = os.path.join(OUT_DIR, "festive_profile.html")

PRED_COL = "PRED_MEAN"          # or PRED_Q60
YEARS    = [2023, 2024, 2025]   # actual years to plot

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)
time_col, group_col, target_col = ROLES["time_col"], ROLES["group_col"], ROLES["target_col"]

N_BLOCK = [f"N-{i}" for i in range(16, 0, -1)] + ["N"] + [f"N+{i}" for i in range(1, 11)]
D_BLOCK = [f"D-{i}" for i in range(3, 0, -1)]  + ["D"] + [f"D+{i}" for i in range(1, 7)]
BLOCKS  = {"Navratri / Pitru Paksha": N_BLOCK, "Diwali": D_BLOCK}


# ---------------- LOAD CALENDAR ----------------
cal = pd.read_parquet(os.path.join(BASE, "shared_calendar.parquet"))
cal[time_col] = pd.to_datetime(cal[time_col])

all_flags = [c for c in (N_BLOCK + D_BLOCK) if c in cal.columns]
missing   = [c for c in (N_BLOCK + D_BLOCK) if c not in cal.columns]
if missing:
    print(f"WARNING: not in calendar, skipped: {missing}")

# map: flag column -> {year: date it fires}
flag_dates = {}
for c in all_flags:
    hits = cal.loc[cal[c] != 0, time_col]
    flag_dates[c] = {d.year: d for d in hits}


# ---------------- LOAD ACTUALS ----------------
print("Loading actual daily totals...")
files = sorted(glob.glob(os.path.join(BASE, "local_train_scooter_data", "chunk_*.parquet")))
hist = pd.concat(
    [pd.read_parquet(f, columns=[time_col, target_col]) for f in files],
    ignore_index=True,
)
hist[time_col] = pd.to_datetime(hist[time_col])
actual_daily = hist.groupby(time_col)[target_col].sum().sort_index()
print(f"  actuals: {actual_daily.index.min().date()} -> {actual_daily.index.max().date()}")


# ---------------- LOAD PREDICTIONS ----------------
pred = pd.read_parquet(os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet"))
pred[time_col] = pd.to_datetime(pred[time_col])
pred_daily = pred.groupby(time_col)[PRED_COL].sum().sort_index()
print(f"  predictions: {pred_daily.index.min().date()} -> {pred_daily.index.max().date()}")


def value_on(date, year):
    """Actual total for a date, or predicted total if the date is in 2026."""
    if date is None:
        return np.nan
    src = pred_daily if year >= 2026 else actual_daily
    return float(src.get(date, np.nan))


# ---------------- BUILD THE COMPARISON TABLE ----------------
rows = []
for c in all_flags:
    block = "N" if c in N_BLOCK else "D"
    for yr in YEARS + [2026]:
        d = flag_dates[c].get(yr)
        rows.append({
            "flag": c, "block": block, "year": yr,
            "date": d,
            "sales": value_on(d, yr),
            "is_pred": yr >= 2026,
        })

tab = pd.DataFrame(rows)
tab["sales_lacs"] = tab["sales"] / 1e5

csv_path = os.path.join(OUT_DIR, "festive_profile.csv")
tab.to_csv(csv_path, index=False)
print(f"  table -> {csv_path}")


# ---------------- FIGURE ----------------
COLORS = {2023: "#7f7f7f", 2024: "#1f77b4", 2025: "#2ca02c", 2026: "#d62728"}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Navratri block (N-16 = Pitru Paksha start … N+10)",
        "Diwali block (D-3 … D+6)",
        "Indexed to each year's own festive total (shape, not level)",
        "Every festive day, ordered by days-from-Diwali",
    ),
    vertical_spacing=0.13, horizontal_spacing=0.08,
)

# --- panels 1 and 2: raw level, per block -----------------------------------
for col_i, (bname, block) in enumerate(BLOCKS.items(), start=1):
    order = [c for c in block if c in all_flags]
    for yr in YEARS + [2026]:
        sub = tab[(tab.year == yr) & (tab.flag.isin(order))].set_index("flag").reindex(order)
        fig.add_trace(
            go.Scatter(
                x=order, y=sub["sales_lacs"],
                name=f"{yr}" + (" (pred)" if yr == 2026 else ""),
                legendgroup=str(yr), showlegend=(col_i == 1),
                mode="lines+markers",
                line=dict(color=COLORS[yr], width=3 if yr == 2026 else 2,
                          dash="dash" if yr == 2026 else "solid"),
                marker=dict(size=6),
                hovertemplate=f"<b>{yr}</b> %{{x}}<br>%{{y:.2f}} lacs<extra></extra>",
            ),
            row=1, col=col_i,
        )

# --- panel 3: indexed shape -------------------------------------------------
# Divides each year by its own festive-day total, so the LEVEL difference is
# removed and only the SHAPE remains. This is where a misplaced peak shows up
# unambiguously -- a year whose peak sits on the wrong flag will visibly
# diverge here even if its total is right.
order_all = [c for c in (N_BLOCK + D_BLOCK) if c in all_flags]
for yr in YEARS + [2026]:
    sub = tab[(tab.year == yr)].set_index("flag").reindex(order_all)
    tot = sub["sales"].sum()
    fig.add_trace(
        go.Scatter(
            x=order_all, y=sub["sales"] / tot * 100 if tot else sub["sales"],
            name=str(yr), legendgroup=str(yr), showlegend=False,
            mode="lines+markers",
            line=dict(color=COLORS[yr], width=3 if yr == 2026 else 2,
                      dash="dash" if yr == 2026 else "solid"),
            marker=dict(size=5),
            hovertemplate=f"<b>{yr}</b> %{{x}}<br>%{{y:.2f}}%% of festive total<extra></extra>",
        ),
        row=2, col=1,
    )

# --- panel 4: ordered by days-from-Diwali -----------------------------------
# The N and D blocks are separate columns but sit on one timeline: N->D is
# exactly 28 days every year. Ordering by offset-from-Diwali puts them on a
# single axis and exposes the 14-day gap between N+10 and D-3.
offsets = {}
for c in all_flags:
    ds = []
    for yr in YEARS + [2026]:
        fd, dd = flag_dates[c].get(yr), flag_dates.get("D", {}).get(yr)
        if fd is not None and dd is not None:
            ds.append((fd - dd).days)
    if ds:
        offsets[c] = int(round(np.mean(ds)))

ordered = sorted(offsets, key=lambda c: offsets[c])
xlab = [f"{offsets[c]:+d} ({c})" for c in ordered]

for yr in YEARS + [2026]:
    sub = tab[tab.year == yr].set_index("flag").reindex(ordered)
    fig.add_trace(
        go.Scatter(
            x=xlab, y=sub["sales_lacs"],
            name=str(yr), legendgroup=str(yr), showlegend=False,
            mode="lines+markers",
            line=dict(color=COLORS[yr], width=3 if yr == 2026 else 2,
                      dash="dash" if yr == 2026 else "solid"),
            marker=dict(size=5),
            hovertemplate=f"<b>{yr}</b> %{{x}} days from Diwali<br>%{{y:.2f}} lacs<extra></extra>",
        ),
        row=2, col=2,
    )

# mark D-2, where the actual peak fell in all three prior years
if "D-2" in ordered:
    fig.add_vline(x=xlab[ordered.index("D-2")], line_dash="dot",
                  line_color="black", opacity=0.5, row=2, col=2)

fig.update_layout(
    title=dict(text="Festive-day NET_SALES by flag — actuals vs 2026 prediction", font=dict(size=18)),
    height=900, hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)
for r in (1, 2):
    for c in (1, 2):
        fig.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=r, col=c)
fig.update_yaxes(title_text="lacs", row=1, col=1)
fig.update_yaxes(title_text="lacs", row=1, col=2)
fig.update_yaxes(title_text="% of festive total", row=2, col=1)
fig.update_yaxes(title_text="lacs", row=2, col=2)

fig.write_html(HTML)
print(f"\nChart -> {HTML}")


# ---------------- TEXT SUMMARY ----------------
print("\n" + "=" * 70)
print("PEAK FLAG BY YEAR")
print("=" * 70)
for yr in YEARS + [2026]:
    sub = tab[(tab.year == yr)].dropna(subset=["sales"])
    if sub.empty:
        continue
    top = sub.loc[sub["sales"].idxmax()]
    lbl = "PREDICTED" if yr >= 2026 else "actual   "
    off = offsets.get(top["flag"], np.nan)
    print(f"  {yr} {lbl}: peak on {top['flag']:>5} "
          f"({off:+d} from Diwali) on {top['date'].date()} = {top['sales_lacs']:.2f} lacs")

print("\n" + "=" * 70)
print("FESTIVE-DAY TOTAL (the 37 flagged days only)")
print("=" * 70)
for yr in YEARS + [2026]:
    t = tab[tab.year == yr]["sales"].sum() / 1e5
    print(f"  {yr}: {t:>6.2f} lacs" + ("   <- predicted" if yr >= 2026 else ""))
