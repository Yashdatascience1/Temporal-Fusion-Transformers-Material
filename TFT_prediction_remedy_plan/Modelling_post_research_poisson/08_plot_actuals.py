"""
Plot actual daily NET_SALES for 01-05-2026 to 31-08-2026.

Run:  python 08_plot_actuals.py
"""

import os, glob, json
import pandas as pd
import plotly.graph_objects as go

BASE    = r"C:\Users\G0004878\Desktop\TFT_Data\Daily_forecasting_model\Model_for_scooters_only"
OUT_DIR = os.path.join(os.getcwd(), "predictions_2026")

START = pd.Timestamp("2026-05-01")
END   = pd.Timestamp("2026-08-31")

with open(os.path.join(BASE, "column_roles.json")) as f:
    ROLES = json.load(f)
time_col, target_col = ROLES["time_col"], ROLES["target_col"]


# ---------------- LOAD ----------------
files = sorted(glob.glob(os.path.join(BASE, "local_train_scooter_data", "chunk_*.parquet")))
print(f"Reading {len(files)} chunk files...")

hist = pd.concat(
    [pd.read_parquet(f, columns=[time_col, target_col]) for f in files],
    ignore_index=True,
)
hist[time_col] = pd.to_datetime(hist[time_col])

daily = (hist[(hist[time_col] >= START) & (hist[time_col] <= END)]
         .groupby(time_col)[target_col].sum().sort_index())

print(f"Days: {len(daily)} (expected {(END - START).days + 1})")
print(f"Range: {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"Total: {daily.sum():,.0f} ({daily.sum()/1e5:.2f} lacs)")
print(f"Daily mean: {daily.mean():,.0f} | min {daily.min():,.0f} | max {daily.max():,.0f}")

missing = pd.date_range(START, END).difference(daily.index)
if len(missing):
    print(f"WARNING: {len(missing)} dates missing, first few: {[d.date() for d in missing[:5]]}")


# ---------------- PLOT ----------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily.index, y=daily.values,
    mode="lines", name="Daily actual",
    line=dict(color="#1f77b4", width=1.5),
    hovertemplate="%{x|%d %b %Y} (%{x|%a})<br>%{y:,.0f} units<extra></extra>",
))

# 7-day rolling mean, so the weekly cycle doesn't hide the underlying trend
fig.add_trace(go.Scatter(
    x=daily.index, y=daily.rolling(7, center=True).mean(),
    mode="lines", name="7-day average",
    line=dict(color="#d62728", width=3),
    hovertemplate="%{x|%d %b %Y}<br>7d avg %{y:,.0f}<extra></extra>",
))

# month boundaries for reading the monthly pattern
for m in pd.date_range(START, END, freq="MS")[1:]:
    fig.add_vline(x=m, line_dash="dot", line_color="grey", opacity=0.4)

fig.update_layout(
    title=f"Actual daily NET_SALES — {START.date()} to {END.date()}",
    xaxis_title="Date", yaxis_title="Units",
    height=520, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(dtick="M1", tickformat="%b %Y")

os.makedirs(OUT_DIR, exist_ok=True)
html = os.path.join(OUT_DIR, "actuals_may_aug_2026.html")
fig.write_html(html)
print(f"\nChart -> {html}")


# ---------------- MONTHLY ----------------
print("\nMonthly totals:")
for d, v in daily.resample("MS").sum().items():
    print(f"  {d.strftime('%b %Y')}: {v:>10,.0f}  ({v/1e5:.2f} lacs)")

print("\nBy weekday (daily mean):")
for name, v in daily.groupby(daily.index.day_name()).mean().items():
    print(f"  {name:<10}: {v:>10,.0f}")
