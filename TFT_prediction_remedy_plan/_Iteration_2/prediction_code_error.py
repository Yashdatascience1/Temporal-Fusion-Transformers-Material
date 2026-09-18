# ---------------- FORECAST ----------------
def summarise(p):
    """NegBin params (mu, alpha) -> mean and quantiles, closed form.
    The mean IS mu, so sampling 200 times to estimate it was wasted work."""
    v = p.values(copy=False)
    mu = v[:, 0].astype(np.float64)
    alpha = np.clip(v[:, 1].astype(np.float64), 1e-6, None)
    n = 1.0 / alpha
    pr = n / (n + mu)
    out = {"PRED_MEAN": mu}
    for q_int, q_val in [(60, 0.60), (70, 0.70), (80, 0.80), (85, 0.85)]:
        out[f"PRED_Q{q_int}"] = stats.nbinom.ppf(q_val, n, pr)
    return out


os.makedirs(OUT_DIR, exist_ok=True)
n_blocks = (len(keys) + BLOCK - 1) // BLOCK
paths, t0, checked = [], datetime.now(), False

for b in range(n_blocks):
    lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(keys))
    path = os.path.join(OUT_DIR, f"{MODEL_NAME}_block_{b:04d}.parquet")
    paths.append(path)

    if os.path.exists(path):
        print(f"[{b+1}/{n_blocks}] exists, skipping")
        continue

    preds = model.predict(
        n=HORIZON,
        series=History(keys[lo:hi], statics[lo:hi]),
        future_covariates=SharedCov(SHARED_COV, hi - lo),
        predict_likelihood_parameters=True,   # 1 pass instead of num_samples passes
        num_samples=1,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    if not checked:   # fail fast if dates or params are wrong
        assert preds[0].start_time() == FORECAST_START, \
            f"forecast starts {preds[0].start_time().date()}, expected {FORECAST_START.date()}"
        print(f"  params: {list(preds[0].components)}")
        checked = True

    pd.concat(
        [pd.DataFrame({group_col: k, time_col: p.time_index, **summarise(p)})
         for k, p in zip(keys[lo:hi], preds)],
        ignore_index=True,
    ).to_parquet(path, index=False)

    el = (datetime.now() - t0).total_seconds()
    print(f"[{b+1}/{n_blocks}] {hi:,}/{len(keys):,} | {hi/el:,.0f} series/s | "
          f"ETA {(len(keys)-hi)/(hi/el)/60:.0f} min")

    del preds
    gc.collect()


# ---------------- OUTPUT ----------------
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
out = os.path.join(OUT_DIR, f"{MODEL_NAME}_predictions.parquet")
df.to_parquet(out, index=False)

print(f"\nWritten -> {out}")
print(f"  rows: {len(df):,} | elapsed: {(datetime.now()-t0).total_seconds()/60:.1f} min")
for c in ["PRED_MEAN", "PRED_Q60", "PRED_Q70","PRED_Q80","PRED_Q85"]:
    print(f"  {c}: {df[c].sum():,.0f} ({df[c].sum()/1e5:.2f} lacs)")