import numpy as np
idx_s = stratified_indices(labels, n_per_group=20)
base  = [ctx.val_seq[i] for i in idx_s]
cov   = [ctx.val_cov_seq[i] for i in idx_s]

def totals(series_list, seed):
    p = ctx.model.predict(n=98, series=series_list, future_covariates=cov,
                          num_samples=100, random_state=seed, verbose=False)
    return np.array([x.all_values(copy=False).mean(axis=2).sum() for x in p])

def shuffled(col, seed=0):
    codes = np.array([s.static_covariates[col].iloc[0] for s in base])
    perm = np.random.default_rng(seed).permutation(codes)
    out = []
    for s, c in zip(base, perm):
        sc = s.static_covariates.copy(); sc[col] = c
        out.append(s.with_static_covariates(sc))
    return out

t0, t0b = totals(base, 0), totals(base, 1)
m = t0 > 1
print("noise floor (same input, different draw): %.1f%%" % (100*np.median(np.abs(t0b-t0)[m]/t0[m])))
for col in ["MODEL_FAMILY", "PARENT_DEALER_CODE", "COLOUR", "DEALER_CITY"]:
    t1 = totals(shuffled(col), 0)
    print(f"{col:20s} median change in 98-day total: {100*np.median(np.abs(t1-t0)[m]/t0[m]):.1f}%")