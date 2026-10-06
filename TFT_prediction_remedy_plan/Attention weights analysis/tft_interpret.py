"""
tft_interpret.py - encoder importance, decoder importance and attention weights
for any trained Darts TFTModel, via forward hooks on the model's own sub-modules.

Why hooks and not TFTExplainer: your series are lazy sequences with ~100K entries.
The hook approach streams batches through predict() and keeps only running sums
(per group), so memory stays flat and nothing depends on explainer internals.

Module names used (all visible in your training log):
    static_covariates_vsn, encoder_vsn, decoder_vsn, multihead_attn
Each VSN returns (output, sparse_weights); multihead_attn returns (output, attn_weights).

Typical use (after your notebook has defined loaded_model, val_seq, val_cov_seq, val_statics):

    from tft_interpret import (extract_tft_interpretation, stratified_indices,
                               plot_importance, plot_attention, plot_var_time, save_results)

    labels = [s["MODEL_FAMILY"].iloc[0] for s in val_statics]       # same order as val_seq
    idx    = stratified_indices(labels, n_per_group=100)

    out = extract_tft_interpretation(
        loaded_model, val_seq, future_covariates=val_cov_seq,
        indices=idx, group_labels=labels,
        future_names=future_covariates, static_names=static_covariates, target_names=[target_col],
    )
    out["ALL"]["encoder_importance"]        # Series, one row per encoder variable
    out["ALL"]["decoder_importance"]        # Series, one row per decoder variable
    out["ALL"]["static_importance"]         # Series
    out["ALL"]["attention"]                 # DataFrame (position in window x horizon step)
    out["SCOOTER_FAMILY_X"]["encoder_importance"]   # same, for one group
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


# ----------------------------------------------------------------------------- sampling
def stratified_indices(labels, n_per_group=100, seed=0):
    """Pick up to n_per_group series indices from every group. Returns sorted list."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = []
    for g in pd.unique(labels):
        pos = np.flatnonzero(labels == g)
        take = pos if len(pos) <= n_per_group else rng.choice(pos, n_per_group, replace=False)
        idx.extend(int(i) for i in take)
    return sorted(idx)


# ----------------------------------------------------------------------------- naming
def _pretty_names(raw_names, target_names, past_names, future_names, static_names):
    """
    Map Darts' internal variable names (e.g. 'future_cov_12') to your real column names.
    Anything that doesn't match stays as the raw name, so print raw names on first run
    and confirm the mapping is right.
    """
    pools = {"target": target_names, "past": past_names, "future": future_names, "static": static_names}
    out = []
    for r in raw_names:
        m = re.match(r"^(.*?)_?(\d+)$", str(r))
        label = str(r)
        if m:
            prefix, i = m.group(1).lower(), int(m.group(2))
            for key, pool in pools.items():
                if key in prefix and pool is not None and i < len(pool):
                    label = pool[i]
                    break
        out.append(label)
    # de-duplicate if two raw names collapsed to the same label
    seen, final = defaultdict(int), []
    for lab, raw in zip(out, raw_names):
        seen[lab] += 1
        final.append(lab if seen[lab] == 1 else f"{lab} [{raw}]")
    return final


def _vsn_var_names(vsn, tm, attr):
    names = getattr(tm, attr, None)
    if names is None:
        names = list(vsn.single_variable_grns.keys())
    return list(names)


# ----------------------------------------------------------------------------- accumulator
class _Accumulator:
    """Running per-group sums so memory does not grow with the number of series."""

    def __init__(self):
        self.sums = defaultdict(dict)   # kind -> group -> ndarray
        self.counts = defaultdict(int)  # group -> n series (counted once, on 'attention')

    def add(self, kind, batch_arr, batch_groups):
        # batch_arr: (B, ...) numpy; batch_groups: length-B list of labels
        groups = np.asarray(batch_groups, dtype=object)
        for g in [None] + list(pd.unique(groups)):          # None == overall
            sel = slice(None) if g is None else np.flatnonzero(groups == g)
            part = batch_arr[sel].sum(axis=0)
            key = "ALL" if g is None else g
            if key in self.sums[kind]:
                self.sums[kind][key] += part
            else:
                self.sums[kind][key] = part.astype(np.float64)
            if kind == "attention":
                self.counts[key] += len(groups) if g is None else len(sel)

    def mean(self, kind, group):
        return self.sums[kind][group] / self.counts[group]


# ----------------------------------------------------------------------------- main
def extract_tft_interpretation(
    model,
    series,
    future_covariates=None,
    past_covariates=None,
    n=None,
    indices=None,
    group_labels=None,
    target_names=None,
    past_names=None,
    future_names=None,
    static_names=None,
    batch_size=None,
    verbose=True,
    accelerator=None,
):
    """
    series / future_covariates : any Sequence of TimeSeries (your lazy sequences work).
    indices      : which series to run (use stratified_indices). Default: all (slow at 100K).
    group_labels : one label per entry of `series` (full length, not just `indices`).
    n            : forecast horizon, defaults to model.output_chunk_length.
    """
    n = n or model.output_chunk_length
    icl, ocl = model.input_chunk_length, model.output_chunk_length
    if n != ocl:
        raise ValueError(f"n={n} != output_chunk_length={ocl}: auto-regressive rollout would fire the "
                         f"attention layer several times per series. Use n == output_chunk_length.")

    if indices is None:
        indices = list(range(len(series)))
    indices = list(indices)
    labels = (np.asarray(group_labels, dtype=object)[indices] if group_labels is not None
              else np.array(["ALL"] * len(indices), dtype=object))

    tm = model.model                       # the underlying _TFTModule
    tm.eval()

    raw = {
        "encoder": _vsn_var_names(tm.encoder_vsn, tm, "encoder_variables"),
        "decoder": _vsn_var_names(tm.decoder_vsn, tm, "decoder_variables"),
        "static": _vsn_var_names(tm.static_covariates_vsn, tm, "static_variables"),
    }
    if verbose:
        for k, v in raw.items():
            print(f"[{k}] raw variable names ({len(v)}): {v[:6]}{' ...' if len(v) > 6 else ''}")
    pretty = {k: _pretty_names(v, target_names, past_names, future_names, static_names)
              for k, v in raw.items()}

    acc = _Accumulator()
    state = {"offset": 0, "seen_batches": defaultdict(int)}
    sum_check = {"max_dev": 0.0}

    def batch_groups(B, kind):
        # each hook kind advances its own cursor so the three hooks stay aligned
        start = state["seen_batches"][kind]
        state["seen_batches"][kind] += B
        return labels[start:start + B]

    def make_vsn_hook(kind):
        V = len(raw[kind])

        def hook(_m, _inp, out):
            w = out[1] if isinstance(out, (tuple, list)) else out
            w = w.detach().float().cpu()
            B = w.shape[0]
            w = w.reshape(B, -1, V)                         # (B, T, V); static -> (B, 1, V)
            sum_check["max_dev"] = max(sum_check["max_dev"],
                                       float((w.sum(-1) - 1).abs().max()))
            arr = w.numpy() if kind != "static" else w.squeeze(1).numpy()
            acc.add(kind, arr, batch_groups(B, kind))
        return hook

    def attn_hook(_m, _inp, out):
        w = out[1] if isinstance(out, (tuple, list)) else out
        w = w.detach().float().cpu()
        while w.ndim > 3:                                   # average heads if they are exposed
            w = w.mean(dim=1)
        B = w.shape[0]
        acc.add("attention", w.numpy(), batch_groups(B, "attention"))

    handles = [
        tm.encoder_vsn.register_forward_hook(make_vsn_hook("encoder")),
        tm.decoder_vsn.register_forward_hook(make_vsn_hook("decoder")),
        tm.static_covariates_vsn.register_forward_hook(make_vsn_hook("static")),
        tm.multihead_attn.register_forward_hook(attn_hook),
    ]

    # accelerator="cpu" gives a readable IndexError instead of an opaque CUDA assert
    trainer = None
    if accelerator is not None:
        import pytorch_lightning as pl
        trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False,
                             enable_checkpointing=False, enable_model_summary=False,
                             enable_progress_bar=verbose)

    try:
        sub_series = [series[i] for i in indices]
        sub_future = None if future_covariates is None else [future_covariates[i] for i in indices]
        sub_past = None if past_covariates is None else [past_covariates[i] for i in indices]
        model.predict(
            n=n,
            series=sub_series,
            future_covariates=sub_future,
            past_covariates=sub_past,
            num_samples=1,          # weights do not depend on sampling; 1 keeps it fast
            batch_size=batch_size,
            verbose=verbose,
            trainer=trainer,
        )
    finally:
        for h in handles:
            h.remove()

    # ---------------- integrity checks (fail loudly rather than return plausible garbage)
    total = len(indices)
    if acc.counts["ALL"] != total:
        raise RuntimeError(f"Hook saw {acc.counts['ALL']} series but {total} were requested. "
                           f"Darts batched/ordered them differently than assumed.")
    for k, c in state["seen_batches"].items():
        if c != total:
            raise RuntimeError(f"'{k}' hook saw {c} series, expected {total}.")
    if sum_check["max_dev"] > 1e-2:
        print(f"WARNING: VSN weights deviate from summing to 1 by {sum_check['max_dev']:.3g}; "
              f"check tensor layout assumptions.")

    # ---------------- assemble results
    results = {}
    for g in acc.sums["attention"].keys():
        enc_t = acc.mean("encoder", g)         # (ICL, V_enc)
        dec_t = acc.mean("decoder", g)         # (OCL, V_dec)
        sta = acc.mean("static", g)            # (V_static,)
        att = acc.mean("attention", g)         # (OCL queries, ICL+OCL keys)

        enc_df = pd.DataFrame(enc_t, index=np.arange(-enc_t.shape[0], 0), columns=pretty["encoder"])
        dec_df = pd.DataFrame(dec_t, index=np.arange(1, dec_t.shape[0] + 1), columns=pretty["decoder"])
        enc_imp = enc_df.mean(axis=0).sort_values(ascending=False)
        dec_imp = dec_df.mean(axis=0).sort_values(ascending=False)
        sta_imp = pd.Series(sta, index=pretty["static"]).sort_values(ascending=False)

        k_len = att.shape[1]
        att_df = pd.DataFrame(
            att.T,
            index=np.arange(-(k_len - att.shape[0]), att.shape[0]),   # -ICL..-1 encoder, 0..OCL-1 decoder
            columns=[f"h{i + 1}" for i in range(att.shape[0])],
        )
        results[g] = {
            "n_series": int(acc.counts[g]),
            "encoder_importance": enc_imp,
            "decoder_importance": dec_imp,
            "static_importance": sta_imp,
            "encoder_time": enc_df,          # variable weight at each encoder step
            "decoder_time": dec_df,          # variable weight at each horizon step
            "attention": att_df,
            "attention_by_position": att_df.mean(axis=1),
            # a weight of 1/n_vars means "no preference"; lift = weight * n_vars
            "encoder_lift": enc_imp * len(enc_imp),
            "decoder_lift": dec_imp * len(dec_imp),
        }
    return results


# ----------------------------------------------------------------------------- plots / io
def plot_importance(imp: pd.Series, top=20, title="Variable importance", save_path=None):
    import matplotlib.pyplot as plt
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values)
    ax.axvline(1 / len(imp), color="red", ls="--", lw=1, label="uniform (1/n)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_attention(att: pd.DataFrame, title="Attention", save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(att.values, aspect="auto", cmap="viridis")
    step = max(1, len(att) // 12)
    ax.set_yticks(range(0, len(att), step))
    ax.set_yticklabels(att.index[::step])
    ax.set_xlabel("Forecast horizon step")
    ax.set_ylabel("Position in window (negative = encoder, >=0 = decoder)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_var_time(df: pd.DataFrame, top=15, title="Variable weight over time", save_path=None):
    """Heatmap of the top variables (by mean weight) across the encoder or decoder steps."""
    import matplotlib.pyplot as plt
    cols = df.mean().sort_values(ascending=False).head(top).index
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(cols) + 2))
    im = ax.imshow(df[cols].T.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def check_static_indices(model, series, indices=None, max_series=2000):
    """
    Pre-flight check for 'CUDA device-side assert' errors: compares the integer codes in each
    series' static covariates against the embedding table sizes stored in the checkpoint.
    Any column with min < 0 or max >= num_embeddings will crash an embedding lookup.
    Assumes embedding tables appear in the same order as the static columns (check the printout).
    """
    indices = list(range(len(series))) if indices is None else list(indices)
    indices = indices[:max_series]
    rows = [series[i].static_covariates.iloc[[0]] for i in indices]
    sc = pd.concat(rows, ignore_index=True)

    emb = [(n, m.num_embeddings) for n, m in model.model.named_modules()
           if isinstance(m, torch.nn.Embedding)]
    print(f"{len(emb)} embedding tables in checkpoint; {sc.shape[1]} static columns in data\n")
    bad = False
    for pos, col in enumerate(sc.columns):
        lo, hi = float(sc[col].min()), float(sc[col].max())
        size = emb[pos][1] if pos < len(emb) else None
        flag = ""
        if size is not None and (lo < 0 or hi >= size):
            flag, bad = "   <-- OUT OF RANGE", True
        print(f"{col:24s} min={lo:8.0f} max={hi:8.0f}  embedding rows={size}{flag}")
    if bad:
        print("\nAt least one column would index outside its embedding table. The static "
              "transformer / cache does not match this checkpoint.")
    return sc


def save_results(results, prefix="tft_interp"):
    """One CSV per (group, quantity). Group names are sanitised for Windows paths."""
    for g, r in results.items():
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(g))
        for k in ("encoder_importance", "decoder_importance", "static_importance"):
            r[k].to_csv(f"{prefix}_{tag}_{k}.csv", header=["weight"])
        for k in ("encoder_time", "decoder_time", "attention"):
            r[k].to_csv(f"{prefix}_{tag}_{k}.csv")
