"""
tft_attention.py - reusable attention-weight extraction for any trained Darts TFTModel.

Usage (notebook):
    from tft_attention import get_tft_attention, plot_attention_heatmap

    res = get_tft_attention(
        model,                       # trained TFTModel, or a path string to model.save() output
        series=one_target_series,    # ONE TimeSeries (materialise it from your lazy sequence first)
        future_covariates=shared_cov,
        past_covariates=None,
    )
    res["attention"]       # DataFrame: rows = relative time step (input + output), cols = horizon step
    res["attention_mean"]  # Series: attention averaged over horizon steps, per relative time step
    res["encoder_importance"], res["decoder_importance"], res["static_importance"]  # VSN weights (may be None)
    plot_attention_heatmap(res["attention"])

Two extraction paths:
  method="explainer" -> Darts TFTExplainer (high level, also returns VSN weights)
  method="hook"      -> forward hook on the interpretable multi-head attention layer
                        (fallback if the explainer fails on your Darts version / data setup)
  method="auto"      -> try explainer, fall back to hook
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from darts.models import TFTModel


# --------------------------------------------------------------------------- helpers
def _load_model(model):
    if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
        return TFTModel.load(str(model))
    return model


def _first(x):
    """Darts sometimes returns a list (one entry per series); take the first."""
    return x[0] if isinstance(x, (list, tuple)) else x


def _to_df(ts):
    return None if ts is None else _first(ts).pd_dataframe()


# --------------------------------------------------------------------------- path 1: TFTExplainer
def _via_explainer(model, series, past_covariates, future_covariates, n):
    from darts.explainability import TFTExplainer

    explainer = TFTExplainer(
        model,
        background_series=series,
        background_past_covariates=past_covariates,
        background_future_covariates=future_covariates,
    )
    result = explainer.explain()

    out = {
        "attention": _to_df(result.get_attention()),
        "encoder_importance": _to_df(result.get_encoder_importance()),
        "decoder_importance": _to_df(result.get_decoder_importance()),
        "static_importance": _to_df(result.get_static_covariates_importance()),
        "raw_result": result,
        "explainer": explainer,
    }
    return out


# --------------------------------------------------------------------------- path 2: forward hook
def _find_attn_module(torch_module: torch.nn.Module):
    for name, m in torch_module.named_modules():
        if "multiheadattention" in type(m).__name__.lower():
            return name, m
    raise RuntimeError(
        "No multi-head attention module found. Inspect model.model.named_modules() "
        "and adapt _find_attn_module."
    )


def _via_hook(model, series, past_covariates, future_covariates, n):
    captured = []

    _, attn_mod = _find_attn_module(model.model)

    def hook(_mod, _inp, output):
        # Darts' interpretable attention returns (attn_out, attn_weights)
        w = output[1] if isinstance(output, (tuple, list)) else output
        captured.append(w.detach().cpu())

    handle = attn_mod.register_forward_hook(hook)
    try:
        model.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=1,
            verbose=False,
        )
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Hook never fired - predict() did not reach the attention layer.")

    w = torch.cat(captured, dim=0)  # (batch, [heads,] query_len, key_len) - shape varies by version
    # Collapse everything except the last two dims (query=decoder steps, key=input+output steps)
    while w.ndim > 3:
        w = w.mean(dim=1)
    w = w.mean(dim=0).numpy()  # (query_len, key_len)

    q_len, k_len = w.shape
    # TFT attends over [encoder steps + decoder steps]; queries are the decoder steps only
    # (if q_len != n your Darts version masks differently - check the shape printed below)
    attn = pd.DataFrame(
        w.T,
        index=pd.Index(range(-(k_len - q_len), q_len), name="relative_step"),
        columns=[f"horizon_{i + 1}" for i in range(q_len)],
    )
    return {
        "attention": attn,
        "encoder_importance": None,
        "decoder_importance": None,
        "static_importance": None,
    }


# --------------------------------------------------------------------------- public API
def get_tft_attention(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    n: int | None = None,
    method: str = "auto",
    save_prefix: str | None = None,
):
    """
    Extract attention weights from a trained Darts TFTModel.

    series : a single darts TimeSeries (NOT your lazy sequence). For 117K series, loop over a
             sample; attention is per-window, so one series = one picture.
    n      : forecast horizon; defaults to model.output_chunk_length.
    """
    model = _load_model(model)
    n = n or model.output_chunk_length
    model.model.eval()

    errors = {}
    out = None
    for name, fn in (("explainer", _via_explainer), ("hook", _via_hook)):
        if method not in ("auto", name):
            continue
        try:
            out = fn(model, series, past_covariates, future_covariates, n)
            out["method_used"] = name
            break
        except Exception as e:  # noqa: BLE001
            errors[name] = repr(e)
            if method != "auto":
                raise

    if out is None:
        raise RuntimeError(f"All extraction paths failed: {errors}")

    att = out["attention"]
    out["attention_mean"] = att.mean(axis=1)
    out["errors_from_skipped_paths"] = errors

    if save_prefix:
        att.to_csv(f"{save_prefix}_attention.csv")
        for k in ("encoder_importance", "decoder_importance", "static_importance"):
            if out.get(k) is not None:
                out[k].to_csv(f"{save_prefix}_{k}.csv")
    return out


def plot_attention_heatmap(attention: pd.DataFrame, title="TFT attention", save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(attention.values, aspect="auto", cmap="viridis")
    ax.set_xlabel("Forecast horizon step")
    ax.set_ylabel("Position in window (encoder -> decoder)")
    ax.set_yticks(range(0, len(attention), max(1, len(attention) // 10)))
    ax.set_yticklabels(attention.index[:: max(1, len(attention) // 10)])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def attention_across_series(model, series_list, past_covariates=None, future_covariates=None, **kw):
    """Mean attention over a sample of series (same input/output chunk lengths required)."""
    mats = []
    for s in series_list:
        r = get_tft_attention(
            model, s, past_covariates=past_covariates, future_covariates=future_covariates, **kw
        )
        mats.append(r["attention"].values)
    ref = r["attention"]
    return pd.DataFrame(np.mean(mats, axis=0), index=ref.index, columns=ref.columns)
