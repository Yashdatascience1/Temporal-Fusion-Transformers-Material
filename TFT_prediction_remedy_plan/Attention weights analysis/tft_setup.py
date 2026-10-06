"""
tft_setup.py - rebuild everything tft_interpret.py needs, from the on-disk artifacts your
training notebook already wrote (series_cache/, shared_calendar.parquet, column_roles.json,
darts_logs/<model_name>/). No retraining, no Snowflake, no chunk parquet files.

Notebook usage:
    from tft_setup import load_everything
    ctx = load_everything(model_name="daily_tft_negbin_scooters_2026-09-10_18_02_22")
    # ctx.model, ctx.val_seq, ctx.val_cov_seq, ctx.val_statics, ctx.labels(...) etc.

Script usage (edit the CONFIG block, then):  python tft_setup.py
"""
from __future__ import annotations

import collections.abc
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import TFTModel

# ============================================================================= CONFIG
MODEL_NAME = "daily_tft_negbin_scooters_2026-09-10_18_02_22"   # checkpoint folder under darts_logs/
BASE_DIR = os.getcwd()          # folder holding column_roles.json, series_cache/, darts_logs/
WORK_DIR = BASE_DIR             # parent of darts_logs/ (where the model was trained from)

# These two MUST equal what the training run used (Section 2 of your notebook),
# otherwise val_keys will not line up with static_covariates.parquet.
DROP_NEVER_SOLD = True
DORMANT_DAYS = None

N_PER_GROUP = 100               # series per MODEL_FAMILY for the interpretation run
GROUP_COL_FOR_LABELS = "MODEL_FAMILY"
OUT_PREFIX = "tft_interp"
# =============================================================================


def safe_name(key):
    return str(key).replace("<>", "_").replace("/", "_").replace("\\", "_")


class DiskLazyTargetSequence(collections.abc.Sequence):
    """Same class as the notebook's Section 5: raw-count, 1-component series from .npz files."""

    def __init__(self, cache_dir, series_keys, static_raw, sc_transformer, target_col,
                 split="val", freq="D", cache_in_ram=True):
        self.cache_dir, self.series_keys = cache_dir, series_keys
        self.static_raw, self.sc_transformer = static_raw, sc_transformer
        self.target_col, self.split, self.freq = target_col, split, freq
        self.cache_in_ram = cache_in_ram
        self._ram = {} if cache_in_ram else None

    def __len__(self):
        return len(self.series_keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ram"] = {} if self.cache_in_ram else None
        return state

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self._ram is not None and idx in self._ram:
            return self._ram[idx]

        key = self.series_keys[idx]
        with np.load(os.path.join(self.cache_dir, f"{safe_name(key)}.npz"), allow_pickle=False) as z:
            sales = z[f"{self.split}_sales"]
            start = str(z[f"{self.split}_start"])
        times = pd.date_range(start=start, periods=len(sales), freq=self.freq)
        ts = TimeSeries.from_times_and_values(
            times, sales.reshape(-1, 1).astype(np.float32),
            columns=[self.target_col], static_covariates=self.static_raw[idx],
        )
        ts = self.sc_transformer.transform(ts)
        if self._ram is not None:
            self._ram[idx] = ts
        return ts


class SharedCovSequence(collections.abc.Sequence):
    """Returns the same in-RAM covariate TimeSeries for every index."""

    def __init__(self, shared_series, n):
        self.shared, self.n = shared_series, n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.shared for _ in range(*idx.indices(self.n))]
        if idx < 0:
            idx += self.n
        if not 0 <= idx < self.n:
            raise IndexError(idx)
        return self.shared


def load_everything(model_name=MODEL_NAME, base_dir=BASE_DIR, work_dir=WORK_DIR,
                    drop_never_sold=DROP_NEVER_SOLD, dormant_days=DORMANT_DAYS,
                    map_location="cpu", best=True):
    cache_dir = os.path.join(base_dir, "series_cache")

    # ---- roles ---------------------------------------------------------------
    with open(os.path.join(base_dir, "column_roles.json")) as f:
        roles = json.load(f)
    time_col, group_col, target_col = roles["time_col"], roles["group_col"], roles["target_col"]
    freq = roles["freq"]
    static_covariates = roles["static_covariates"]
    future_covariates = roles["future_covariates"]

    # ---- manifest + same dead-series filter as Section 2 ---------------------------
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    series_keys, has_val = manifest["series_keys"], manifest["has_val"]
    static_df = pd.read_parquet(os.path.join(cache_dir, "static_covariates.parquet"))
    assert len(static_df) == len(series_keys), "static_covariates.parquet and manifest are out of sync"

    keep = []
    for k in series_keys:
        with np.load(os.path.join(cache_dir, f"{safe_name(k)}.npz")) as z:
            s = z["train_sales"]
        if drop_never_sold and s.sum() == 0:
            keep.append(False)
        elif dormant_days and s[-dormant_days:].sum() == 0:
            keep.append(False)
        else:
            keep.append(True)
    keep = np.asarray(keep)
    series_keys = [k for k, m in zip(series_keys, keep) if m]
    has_val = [h for h, m in zip(has_val, keep) if m]
    static_df = static_df.loc[keep].reset_index(drop=True)[static_covariates].astype(str)
    static_raw = [static_df.iloc[[i]].reset_index(drop=True) for i in range(len(static_df))]

    # ---- saved artifacts from Section 6 ---------------------------------------
    shared_cov = TimeSeries.from_pickle(os.path.join(cache_dir, "shared_cov.pkl"))
    with open(os.path.join(cache_dir, "static_cov_transformer.pkl"), "rb") as f:
        sc_transformer = pickle.load(f)

    # ---- validation-strip sequences (the ones that end on FORECAST_START - 1) -------
    val_mask = list(has_val)
    val_keys = [k for k, h in zip(series_keys, val_mask) if h]
    val_statics = [s for s, h in zip(static_raw, val_mask) if h]
    val_seq = DiskLazyTargetSequence(cache_dir, val_keys, val_statics, sc_transformer,
                                     target_col, split="val", freq=freq)
    val_cov_seq = SharedCovSequence(shared_cov, len(val_keys))

    # ---- model ---------------------------------------------------------------------
    model = TFTModel.load_from_checkpoint(
        model_name=model_name, work_dir=work_dir, best=best, map_location=map_location
    )
    icl, ocl = model.input_chunk_length, model.output_chunk_length

    # ---- consistency checks: fail here, not deep inside predict() -------------------
    probe = val_seq[0]
    need_end = probe.end_time() + pd.Timedelta(days=ocl)
    if shared_cov.end_time() < need_end:
        raise ValueError(f"Calendar ends {shared_cov.end_time().date()} but forecasting from "
                         f"{probe.end_time().date()} needs it to reach {need_end.date()}.")
    if len(probe) < icl:
        raise ValueError(f"Probe series has {len(probe)} steps < input_chunk_length {icl}.")
    n_stat = probe.static_covariates.shape[1]
    if n_stat != len(static_covariates):
        raise ValueError(f"Probe has {n_stat} static covariates, roles file lists {len(static_covariates)}.")

    print(f"model            : {model_name}")
    print(f"ICL / OCL        : {icl} / {ocl}")
    print(f"series in manifest after filter : {len(series_keys):,}")
    print(f"series with val strip           : {len(val_keys):,}")
    print(f"history ends     : {probe.end_time().date()}  (forecast starts next day)")
    print(f"calendar covers  : {shared_cov.start_time().date()} -> {shared_cov.end_time().date()}")

    return SimpleNamespace(
        model=model, val_seq=val_seq, val_cov_seq=val_cov_seq,
        val_keys=val_keys, val_statics=val_statics, shared_cov=shared_cov,
        sc_transformer=sc_transformer, roles=roles,
        target_col=target_col, static_covariates=static_covariates,
        future_covariates=future_covariates, time_col=time_col, group_col=group_col,
        icl=icl, ocl=ocl,
        labels=lambda col=GROUP_COL_FOR_LABELS: [s[col].iloc[0] for s in val_statics],
    )


if __name__ == "__main__":
    from tft_interpret import (extract_tft_interpretation, plot_attention, plot_importance,
                               save_results, stratified_indices)

    ctx = load_everything()
    labels = ctx.labels()
    idx = stratified_indices(labels, n_per_group=N_PER_GROUP)
    print(f"Running on {len(idx):,} series across {len(set(labels))} groups")

    out = extract_tft_interpretation(
        ctx.model, ctx.val_seq, future_covariates=ctx.val_cov_seq,
        indices=idx, group_labels=labels,
        target_names=[ctx.target_col],
        future_names=ctx.future_covariates,
        static_names=ctx.static_covariates,
    )
    save_results(out, OUT_PREFIX)
    plot_importance(out["ALL"]["decoder_importance"], title="Decoder importance (ALL)",
                    save_path=f"{OUT_PREFIX}_decoder_importance.png")
    plot_importance(out["ALL"]["encoder_importance"], title="Encoder importance (ALL)",
                    save_path=f"{OUT_PREFIX}_encoder_importance.png")
    plot_attention(out["ALL"]["attention"], title="Attention (ALL)",
                   save_path=f"{OUT_PREFIX}_attention.png")
    print("Saved CSVs and PNGs with prefix:", OUT_PREFIX)
