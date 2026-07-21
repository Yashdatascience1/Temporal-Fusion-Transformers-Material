import psutil, threading, time, os

_proc = psutil.Process(os.getpid())
_stop = threading.Event()

def _log_ram(interval=300):
    peak = 0
    while not _stop.is_set():
        rss  = _proc.memory_info().rss / 1e9
        kids = sum(c.memory_info().rss for c in _proc.children(recursive=True)) / 1e9
        vm   = psutil.virtual_memory()
        peak = max(peak, rss + kids)
        print(f"[{time.strftime('%H:%M:%S')}] main: {rss:.1f} GB | "
              f"workers: {kids:.1f} GB | total: {rss+kids:.1f} GB (peak {peak:.1f}) | "
              f"system: {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB", flush=True)
        _stop.wait(interval)

_stop.clear()
threading.Thread(target=_log_ram, daemon=True).start()
print("RAM logger started.")

try:
    model.fit(
        series=train_seq,
        future_covariates=train_cov_seq,

        val_series=val_seq,
        val_future_covariates=val_cov_seq,

        max_samples_per_ts=50,

        dataloader_kwargs={
            "num_workers": 4,
            "persistent_workers": True,
            "pin_memory": True,
        },
        verbose=True,
    )
finally:
    _stop.set()
    print("RAM logger stopped.")