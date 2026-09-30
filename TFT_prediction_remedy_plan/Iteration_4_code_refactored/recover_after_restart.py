# Standalone recovery cell: run in a fresh kernel (after restart) from the SAME folder as the notebooks.
# Rebuilds what the end of the fit cell would have written, using only files on disk.
from pathlib import Path
import json, hashlib
import torch, darts
import pytorch_lightning as pl
from darts.models import TFTModel


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


PROJECT_DIR = Path.cwd()
FIT_DIR_OVERRIDE = None   # e.g. Path(r"C:\...\fit_20260925_123456_abcd1234") if the auto-pick below is wrong

RUN_DIR = Path(read_json(PROJECT_DIR / "iteration3_active_run.json")["run_dir"])
cfg = read_json(RUN_DIR / "config.json")
dm = read_json(RUN_DIR / "data_manifest.json")
ICL, OCL = cfg["icl"], cfg["ocl"]

FIT_DIR = Path(FIT_DIR_OVERRIDE) if FIT_DIR_OVERRIDE else sorted(
    p for p in RUN_DIR.glob("fit_*") if p.is_dir()
)[-1]
MODEL_NAME = "daily_tft_" + FIT_DIR.name
WORK_DIR = FIT_DIR / "darts_logs"
records = read_json(FIT_DIR / "cache_manifest.json")
CSV_LOG_DIR = sorted((FIT_DIR / "training_metrics").glob("version_*"))[-1]
TB_LOG_DIR = sorted((FIT_DIR / "tensorboard").glob("version_*"))[-1]
print("Using fit folder:", FIT_DIR)
print("Checkpoints on disk:", sorted(p.name for p in WORK_DIR.rglob("*.ckpt")))

best = TFTModel.load_from_checkpoint(
    model_name=MODEL_NAME, work_dir=str(WORK_DIR), best=True, map_location="cpu"
)
assert best.input_chunk_length == ICL and best.output_chunk_length == OCL
embedding_count = sum(p.numel() for p in best.model.input_embeddings.parameters())
if embedding_count <= 0:
    raise RuntimeError("Categorical embeddings were not built")
files = [
    "cache_manifest.json", "static_raw.parquet", "static_transformer.pkl", "static_encoded.pkl",
    "shared_cov.pkl", "training_settings.json", "sample_weight.parquet",
]
files += [
    str(p.relative_to(FIT_DIR)) for p in WORK_DIR.rglob("*")
    if p.is_file() and (p.suffix == ".ckpt" or p.name.endswith(".pth.tar"))
]
bundle = dict(
    run_id=cfg["run_id"], model_name=MODEL_NAME, work_dir=str(WORK_DIR.resolve()),
    config_hash=dm["config_hash"], calendar_hash=dm["calendar_hash"],
    versions={"darts": darts.__version__, "torch": torch.__version__, "lightning": pl.__version__},
    file_hashes={name: digest(FIT_DIR / name) for name in files},
    series_count=len(records), gpu=torch.cuda.get_device_name(0),
    embedding_parameters=embedding_count,
    csv_log_dir=str(CSV_LOG_DIR), tensorboard_log_dir=str(TB_LOG_DIR),
)
write_json(FIT_DIR / "bundle.json", bundle)
write_json(RUN_DIR / "active_fit.json",
           {"fit_dir": str(FIT_DIR.resolve()), "bundle_hash": digest(FIT_DIR / "bundle.json")})
print("Bundle written. Best checkpoint ready:", FIT_DIR)
