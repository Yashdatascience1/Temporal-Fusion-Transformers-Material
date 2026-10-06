import os, torch
from darts.models import TFTModel

logs = os.path.join(r"<Modelling_code folder>", "darts_logs")
target = [1189, 14, 14, 3, 3, 5, 32, 868, 5, 7, 2]       # current data categories + 1, in column order
print("target:", target, "\n")
for name in sorted(os.listdir(logs)):
    try:
        m = TFTModel.load_from_checkpoint(model_name=name, work_dir=logs, best=True, map_location="cpu")
        rows = [e.num_embeddings for e in m.model.modules() if isinstance(e, torch.nn.Embedding)]
        print(name, rows, "<-- MATCH" if rows == target else "")
    except Exception as e:
        print(name, "could not load:", type(e).__name__)