"""
tft_checks.py -- verifies that static categorical covariates reach the TFT through
embedding tables, not as plain numbers.

Kept in a .py file (not inside the notebook) for one practical reason: the training
guard below is stored with the model's settings when Darts saves a checkpoint. A class
defined inside a notebook cannot be found when the checkpoint is loaded from a different
notebook (the prediction notebook); a class in an importable file can.

How Darts names things: inside the network the static columns are called
`static_covariate_0`, `static_covariate_1`, ... in the same order as the columns of the
series' static-covariate table. The functions below translate those back to real names.
"""
import pandas as pd
from pytorch_lightning.callbacks import Callback


def embedding_report(pl_module, static_covariates, embedding_sizes):
    """One row per static column: how the network actually treats it."""
    categorical = set(pl_module.categorical_static_variables)
    numeric = set(pl_module.numeric_static_variables)
    tables = pl_module.input_embeddings.embeddings
    rows = []
    for i, col in enumerate(static_covariates):
        internal = f"static_covariate_{i}"
        table = tables[internal] if internal in tables else None
        rows.append(dict(
            column=col,
            internal_name=internal,
            treated_as=("EMBEDDING" if internal in categorical
                        else "NUMBER (wrong)" if internal in numeric else "MISSING"),
            table_rows=table.num_embeddings if table is not None else 0,
            expected_rows=embedding_sizes[col][0],
            vector_size=table.embedding_dim if table is not None else 0,
            expected_vector_size=embedding_sizes[col][1],
        ))
    return pd.DataFrame(rows)


def assert_embeddings(pl_module, static_covariates, embedding_sizes):
    """Raise unless every static column has an embedding table of the configured size."""
    rep = embedding_report(pl_module, static_covariates, embedding_sizes)
    bad = rep[(rep.treated_as != "EMBEDDING")
              | (rep.table_rows != rep.expected_rows)
              | (rep.vector_size != rep.expected_vector_size)]
    n_params = sum(p.numel() for n, p in pl_module.named_parameters() if "input_embeddings" in n)
    if len(bad) or n_params == 0:
        raise RuntimeError(
            "Static categorical covariates are NOT going through embedding tables.\n"
            f"{bad.to_string(index=False)}\nEmbedding parameters: {n_params}\n"
            "Most likely cause: this TFTModel object was built twice (an interrupted or failed fit, "
            "or lr_find, on the same object). Darts replaces `categorical_embedding_sizes` with "
            "internal names after the first build, so a second build matches nothing and silently "
            "treats every category as a number. Create a NEW TFTModel and fit again.")
    return rep, n_params


class EmbeddingGate(Callback):
    """Stops training before the first batch if the embeddings were not built."""

    def __init__(self, static_covariates, embedding_sizes):
        self.static_covariates = list(static_covariates)
        self.embedding_sizes = dict(embedding_sizes)

    def on_fit_start(self, trainer, pl_module):
        rep, n = assert_embeddings(pl_module, self.static_covariates, self.embedding_sizes)
        print(f"EmbeddingGate: PASSED -- {len(rep)} static columns use embedding tables "
              f"({n:,} embedding parameters).")
