import torch

m = ctx.model

print("Model name:", m.model_name)
print("Work directory:", m.work_dir)
print(
    "Saved embedding configuration:",
    m.model_params.get("categorical_embedding_sizes"),
)
print(
    "Categorical variables:",
    m.model.categorical_static_variables,
)
print(
    "Numeric variables:",
    m.model.numeric_static_variables,
)
print(
    "Actual embedding tables:",
    [
        (name, layer.num_embeddings, layer.embedding_dim)
        for name, layer in m.model.named_modules()
        if isinstance(layer, torch.nn.Embedding)
    ],
)
print(
    "Encoded static columns:",
    ctx.val_seq[0].static_covariates.columns.tolist(),
)