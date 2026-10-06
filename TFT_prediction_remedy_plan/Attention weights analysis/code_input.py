import darts

print("Darts version:", darts.__version__)

print(
    "Wrapper embedding configuration:",
    getattr(m, "categorical_embedding_sizes", None),
)

print(
    "Network embedding configuration:",
    getattr(m.model, "categorical_embedding_sizes", None),
)

saved_static = getattr(m, "static_covariates", None)
print(
    "Static columns stored on the fitted model:",
    None if saved_static is None else saved_static.columns.tolist(),
)